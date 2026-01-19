"""
Universal ASCII File Parser for Well Log Data
Handles generic ASCII, CSV, TXT, and other delimited formats with intelligent
delimiter and depth column detection.
"""

import pandas as pd
import numpy as np
import io
import re
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Dict, Any
from enum import Enum


class DelimiterType(Enum):
    """Supported delimiter types for ASCII files"""
    COMMA = ','
    TAB = '\t'
    SPACE = ' '
    SEMICOLON = ';'
    PIPE = '|'
    WHITESPACE = 'whitespace'  # Multiple spaces
    UNKNOWN = 'unknown'


@dataclass
class AsciiFileMetadata:
    """Stores detection results from ASCII file analysis"""
    detected_delimiter: str
    delimiter_type: DelimiterType
    header_row_index: int
    skip_rows: int
    depth_column: Optional[str]
    depth_column_index: Optional[int]
    depth_unit: str
    num_data_columns: int
    num_data_rows: int
    column_names: List[str]
    confidence_score: float
    encoding: str
    comment_char: Optional[str]
    null_values: List[float]
    detection_notes: List[str]


@dataclass 
class DepthDetectionResult:
    """Result of depth column detection"""
    column_name: Optional[str]
    column_index: Optional[int]
    detection_method: str  # 'keyword', 'monotonic', 'unit', 'manual'
    confidence: float
    detected_unit: str
    is_monotonic: bool
    is_increasing: bool
    depth_range: Tuple[float, float]
    step_size: Optional[float]


class AsciiFormatDetector:
    """
    Detects file format characteristics for ASCII well log files.
    
    Handles:
    - Delimiter detection (comma, tab, space, semicolon)
    - Header row detection (automatic or configurable)
    - Comment line detection (# or other prefixes)
    - Encoding detection (UTF-8, Latin-1, etc.)
    """
    
    COMMON_DELIMITERS = [',', '\t', ';', '|', ' ']
    COMMON_COMMENT_CHARS = ['#', '%', '!', '//', '--']
    COMMON_ENCODINGS = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    def __init__(self):
        self.notes = []
    
    def detect_encoding(self, raw_bytes: bytes) -> str:
        """Detect file encoding by testing different encodings."""
        for encoding in self.COMMON_ENCODINGS:
            try:
                raw_bytes.decode(encoding)
                return encoding
            except (UnicodeDecodeError, LookupError):
                continue
        return 'utf-8'  # Default fallback
    
    def detect_comment_char(self, lines: List[str]) -> Optional[str]:
        """Detect the comment character used in the file."""
        for char in self.COMMON_COMMENT_CHARS:
            comment_lines = sum(1 for line in lines[:20] if line.strip().startswith(char))
            if comment_lines >= 2:
                self.notes.append(f"Detected comment character: '{char}'")
                return char
        return None
    
    def detect_delimiter(self, lines: List[str], skip_comment_lines: bool = True) -> Tuple[str, DelimiterType]:
        """
        Detect the delimiter used in the file.
        
        Returns:
            Tuple of (delimiter_string, DelimiterType)
        """
        # Filter out empty and comment lines for analysis
        data_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if skip_comment_lines and any(stripped.startswith(c) for c in self.COMMON_COMMENT_CHARS):
                continue
            # Skip potential header lines that look like well log headers
            if any(kw in stripped.upper() for kw in ['~', 'VERS', 'WELL:', 'COMP:', 'CURVE']):
                continue
            data_lines.append(stripped)
            if len(data_lines) >= 10:
                break
        
        if not data_lines:
            return ',', DelimiterType.COMMA  # Default
        
        # Count delimiter occurrences
        delimiter_counts = {}
        for delim in self.COMMON_DELIMITERS:
            counts = [line.count(delim) for line in data_lines]
            if len(set(counts)) == 1 and counts[0] > 0:
                # All lines have same count - good sign of consistent delimiter
                delimiter_counts[delim] = (counts[0], 1.0)  # (count, consistency)
            elif counts[0] > 0:
                variance = np.var(counts) if len(counts) > 1 else 0
                consistency = 1.0 / (1.0 + variance) if variance > 0 else 0.5
                delimiter_counts[delim] = (max(counts), consistency)
        
        # Check for whitespace-delimited (multiple spaces)
        whitespace_counts = []
        for line in data_lines:
            parts = line.split()
            if len(parts) > 1:
                whitespace_counts.append(len(parts) - 1)
        
        if whitespace_counts:
            consistency = 1.0 if len(set(whitespace_counts)) == 1 else 0.7
            delimiter_counts['whitespace'] = (max(whitespace_counts), consistency)
        
        # Select best delimiter
        if not delimiter_counts:
            return ',', DelimiterType.COMMA
        
        best_delim = max(delimiter_counts.items(), key=lambda x: x[1][0] * x[1][1])
        delim_str = best_delim[0]
        
        if delim_str == 'whitespace':
            self.notes.append("Detected whitespace-delimited format")
            return delim_str, DelimiterType.WHITESPACE
        
        delim_type = {
            ',': DelimiterType.COMMA,
            '\t': DelimiterType.TAB,
            ';': DelimiterType.SEMICOLON,
            '|': DelimiterType.PIPE,
            ' ': DelimiterType.SPACE
        }.get(delim_str, DelimiterType.UNKNOWN)
        
        self.notes.append(f"Detected delimiter: '{delim_str}' ({delim_type.name})")
        return delim_str, delim_type
    
    def detect_header_row(self, lines: List[str], delimiter: str) -> Tuple[int, int]:
        """
        Detect which row contains column headers and how many rows to skip.
        
        Returns:
            Tuple of (header_row_index, skip_rows)
            header_row_index: -1 if no header found (use default column names)
        """
        skip_count = 0
        
        for i, line in enumerate(lines[:30]):  # Check first 30 lines
            stripped = line.strip()
            
            if not stripped:
                skip_count = i + 1
                continue
            
            # Skip comment lines
            if any(stripped.startswith(c) for c in self.COMMON_COMMENT_CHARS):
                skip_count = i + 1
                continue
            
            # Skip LAS-style header lines
            if stripped.startswith('~') or ':' in stripped[:30]:
                skip_count = i + 1
                continue
            
            # Check if this line looks like a header (text) vs data (numbers)
            if delimiter == 'whitespace':
                parts = stripped.split()
            else:
                parts = stripped.split(delimiter)
            
            if len(parts) < 2:
                skip_count = i + 1
                continue
            
            # Count how many parts are numeric
            numeric_count = 0
            for part in parts:
                part = part.strip()
                try:
                    float(part.replace('E', 'e').replace('D', 'e'))
                    numeric_count += 1
                except ValueError:
                    pass
            
            numeric_ratio = numeric_count / len(parts)
            
            if numeric_ratio < 0.5:
                # More text than numbers - likely header row
                self.notes.append(f"Header detected at row {i}")
                return i, i + 1
            else:
                # This looks like data - no header found
                self.notes.append(f"No header row detected, data starts at row {i}")
                return -1, i
        
        return -1, skip_count


class DepthColumnDetector:
    """
    Intelligent depth column identification for well log data.
    
    Detection methods:
    1. Keyword matching: DEPTH, DEPT, MD, TVD, TVDSS, TVDKB
    2. Monotonic sequence detection (strictly increasing/decreasing)
    3. Unit validation (m, ft, M, FT patterns)
    4. Physical range validation (0-20000m for wells)
    """
    
    # Depth-related keywords (order by priority)
    DEPTH_KEYWORDS = [
        'DEPTH', 'DEPT', 'MD', 'TVD', 'TVDSS', 'TVDKB', 'MDKB', 'MDRT',
        'MEASURED_DEPTH', 'MEAS_DEPTH', 'TRUE_VERTICAL_DEPTH', 'AHD',
        'Z', 'ELEVATION', 'ELEV', 'PROFUNDIDAD', 'TIEFE'
    ]
    
    # Unit patterns for depth
    DEPTH_UNIT_PATTERNS = {
        'm': ['m', 'meter', 'meters', 'metre', 'metres'],
        'ft': ['ft', 'feet', 'foot', 'f']
    }
    
    def __init__(self):
        self.notes = []
    
    def detect(self, df: pd.DataFrame, column_units: Optional[Dict[str, str]] = None) -> DepthDetectionResult:
        """
        Detect the depth column in a DataFrame.
        
        Args:
            df: DataFrame with well log data
            column_units: Optional dictionary mapping column names to units
            
        Returns:
            DepthDetectionResult with detection details
        """
        candidates = []
        
        for col_idx, col_name in enumerate(df.columns):
            score = 0.0
            method = 'unknown'
            
            col_upper = col_name.upper().strip()
            
            # Method 1: Keyword matching (highest priority)
            for i, keyword in enumerate(self.DEPTH_KEYWORDS):
                if keyword == col_upper or keyword in col_upper:
                    score = 1.0 - (i * 0.02)  # Earlier keywords get higher score
                    method = 'keyword'
                    break
            
            # Method 2: Check unit if available
            if column_units and col_name in column_units:
                unit = column_units[col_name].lower()
                for unit_type, patterns in self.DEPTH_UNIT_PATTERNS.items():
                    if any(p in unit for p in patterns):
                        score = max(score, 0.7)
                        if method == 'unknown':
                            method = 'unit'
            
            # Method 3: Check if column is monotonic (depth should be monotonic)
            if score < 0.5:  # Only check if not already confident
                try:
                    data = df[col_name].dropna()
                    if len(data) > 10:
                        is_increasing = (data.diff().dropna() > 0).mean() > 0.95
                        is_decreasing = (data.diff().dropna() < 0).mean() > 0.95
                        
                        if is_increasing or is_decreasing:
                            # Check physical range (typical well depths)
                            data_min, data_max = data.min(), data.max()
                            if 0 <= data_min < data_max <= 30000:  # Max depth 30km
                                score = max(score, 0.6)
                                if method == 'unknown':
                                    method = 'monotonic'
                except (TypeError, ValueError):
                    pass
            
            if score > 0:
                candidates.append((col_name, col_idx, score, method))
        
        if not candidates:
            # Fallback: use first column if numeric and monotonic
            first_col = df.columns[0]
            try:
                data = pd.to_numeric(df[first_col], errors='coerce').dropna()
                if len(data) > 0:
                    is_mono = (data.diff().dropna() > 0).mean() > 0.9
                    if is_mono:
                        self.notes.append(f"Fallback: using first column '{first_col}' as depth")
                        candidates.append((first_col, 0, 0.4, 'fallback'))
            except:
                pass
        
        if not candidates:
            return DepthDetectionResult(
                column_name=None,
                column_index=None,
                detection_method='none',
                confidence=0.0,
                detected_unit='unknown',
                is_monotonic=False,
                is_increasing=False,
                depth_range=(0, 0),
                step_size=None
            )
        
        # Select best candidate
        candidates.sort(key=lambda x: x[2], reverse=True)
        best = candidates[0]
        col_name, col_idx, confidence, method = best
        
        # Get depth column statistics
        depth_data = pd.to_numeric(df[col_name], errors='coerce').dropna()
        is_increasing = True
        step_size = None
        depth_range = (0, 0)
        
        if len(depth_data) > 1:
            diff = depth_data.diff().dropna()
            is_increasing = diff.mean() > 0
            depth_range = (float(depth_data.min()), float(depth_data.max()))
            step_size = float(abs(diff.median()))
        
        # Detect unit from column name or data range
        unit = 'unknown'
        col_lower = col_name.lower()
        if 'm' in col_lower and 'ft' not in col_lower:
            unit = 'm'
        elif 'ft' in col_lower or 'feet' in col_lower:
            unit = 'ft'
        elif depth_range[1] > 0:
            # Heuristic: if max > 1000, likely feet if > 10000
            unit = 'ft' if depth_range[1] > 6000 else 'm'
        
        self.notes.append(f"Depth column detected: '{col_name}' (confidence: {confidence:.2f}, method: {method})")
        
        return DepthDetectionResult(
            column_name=col_name,
            column_index=col_idx,
            detection_method=method,
            confidence=confidence,
            detected_unit=unit,
            is_monotonic=True,
            is_increasing=is_increasing,
            depth_range=depth_range,
            step_size=step_size
        )


def load_ascii_file(
    file_obj: Union[str, bytes, io.IOBase],
    delimiter: Optional[str] = None,
    header_row: Optional[int] = None,
    skip_rows: Optional[int] = None,
    depth_column: Optional[str] = None,
    encoding: Optional[str] = None,
    null_values: Optional[List[float]] = None
) -> Tuple[pd.DataFrame, AsciiFileMetadata, DepthDetectionResult]:
    """
    Load and parse an ASCII well log file with intelligent format detection.
    
    Args:
        file_obj: File path, bytes, or file-like object
        delimiter: Override auto-detected delimiter
        header_row: Override auto-detected header row
        skip_rows: Number of rows to skip before data
        depth_column: Override auto-detected depth column
        encoding: Override auto-detected encoding
        null_values: Custom null value markers
        
    Returns:
        Tuple of (DataFrame, AsciiFileMetadata, DepthDetectionResult)
    """
    # Default null values for well log data
    default_null_values = [-999.25, -999, -9999, -9999.25, -999.00, -999.2500]
    if null_values is None:
        null_values = default_null_values
    
    # Read raw bytes
    if isinstance(file_obj, str):
        with open(file_obj, 'rb') as f:
            raw_bytes = f.read()
    elif isinstance(file_obj, bytes):
        raw_bytes = file_obj
    else:
        raw_bytes = file_obj.read()
        if isinstance(raw_bytes, str):
            raw_bytes = raw_bytes.encode('utf-8')
    
    # Initialize detectors
    format_detector = AsciiFormatDetector()
    depth_detector = DepthColumnDetector()
    notes = []
    
    # Detect encoding
    if encoding is None:
        encoding = format_detector.detect_encoding(raw_bytes)
        notes.append(f"Encoding: {encoding}")
    
    # Decode to string
    text_content = raw_bytes.decode(encoding, errors='ignore')
    lines = text_content.split('\n')
    
    # Detect comment character
    comment_char = format_detector.detect_comment_char(lines)
    
    # Detect delimiter
    if delimiter is None:
        delimiter, delimiter_type = format_detector.detect_delimiter(lines)
    else:
        delimiter_type = DelimiterType.UNKNOWN
    
    # Detect header row
    if header_row is None and skip_rows is None:
        header_idx, skip_rows = format_detector.detect_header_row(lines, delimiter)
    else:
        header_idx = header_row if header_row is not None else -1
        skip_rows = skip_rows if skip_rows is not None else 0
    
    # Parse the file with pandas
    try:
        if delimiter == 'whitespace':
            df = pd.read_csv(
                io.StringIO(text_content),
                delim_whitespace=True,
                header=header_idx if header_idx >= 0 else None,
                skiprows=skip_rows if header_idx < 0 else None,
                na_values=null_values,
                comment=comment_char,
                engine='python'
            )
        else:
            df = pd.read_csv(
                io.StringIO(text_content),
                delimiter=delimiter,
                header=header_idx if header_idx >= 0 else None,
                skiprows=skip_rows if header_idx < 0 else None,
                na_values=null_values,
                comment=comment_char,
                engine='python'
            )
    except Exception as e:
        # Fallback: try with more lenient settings
        df = pd.read_csv(
            io.StringIO(text_content),
            delimiter=delimiter if delimiter != 'whitespace' else None,
            delim_whitespace=delimiter == 'whitespace',
            header=None,
            skiprows=skip_rows,
            na_values=null_values,
            on_bad_lines='skip',
            engine='python'
        )
    
    # Clean column names
    if df.columns.dtype == 'int64':
        # No header - generate column names
        df.columns = [f'COL_{i}' for i in range(len(df.columns))]
    else:
        df.columns = [str(c).strip() for c in df.columns]
    
    # Convert numeric columns
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        except:
            pass
    
    # Detect depth column
    if depth_column:
        depth_result = DepthDetectionResult(
            column_name=depth_column,
            column_index=df.columns.tolist().index(depth_column) if depth_column in df.columns else 0,
            detection_method='manual',
            confidence=1.0,
            detected_unit='unknown',
            is_monotonic=True,
            is_increasing=True,
            depth_range=(df[depth_column].min(), df[depth_column].max()) if depth_column in df.columns else (0, 0),
            step_size=None
        )
    else:
        depth_result = depth_detector.detect(df)
    
    # Combine notes
    all_notes = format_detector.notes + depth_detector.notes + notes
    
    # Calculate confidence score
    confidence = 0.5
    if delimiter_type != DelimiterType.UNKNOWN:
        confidence += 0.2
    if header_idx >= 0:
        confidence += 0.1
    if depth_result.confidence > 0.5:
        confidence += 0.2
    
    # Build metadata
    metadata = AsciiFileMetadata(
        detected_delimiter=delimiter,
        delimiter_type=delimiter_type,
        header_row_index=header_idx,
        skip_rows=skip_rows,
        depth_column=depth_result.column_name,
        depth_column_index=depth_result.column_index,
        depth_unit=depth_result.detected_unit,
        num_data_columns=len(df.columns),
        num_data_rows=len(df),
        column_names=df.columns.tolist(),
        confidence_score=min(confidence, 1.0),
        encoding=encoding,
        comment_char=comment_char,
        null_values=null_values,
        detection_notes=all_notes
    )
    
    return df, metadata, depth_result


def create_pseudo_las_from_ascii(df: pd.DataFrame, metadata: AsciiFileMetadata, 
                                  depth_result: DepthDetectionResult) -> 'lasio.LASFile':
    """
    Create a pseudo-lasio LASFile object from ASCII data for compatibility
    with existing processing pipelines.
    
    Args:
        df: Parsed DataFrame
        metadata: ASCII file metadata
        depth_result: Depth detection result
        
    Returns:
        lasio.LASFile-like object
    """
    import lasio
    
    las = lasio.LASFile()
    
    # Set basic header info
    las.well.WELL = 'UNKNOWN'
    las.well.NULL = -999.25
    
    # Rename depth column to standard name
    depth_col = depth_result.column_name
    if depth_col and depth_col in df.columns:
        las.well.STRT = df[depth_col].min()
        las.well.STOP = df[depth_col].max()
        if depth_result.step_size:
            las.well.STEP = depth_result.step_size
        
        # Add depth curve first
        las.append_curve('DEPT', df[depth_col].values, 
                         unit=depth_result.detected_unit, 
                         descr='Depth')
    
    # Add other curves
    for col in df.columns:
        if col != depth_col:
            try:
                values = pd.to_numeric(df[col], errors='coerce').values
                las.append_curve(col, values, unit='', descr=col)
            except:
                pass
    
    return las


def validate_ascii_data(df: pd.DataFrame, depth_result: DepthDetectionResult) -> Dict[str, Any]:
    """
    Validate ASCII data quality and return a validation report.
    
    Args:
        df: Parsed DataFrame
        depth_result: Depth detection result
        
    Returns:
        Dictionary with validation results
    """
    report = {
        'is_valid': True,
        'issues': [],
        'warnings': [],
        'column_stats': {}
    }
    
    # Check depth column
    if not depth_result.column_name:
        report['is_valid'] = False
        report['issues'].append("No depth column detected")
    
    # Check for empty DataFrame
    if df.empty:
        report['is_valid'] = False
        report['issues'].append("No data rows found")
        return report
    
    # Check each column
    for col in df.columns:
        col_stats = {
            'dtype': str(df[col].dtype),
            'null_count': int(df[col].isna().sum()),
            'null_percentage': float(df[col].isna().mean() * 100),
            'is_numeric': pd.api.types.is_numeric_dtype(df[col])
        }
        
        if col_stats['is_numeric']:
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                col_stats['min'] = float(valid_data.min())
                col_stats['max'] = float(valid_data.max())
                col_stats['mean'] = float(valid_data.mean())
                col_stats['std'] = float(valid_data.std())
        
        report['column_stats'][col] = col_stats
        
        # Warnings
        if col_stats['null_percentage'] > 50:
            report['warnings'].append(f"Column '{col}' has {col_stats['null_percentage']:.1f}% null values")
    
    return report
