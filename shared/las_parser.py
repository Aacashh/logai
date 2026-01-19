"""
LAS and DLIS File Parser
Handles loading and parsing of well log files (LAS and DLIS formats).
"""

import lasio
import pandas as pd
import numpy as np
import io

# Try to import dlisio for DLIS support
try:
    import dlisio
    DLISIO_AVAILABLE = True
except ImportError:
    DLISIO_AVAILABLE = False

# Try to import ASCII parser for fallback support
try:
    from shared.ascii_parser import load_ascii_file, create_pseudo_las_from_ascii
    ASCII_PARSER_AVAILABLE = True
except ImportError:
    ASCII_PARSER_AVAILABLE = False


def load_las(file_obj, try_ascii_fallback=True):
    """
    Loads a LAS file from a file-like object or file path.
    Improved error handling for problematic LAS files with ASCII fallback.
    
    Args:
        file_obj: File-like object, bytes, or file path string
        try_ascii_fallback: If True, try ASCII parser when lasio fails
        
    Returns:
        lasio.LASFile object
    """
    original_error = None
    
    try:
        # Prepare data for reading
        if isinstance(file_obj, str):
            # File path - read directly
            return lasio.read(
                file_obj,
                ignore_header_errors=True,
                mnemonic_case='preserve'
            )
        elif isinstance(file_obj, bytes):
            # Raw bytes
            str_data = file_obj.decode("utf-8", errors="ignore")
            raw_bytes = file_obj  # Keep for ASCII fallback
        else:
            # File-like object (e.g., from upload)
            bytes_data = file_obj.read()
            str_data = bytes_data.decode("utf-8", errors="ignore")
            raw_bytes = bytes_data  # Keep for ASCII fallback
        
        # Try reading with various options to handle problematic files
        try:
            return lasio.read(
                io.StringIO(str_data),
                ignore_header_errors=True,
                mnemonic_case='preserve'
            )
        except Exception as e1:
            original_error = e1
            # Try with more lenient options
            try:
                return lasio.read(
                    io.StringIO(str_data),
                    ignore_header_errors=True,
                    mnemonic_case='preserve',
                    ignore_data_comments='#',
                    dtypes='auto'
                )
            except Exception as e2:
                # Final fallback: minimal parsing
                try:
                    return lasio.read(
                        io.StringIO(str_data),
                        ignore_header_errors=True,
                        mnemonic_case='upper',
                        read_policy='default'
                    )
                except Exception as e3:
                    original_error = e1
                    raise ValueError(f"Error loading LAS file after multiple attempts: {e1}")
    except Exception as e:
        original_error = e
        
        # Try ASCII fallback if available
        if try_ascii_fallback and ASCII_PARSER_AVAILABLE:
            try:
                # Reset file position if needed
                if hasattr(file_obj, 'seek'):
                    file_obj.seek(0)
                    raw_bytes = file_obj.read()
                
                df, metadata, depth_result = load_ascii_file(raw_bytes)
                
                # Only accept ASCII fallback if we got reasonable data
                if len(df) > 0 and metadata.confidence_score >= 0.4:
                    las = create_pseudo_las_from_ascii(df, metadata, depth_result)
                    return las
            except Exception as ascii_error:
                pass  # ASCII fallback failed, raise original error
        
        raise ValueError(f"Error loading LAS file: {original_error}")


def load_dlis(file_obj):
    """
    Loads a DLIS file and converts to a format compatible with the LAS workflow.
    
    Args:
        file_obj: File-like object, bytes, or file path string
        
    Returns:
        Tuple of (DataFrame, header_dict, curves_info) mimicking LAS structure
    """
    if not DLISIO_AVAILABLE:
        raise ImportError(
            "DLIS file support requires the 'dlisio' library. "
            "Install it with: pip install dlisio"
        )
    
    try:
        # Handle different input types
        if isinstance(file_obj, str):
            file_path = file_obj
        else:
            # Write to temp file for dlisio (it requires file path)
            import tempfile
            import os
            
            if hasattr(file_obj, 'read'):
                bytes_data = file_obj.read()
            else:
                bytes_data = file_obj
            
            # Create temp file
            fd, file_path = tempfile.mkstemp(suffix='.dlis')
            try:
                os.write(fd, bytes_data)
                os.close(fd)
            except:
                os.close(fd)
                raise
        
        # Load DLIS file
        with dlisio.dlis.load(file_path) as files:
            # Take the first logical file
            dlis_file = files[0]
            
            # Extract header information
            header = {}
            try:
                origin = dlis_file.origins[0] if dlis_file.origins else None
                if origin:
                    header['WELL'] = getattr(origin, 'well_name', 'UNKNOWN') or 'UNKNOWN'
                    header['FIELD'] = getattr(origin, 'field_name', '') or ''
                    header['COMP'] = getattr(origin, 'company', '') or ''
            except:
                header['WELL'] = 'UNKNOWN'
            
            # Extract frames (data tables)
            all_data = {}
            curves_info = []
            
            for frame in dlis_file.frames:
                # Get channels (curves) from this frame
                for channel in frame.channels:
                    try:
                        mnemonic = channel.name
                        unit = channel.units or ''
                        descr = channel.long_name or ''
                        
                        # Get data
                        data = channel.curves()
                        
                        # Handle multi-dimensional data (take first column)
                        if len(data.shape) > 1:
                            data = data[:, 0]
                        
                        all_data[mnemonic] = data
                        curves_info.append({
                            'mnemonic': mnemonic,
                            'unit': unit,
                            'description': descr
                        })
                    except Exception as e:
                        continue
            
            if not all_data:
                raise ValueError("No valid channels found in DLIS file")
            
            # Create DataFrame
            df = pd.DataFrame(all_data)
            
            # Clean up temp file if created
            if not isinstance(file_obj, str):
                try:
                    os.unlink(file_path)
                except:
                    pass
            
            return df, header, curves_info
            
    except Exception as e:
        raise ValueError(f"Error loading DLIS file: {e}")


def load_well_log_file(file_obj, filename=None):
    """
    Unified loader for well log files (LAS or DLIS).
    Auto-detects file type based on extension.
    
    Args:
        file_obj: File-like object, bytes, or file path string
        filename: Optional filename to determine file type (if file_obj is bytes)
        
    Returns:
        For LAS: lasio.LASFile object
        For DLIS: Tuple of (DataFrame, header_dict, curves_info)
    """
    # Determine file type
    if isinstance(file_obj, str):
        file_ext = file_obj.lower().split('.')[-1]
    elif filename:
        file_ext = filename.lower().split('.')[-1]
    elif hasattr(file_obj, 'name'):
        file_ext = file_obj.name.lower().split('.')[-1]
    else:
        file_ext = 'las'  # Default to LAS
    
    if file_ext == 'dlis':
        return load_dlis(file_obj), 'DLIS'
    else:
        return load_las(file_obj), 'LAS'


def extract_header_info(las):
    """
    Extracts metadata from LAS file header for display.
    
    Args:
        las: lasio.LASFile object
        
    Returns:
        Dictionary with well metadata
    """
    def safe_get(attr, default=''):
        try:
            if attr in las.well:
                val = getattr(las.well, attr).value
                return val if val else default
        except:
            pass
        return default
    
    return {
        'WELL': safe_get('WELL', 'UNKNOWN'),
        'FIELD': safe_get('FLD', '') or safe_get('FIELD', ''),
        'LOC': safe_get('LOC', '') or safe_get('LOCATION', ''),
        'COMP': safe_get('COMP', '') or safe_get('OPER', ''),
        'STRT': safe_get('STRT', 0),
        'STOP': safe_get('STOP', 0),
        'STEP': safe_get('STEP', 0),
        'CTRY': safe_get('CTRY', ''),
        'SRVC': safe_get('SRVC', ''),
        'DATE': safe_get('DATE', ''),
        'UWI': safe_get('UWI', ''),
        'API': safe_get('API', ''),
    }


def detect_depth_units(las):
    """
    Detect depth units from LAS header.
    
    Args:
        las: lasio.LASFile object
        
    Returns:
        'm' for meters, 'ft' for feet
    """
    # Check STRT unit first
    try:
        if 'STRT' in las.well:
            unit = las.well.STRT.unit.upper()
            if 'F' in unit:
                return 'ft'
            elif 'M' in unit:
                return 'm'
    except:
        pass
    
    # Check curve definition for DEPT
    try:
        for curve in las.curves:
            if curve.mnemonic.upper() in ['DEPT', 'DEPTH', 'MD']:
                unit = curve.unit.upper()
                if 'F' in unit:
                    return 'ft'
                elif 'M' in unit:
                    return 'm'
    except:
        pass
    
    return 'm'  # Default to meters


def get_available_curves(las):
    """
    Get list of all available curves in LAS file.
    
    Args:
        las: lasio.LASFile object
        
    Returns:
        List of dictionaries with curve info
    """
    curves = []
    for curve in las.curves:
        # Use numpy's nanmin/nanmax for faster computation
        data = curve.data
        valid_count = np.count_nonzero(~np.isnan(data)) if len(data) > 0 else 0
        curves.append({
            'mnemonic': curve.mnemonic,
            'unit': curve.unit,
            'description': curve.descr,
            'data_range': {
                'min': float(np.nanmin(data)) if valid_count > 0 else None,
                'max': float(np.nanmax(data)) if valid_count > 0 else None
            }
        })
    return curves
