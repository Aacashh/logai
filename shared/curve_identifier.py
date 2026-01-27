"""
Layered Methodology for Standard Curve Identification

Implements a 12-layer approach for identifying well log curves:
    Layer 0:  File & Context - File type: LAS / DLIS
    Layer 1:  Header Reading - Capture human intent from logging engineers
    Layer 2:  Keyword / NLP - Keyword matching (gamma, neutron, density, resistivity)
    Layer 3:  Mnemonic Matching - Capture industry conventions
    Layer 4:  Unit Validation - Validate physical meaning
    Layer 5:  Physical Range - Apply laws of physics (Density: 1.9–3.0 g/cc)
    Layer 6:  Statistical Shape - Identify curve behavior patterns
    Layer 7:  Tool Context - Resolve ambiguity using logging context (LWD → R40/R36 more likely)
    Layer 8:  Cross-Curve Checks - High GR + low Rt → shale
    Layer 9:  Duplicate Resolution - Choose the best curve when multiples exist
    Layer 10: Confidence & Explainability - Confidence score (0-1)
    Layer 11: Learning & Exceptions - Active learning
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import re


# =============================================================================
# ROBUST SCORING WEIGHTS (from client log_core.py)
# =============================================================================
# Explicit point-based scoring for curve identification.
# Total max score: 100 points

SCORE_WEIGHTS = {
    "unit": 40,          # Unit validation - strongest signal
    "range": 25,         # Physical range validation
    "description": 15,   # Keyword match in description
    "mnemonic": 10,      # Mnemonic pattern match
    "coverage": 10,      # Non-null data coverage (scaled 0-10)
}

# Minimum score threshold to identify a curve (below this = UNKNOWN)
MIN_CONFIDENCE_THRESHOLD = 15

# Null values commonly used in well logs
NULL_VALUES = {-999, -999.25, -999.0, -9999, -9999.25}


def _clean_unit(unit: str) -> str:
    """Normalize unit string for robust comparison."""
    if not unit:
        return ""
    return unit.lower().replace(" ", "").replace(".", "").replace("-", "").replace("_", "")


def _ensure_array(data) -> np.ndarray:
    """Ensure data is a numpy array, not a scalar. Fixes scalar-passed-as-array bug."""
    if data is None:
        return np.array([])
    if np.ndim(data) == 0:
        return np.array([data])
    return np.asarray(data)


def _calculate_coverage(data: np.ndarray) -> float:
    """Calculate non-null data coverage ratio (0.0 to 1.0)."""
    data = _ensure_array(data)
    if len(data) == 0:
        return 0.0
    # Replace common null values with NaN
    data = data.astype(float)
    for null_val in NULL_VALUES:
        data = np.where(data == null_val, np.nan, data)
    valid_count = np.sum(~np.isnan(data))
    return valid_count / len(data)


def _percentile_range(data: np.ndarray) -> Optional[Tuple[float, float]]:
    """Get 7th and 93rd percentile range for robust outlier handling."""
    data = _ensure_array(data)
    if len(data) == 0:
        return None
    # Replace null values
    data = data.astype(float)
    for null_val in NULL_VALUES:
        data = np.where(data == null_val, np.nan, data)
    valid_data = data[~np.isnan(data)]
    if len(valid_data) < 10:
        return None
    return (float(np.percentile(valid_data, 7)), float(np.percentile(valid_data, 93)))


# =============================================================================
# DOI-AWARE RESISTIVITY CLASSIFICATION HELPERS
# =============================================================================

def extract_valid_doi(mnemonic: str) -> Optional[int]:
    """
    Extract Depth of Investigation (DOI) from mnemonic.
    
    DOI must be a 2 or 3 digit numeric suffix. Single-digit suffixes 
    (e.g. RLA5) are NOT DOI - they represent curve index/channel number.
    
    Args:
        mnemonic: Curve mnemonic string (e.g., 'AT90', 'AE60', 'RLA5')
        
    Returns:
        Integer DOI value (e.g., 90, 60) or None if no valid DOI suffix
        
    Examples:
        'AT90' -> 90 (valid: 2-digit)
        'AE60' -> 60 (valid: 2-digit)  
        'R40'  -> 40 (valid: 2-digit)
        'RLA5' -> None (invalid: single digit = channel index)
        'GR'   -> None (no numeric suffix)
    """
    if not mnemonic:
        return None
    
    # Match exactly 2 or 3 digit suffix at end of string
    m = re.search(r'(\d{2,3})$', mnemonic)
    if m:
        return int(m.group(1))
    
    return None


def classify_resistivity_by_doi(
    resistivity_curves: List[Tuple[str, float, 'CurveType']]
) -> Dict[str, 'CurveType']:
    """
    Classify resistivity curves by Depth of Investigation (DOI) suffix.
    
    This is a deterministic post-identification classification step that uses
    DOI suffixes to correctly distinguish deep/medium/shallow resistivity.
    
    Args:
        resistivity_curves: List of (mnemonic, confidence_score, current_type) tuples
                           for all identified resistivity curves
        
    Returns:
        Dict mapping mnemonic to reclassified CurveType:
        - Highest DOI suffix -> RES_DEEP
        - Next lower DOI -> RES_MED  
        - Lowest DOI suffix -> RES_SHAL
        
        Returns empty dict if no valid DOI suffixes found (fallback to scoring)
        
    Examples:
        [('AT90', 0.8, RES_DEEP), ('AT60', 0.75, RES_MED), ('AT10', 0.7, RES_SHAL)]
        -> {'AT90': RES_DEEP, 'AT60': RES_MED, 'AT10': RES_SHAL}
    """
    # Import here to avoid circular reference issues
    from enum import Enum
    
    curves_with_doi = []
    
    for mnemonic, score, current_type in resistivity_curves:
        doi = extract_valid_doi(mnemonic)
        if doi is not None:
            curves_with_doi.append((mnemonic, doi, score))
    
    # If no valid DOI found, return empty -> fallback to existing logic
    if not curves_with_doi:
        return {}
    
    # Sort by DOI descending (highest DOI = deepest investigation)
    curves_with_doi.sort(key=lambda x: x[1], reverse=True)
    
    result = {}
    
    # Import CurveType for return values (forward reference workaround)
    # Note: CurveType is defined below, so we reference it dynamically
    if len(curves_with_doi) >= 1:
        result[curves_with_doi[0][0]] = "RES_DEEP"  # Will be converted later
    
    if len(curves_with_doi) >= 2:
        result[curves_with_doi[1][0]] = "RES_MED"
    
    if len(curves_with_doi) >= 3:
        result[curves_with_doi[2][0]] = "RES_SHAL"
    
    return result


class CurveType(Enum):
    """Standard curve types in well logging."""
    DEPTH = "DEPTH"
    GR = "GR"                    # Gamma Ray
    RES_DEEP = "RES_DEEP"        # Deep Resistivity
    RES_MED = "RES_MED"          # Medium Resistivity
    RES_SHAL = "RES_SHAL"        # Shallow Resistivity
    DENS = "DENS"                # Bulk Density
    NEUT = "NEUT"                # Neutron Porosity
    SONIC = "SONIC"              # Sonic/Acoustic
    CALIPER = "CALIPER"          # Caliper
    SP = "SP"                    # Spontaneous Potential
    PEF = "PEF"                  # Photoelectric Factor
    DPHI = "DPHI"                # Density Porosity
    PERM = "PERM"                # Permeability
    UNKNOWN = "UNKNOWN"


class FileType(Enum):
    """Well log file types."""
    LAS = "LAS"
    DLIS = "DLIS"
    UNKNOWN = "UNKNOWN"


class ToolContext(Enum):
    """Logging tool context."""
    WIRELINE = "WIRELINE"
    LWD = "LWD"          # Logging While Drilling
    MWD = "MWD"          # Measurement While Drilling
    UNKNOWN = "UNKNOWN"


@dataclass
class LayerResult:
    """Result from a single identification layer."""
    layer_name: str
    layer_number: int
    confidence_contribution: float  # 0.0 to 1.0
    identified_type: Optional[CurveType]
    reasoning: str
    passed: bool


@dataclass
class CurveIdentificationResult:
    """Complete identification result for a single curve."""
    original_mnemonic: str
    identified_type: CurveType
    confidence_score: float  # 0.0 to 1.0
    raw_score: float  # Raw point score (0-100)
    layer_results: List[LayerResult]
    explanation: str
    alternative_types: List[Tuple[CurveType, float]]  # (type, confidence)
    score_breakdown: Dict[str, float] = field(default_factory=dict)  # Points per category
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    selected_as_primary: bool = True


@dataclass
class IdentificationReport:
    """Complete identification report for all curves."""
    file_type: FileType
    tool_context: ToolContext
    curve_results: Dict[str, CurveIdentificationResult]
    mapping: Dict[str, str]  # CurveType -> original mnemonic
    duplicate_warnings: List[str]
    cross_curve_insights: List[str]
    overall_confidence: float


# =============================================================================
# LAYER CONFIGURATION: Industry Knowledge Base
# =============================================================================

# Layer 2 & 3: Keyword and Mnemonic patterns
CURVE_PATTERNS = {
    # NOTE: Order matters for DEPTH - MD is prioritized over TVD/TVDSS
    CurveType.DEPTH: {
        'mnemonics': ['MD', 'DEPT', 'DEPTH', 'DPTH', 'TVD', 'TVDSS', 'AHD'],
        'keywords': ['depth', 'measured', 'vertical'],
        'weight': 1.0
    },
    CurveType.GR: {
        'mnemonics': ['GR', 'GRC', 'GR_EDTC', 'SGR', 'CGR', 'HCGR', 'ECGR', 
                      'GRD', 'GRS', 'GRAM', 'GAPI', 'NGR', 'NGAM'],
        'keywords': ['gamma', 'ray', 'natural', 'spectral'],
        'weight': 1.0
    },
    CurveType.RES_DEEP: {
        'mnemonics': ['RT', 'RDEP', 'RLLD', 'RLL3', 'ILD', 'RILD', 'LLD', 'RD',
                      'RDLA', 'RLA5', 'RLA3', 'AT90', 'AHT90', 'R40', 'R36',
                      'HDRS', 'ATRT', 'RTRUE', 'RXO8'],
        'keywords': ['deep', 'resistivity', 'true', 'formation'],
        'weight': 1.0
    },
    CurveType.RES_MED: {
        'mnemonics': ['RM', 'RMED', 'ILM', 'RILM', 'RLM', 'RILM', 'AT60',
                      'AHT60', 'R24', 'R20', 'HMRS', 'RLA2'],
        'keywords': ['medium', 'mid', 'intermediate'],
        'weight': 1.0
    },
    CurveType.RES_SHAL: {
        'mnemonics': ['RS', 'RSHAL', 'MSFL', 'RXOZ', 'RXO', 'RLL1', 'SFL', 'LLS',
                      'RSFL', 'RSHA', 'AT10', 'AHT10', 'R10', 'LLHR', 'MLL'],
        'keywords': ['shallow', 'invaded', 'flushed', 'microspherically'],
        'weight': 1.0
    },
    CurveType.DENS: {
        'mnemonics': ['RHOB', 'RHOZ', 'DEN', 'ZDEN', 'BDEL', 'DENB', 'DENC',
                      'RHO', 'RHOF', 'DENS', 'HDENS', 'DRHO', 'DCOR'],
        'keywords': ['density', 'bulk', 'formation'],
        'weight': 1.0
    },
    CurveType.NEUT: {
        'mnemonics': ['NPHI', 'TNPH', 'NPHZ', 'CNPOR', 'NPOR', 'TNPHI', 'PHIN',
                      'NEU', 'NEUT', 'NPSS', 'NPLS', 'HNPO', 'APLC', 'APSC'],
        'keywords': ['neutron', 'porosity', 'hydrogen'],
        'weight': 1.0
    },
    CurveType.SONIC: {
        'mnemonics': ['DT', 'DTC', 'DTCO', 'AC', 'SONIC', 'DTS', 'DTCS',
                      'DTSM', 'DT4P', 'DT4S', 'DTLN', 'DTSH'],
        'keywords': ['sonic', 'acoustic', 'slowness', 'compressional', 'shear'],
        'weight': 1.0
    },
    CurveType.CALIPER: {
        'mnemonics': ['CALI', 'CAL', 'HCAL', 'DCAL', 'CALS', 'BS', 'DEVI',
                      'C1', 'C2', 'C13', 'C24', 'LCAL', 'SCAL'],
        'keywords': ['caliper', 'borehole', 'diameter', 'hole'],
        'weight': 1.0
    },
    CurveType.SP: {
        'mnemonics': ['SP', 'SSP', 'PSP', 'SPR', 'SPHI'],
        'keywords': ['spontaneous', 'potential', 'self'],
        'weight': 1.0
    },
    CurveType.PEF: {
        'mnemonics': ['PEF', 'PE', 'PEFZ', 'PEZ', 'PEFA', 'PEFL'],
        'keywords': ['photoelectric', 'factor', 'pe'],
        'weight': 1.0
    },
    CurveType.DPHI: {
        'mnemonics': ['DPHI', 'PHID', 'DPOR', 'DENPOR', 'PHD'],
        'keywords': ['density', 'porosity'],
        'weight': 1.0
    },
}

# Layer 4: Unit validation patterns
UNIT_PATTERNS = {
    CurveType.DEPTH: {
        'valid_units': ['m', 'ft', 'f', 'meter', 'meters', 'feet', 'foot'],
        'typical_range': None  # Variable
    },
    CurveType.GR: {
        'valid_units': ['gapi', 'api', 'gAPI', 'API', 'ur', 'cps'],
        'typical_range': (0, 300)  # gAPI
    },
    CurveType.RES_DEEP: {
        # Includes all common resistivity unit variations: OHMM, OHM-M, OHM.M, etc.
        'valid_units': ['ohmm', 'ohm.m', 'ohm-m', 'ohm', 'ohmm2/m', 'OHMM', 'OHM-M', 'OHM.M', 'ohms'],
        'typical_range': (0.1, 10000)  # ohm.m
    },
    CurveType.RES_MED: {
        'valid_units': ['ohmm', 'ohm.m', 'ohm-m', 'ohm', 'ohmm2/m', 'OHMM', 'OHM-M', 'OHM.M', 'ohms'],
        'typical_range': (0.1, 10000)
    },
    CurveType.RES_SHAL: {
        'valid_units': ['ohmm', 'ohm.m', 'ohm-m', 'ohm', 'ohmm2/m', 'OHMM', 'OHM-M', 'OHM.M', 'ohms'],
        'typical_range': (0.1, 10000)
    },
    CurveType.DENS: {
        'valid_units': ['g/cc', 'g/cm3', 'gm/cc', 'kg/m3', 'g/c3'],
        'typical_range': (1.0, 3.5)  # g/cc
    },
    CurveType.NEUT: {
        # Supports both decimal (v/v) and percentage (%, PU) units
        'valid_units': ['v/v', 'pu', 'PU', '%', 'frac', 'dec', 'm3/m3', 'percent'],
        'typical_range': (-0.15, 0.60),  # v/v or fractional
        'percentage_range': (-15, 60)    # For % or PU units
    },
    CurveType.SONIC: {
        'valid_units': ['us/ft', 'us/m', 'usec/ft', 'usec/m', 'µs/ft', 'µs/m'],
        'typical_range': (40, 200)  # us/ft
    },
    CurveType.CALIPER: {
        'valid_units': ['in', 'inch', 'inches', 'cm', 'mm'],
        'typical_range': (4, 20)  # inches
    },
    CurveType.SP: {
        'valid_units': ['mv', 'millivolts', 'v'],
        'typical_range': (-200, 100)  # mV
    },
    CurveType.PEF: {
        'valid_units': ['b/e', 'barn/electron', 'barns/e', ''],
        'typical_range': (1.5, 6.0)  # b/e
    },
}

# Layer 5: Physical ranges for validation
PHYSICAL_RANGES = {
    CurveType.GR: {'min': 0, 'max': 500, 'typical_min': 0, 'typical_max': 200},
    CurveType.RES_DEEP: {'min': 0.01, 'max': 100000, 'typical_min': 0.1, 'typical_max': 2000},
    CurveType.RES_MED: {'min': 0.01, 'max': 100000, 'typical_min': 0.1, 'typical_max': 2000},
    CurveType.RES_SHAL: {'min': 0.01, 'max': 100000, 'typical_min': 0.1, 'typical_max': 2000},
    CurveType.DENS: {'min': 1.0, 'max': 3.5, 'typical_min': 1.9, 'typical_max': 3.0},
    # NEUT ranges depend on unit: v/v uses decimals, % and PU use percentages
    CurveType.NEUT: {'min': -0.20, 'max': 1.0, 'typical_min': -0.05, 'typical_max': 0.45,
                     'min_pct': -20, 'max_pct': 100, 'typical_min_pct': -5, 'typical_max_pct': 45},
    CurveType.SONIC: {'min': 30, 'max': 250, 'typical_min': 40, 'typical_max': 140},
    CurveType.CALIPER: {'min': 0, 'max': 30, 'typical_min': 6, 'typical_max': 18},
    CurveType.SP: {'min': -500, 'max': 500, 'typical_min': -200, 'typical_max': 50},
    CurveType.PEF: {'min': 0.5, 'max': 10, 'typical_min': 1.5, 'typical_max': 6.0},
}

# Layer 7: Tool context patterns
TOOL_CONTEXT_PATTERNS = {
    ToolContext.LWD: {
        'mnemonics': ['R40', 'R36', 'R24', 'R20', 'R10', 'ARC', 'ADN', 'CDN'],
        'keywords': ['lwd', 'while drilling', 'ring', 'bit'],
        'resistivity_preference': ['R40', 'R36', 'AT90', 'AT60', 'AT10']
    },
    ToolContext.WIRELINE: {
        'mnemonics': ['LLD', 'LLS', 'MSFL', 'ILD', 'ILM', 'SFL'],
        'keywords': ['wireline', 'induction', 'laterolog'],
        'resistivity_preference': ['LLD', 'ILD', 'RLLD', 'RT']
    }
}


# =============================================================================
# LAYER IMPLEMENTATIONS
# =============================================================================

class CurveIdentifier:
    """
    12-Layer Curve Identification System.
    
    Implements a comprehensive methodology for identifying well log curves
    using multiple validation layers.
    """
    
    def __init__(self):
        self.layer_weights = {
            0: 0.05,   # File & Context
            1: 0.10,   # Header Reading
            2: 0.10,   # Keyword / NLP
            3: 0.20,   # Mnemonic Matching
            4: 0.15,   # Unit Validation
            5: 0.15,   # Physical Range
            6: 0.10,   # Statistical Shape
            7: 0.05,   # Tool Context
            8: 0.05,   # Cross-Curve Checks
            9: 0.02,   # Duplicate Resolution
            10: 0.02,  # Confidence & Explainability
            11: 0.01,  # Learning & Exceptions
        }
        self.learning_exceptions: Dict[str, CurveType] = {}
    
    def identify_curves(self, las, file_type: FileType = FileType.LAS) -> IdentificationReport:
        """
        Main entry point: identify all curves in a LAS file.
        
        Args:
            las: lasio.LASFile object
            file_type: Type of file (LAS or DLIS)
            
        Returns:
            IdentificationReport with all results
        """
        # Layer 0: File & Context
        file_context = self._layer_0_file_context(las, file_type)
        
        # Layer 7: Detect tool context early (affects later layers)
        tool_context = self._layer_7_detect_tool_context(las)
        
        # Process each curve
        curve_results = {}
        candidates_by_type: Dict[CurveType, List[Tuple[str, float]]] = {}
        
        for curve in las.curves:
            mnemonic = curve.mnemonic
            result = self._identify_single_curve(
                curve, las, file_context, tool_context
            )
            curve_results[mnemonic] = result
            
            # Track candidates for duplicate resolution
            if result.identified_type != CurveType.UNKNOWN:
                if result.identified_type not in candidates_by_type:
                    candidates_by_type[result.identified_type] = []
                candidates_by_type[result.identified_type].append(
                    (mnemonic, result.confidence_score)
                )
        
        # Layer 8: Cross-curve validation
        cross_insights = self._layer_8_cross_curve_checks(curve_results, las)
        
        # Layer 9: Duplicate resolution
        curve_results, duplicates = self._layer_9_duplicate_resolution(
            curve_results, candidates_by_type
        )
        
        # Build final mapping
        mapping = {}
        for mnemonic, result in curve_results.items():
            if result.selected_as_primary and result.identified_type != CurveType.UNKNOWN:
                type_name = result.identified_type.value
                if type_name not in mapping:
                    mapping[type_name] = mnemonic
        
        # Calculate overall confidence
        confidences = [r.confidence_score for r in curve_results.values() 
                      if r.identified_type != CurveType.UNKNOWN]
        overall_confidence = np.mean(confidences) if confidences else 0.0
        
        return IdentificationReport(
            file_type=file_type,
            tool_context=tool_context,
            curve_results=curve_results,
            mapping=mapping,
            duplicate_warnings=duplicates,
            cross_curve_insights=cross_insights,
            overall_confidence=overall_confidence
        )
    
    def _identify_single_curve(
        self, 
        curve, 
        las, 
        file_context: dict,
        tool_context: ToolContext
    ) -> CurveIdentificationResult:
        """
        Identify a single curve using robust point-based scoring.
        
        Uses explicit weights from SCORE_WEIGHTS for transparent scoring:
        - Unit: 40 pts
        - Range: 25 pts  
        - Description: 15 pts
        - Mnemonic: 10 pts
        - Coverage: 0-10 pts (scaled)
        """
        
        mnemonic = curve.mnemonic
        unit = curve.unit
        description = curve.descr
        data = _ensure_array(curve.data)  # Fix scalar bug
        
        layer_results = []
        
        # Track scores per curve type using point-based system
        type_scores: Dict[CurveType, float] = {ct: 0.0 for ct in CurveType}
        type_score_breakdown: Dict[CurveType, Dict[str, float]] = {
            ct: {"unit": 0, "range": 0, "description": 0, "mnemonic": 0, "coverage": 0}
            for ct in CurveType
        }
        
        # Layer 0: File & Context (metadata only)
        layer_results.append(LayerResult(
            layer_name="File & Context",
            layer_number=0,
            confidence_contribution=0,
            identified_type=None,
            reasoning=f"File type: {file_context['type']}",
            passed=True
        ))
        
        # Layer 1 & 2: Header/Description Keyword Matching (+15 pts max)
        l1_result, desc_matches = self._layer_1_2_description_scoring(description, mnemonic)
        layer_results.append(l1_result)
        for ct in desc_matches:
            type_scores[ct] += SCORE_WEIGHTS["description"]
            type_score_breakdown[ct]["description"] = SCORE_WEIGHTS["description"]
        
        # Layer 3: Mnemonic Matching (+10 pts for exact, +5 for partial)
        l3_result, mnem_match, mnem_score = self._layer_3_mnemonic_scoring(mnemonic)
        layer_results.append(l3_result)
        if mnem_match:
            type_scores[mnem_match] += mnem_score
            type_score_breakdown[mnem_match]["mnemonic"] = mnem_score
        
        # Layer 4: Unit Validation (+40 pts for valid unit match)
        l4_result, unit_matches = self._layer_4_unit_scoring(unit)
        layer_results.append(l4_result)
        for ct in unit_matches:
            type_scores[ct] += SCORE_WEIGHTS["unit"]
            type_score_breakdown[ct]["unit"] = SCORE_WEIGHTS["unit"]
        
        # Layer 5: Physical Range Validation (+25 pts if in range)
        l5_result, range_matches = self._layer_5_range_scoring(data)
        layer_results.append(l5_result)
        for ct, pts in range_matches.items():
            type_scores[ct] += pts
            type_score_breakdown[ct]["range"] = pts
        
        # Layer 6: Statistical Shape (boost existing scores)
        l6_result = self._layer_6_statistical_shape(data, type_scores)
        layer_results.append(l6_result)
        
        # Layer 7: Tool Context adjustment
        l7_result = self._layer_7_tool_context_adjustment(
            mnemonic, type_scores, tool_context
        )
        layer_results.append(l7_result)
        
        # Coverage Scoring (+0-10 pts based on data quality)
        coverage_ratio = _calculate_coverage(data)
        coverage_score = SCORE_WEIGHTS["coverage"] * coverage_ratio
        for ct in type_scores:
            if type_scores[ct] > 0:  # Only add coverage to candidates
                type_scores[ct] += coverage_score
                type_score_breakdown[ct]["coverage"] = coverage_score
        
        # Find best match
        best_type = max(type_scores, key=type_scores.get)
        best_raw_score = type_scores[best_type]
        
        # Convert raw score to confidence (0-1 range, max 100 pts)
        confidence = min(1.0, max(0.0, best_raw_score / 100.0))
        
        # If score is too low, mark as unknown
        if best_raw_score < MIN_CONFIDENCE_THRESHOLD:
            best_type = CurveType.UNKNOWN
        
        # Build alternatives (other candidates with scores)
        alternatives = [
            (ct, score / 100.0) for ct, score in sorted(
                type_scores.items(), key=lambda x: -x[1]
            ) if ct != best_type and score >= MIN_CONFIDENCE_THRESHOLD
        ][:3]
        
        # Generate explanation with score breakdown
        explanation = self._generate_explanation_with_scores(
            layer_results, best_type, best_raw_score, type_score_breakdown.get(best_type, {})
        )
        
        # Layer 11: Check learning exceptions
        if mnemonic in self.learning_exceptions:
            learned_type = self.learning_exceptions[mnemonic]
            explanation += f" [Override from learned exception: {learned_type.value}]"
            best_type = learned_type
            best_raw_score = 95  # High score for learned
            confidence = 0.95
        
        return CurveIdentificationResult(
            original_mnemonic=mnemonic,
            identified_type=best_type,
            confidence_score=confidence,
            raw_score=best_raw_score,
            layer_results=layer_results,
            explanation=explanation,
            alternative_types=alternatives,
            score_breakdown=type_score_breakdown.get(best_type, {})
        )
    
    def _layer_0_file_context(self, las, file_type: FileType) -> dict:
        """Layer 0: Analyze file type and context."""
        context = {
            'type': file_type.value,
            'num_curves': len(las.curves),
            'has_header': bool(las.well),
        }
        return context
    
    # =========================================================================
    # NEW POINT-BASED SCORING METHODS
    # =========================================================================
    
    def _layer_1_2_description_scoring(
        self, description: str, mnemonic: str
    ) -> Tuple[LayerResult, List[CurveType]]:
        """
        Layer 1 & 2: Description/keyword scoring (+15 pts).
        
        Returns:
            Tuple of (LayerResult, list of matched CurveTypes)
        """
        combined_text = f"{mnemonic} {description or ''}".lower()
        
        # Keywords that indicate curve type
        keyword_patterns = {
            CurveType.GR: ['gamma', 'gr ', 'natural radioactivity', 'gapi'],
            CurveType.RES_DEEP: ['deep resistivity', 'true resistivity', 'formation resistivity', 
                                 'deep induction', 'laterolog deep'],
            CurveType.RES_MED: ['medium resistivity', 'intermediate', 'medium induction'],
            CurveType.RES_SHAL: ['shallow resistivity', 'invaded zone', 'flushed', 'microsphere'],
            CurveType.DENS: ['bulk density', 'formation density', 'density log', 'rhob'],
            CurveType.NEUT: ['neutron porosity', 'thermal neutron', 'hydrogen index', 'nphi'],
            CurveType.SONIC: ['sonic', 'acoustic', 'compressional slowness', 'transit time'],
            CurveType.CALIPER: ['caliper', 'borehole size', 'hole diameter', 'bit size'],
            CurveType.SP: ['spontaneous potential', 'self potential'],
            CurveType.PEF: ['photoelectric', 'pe factor', 'pef'],
            CurveType.DEPTH: ['measured depth', 'true vertical', 'depth'],
        }
        
        matches = []
        matched_keywords = []
        
        for curve_type, keywords in keyword_patterns.items():
            for keyword in keywords:
                if keyword in combined_text:
                    if curve_type not in matches:
                        matches.append(curve_type)
                        matched_keywords.append(keyword)
                    break
        
        return LayerResult(
            layer_name="Description/Keyword",
            layer_number=1,
            confidence_contribution=SCORE_WEIGHTS["description"] if matches else 0,
            identified_type=matches[0] if matches else None,
            reasoning=f"Keywords found: {matched_keywords[:3]}" if matches else "No keyword match",
            passed=len(matches) > 0
        ), matches
    
    def _layer_3_mnemonic_scoring(
        self, mnemonic: str
    ) -> Tuple[LayerResult, Optional[CurveType], float]:
        """
        Layer 3: Mnemonic scoring (+10 pts exact, +5 pts partial).
        
        Returns:
            Tuple of (LayerResult, matched CurveType or None, score points)
        """
        mnemonic_upper = mnemonic.upper().strip()
        
        # Check exact match first (10 points)
        for curve_type, config in CURVE_PATTERNS.items():
            for std_mnemonic in config['mnemonics']:
                if mnemonic_upper == std_mnemonic.upper():
                    return LayerResult(
                        layer_name="Mnemonic Match",
                        layer_number=3,
                        confidence_contribution=SCORE_WEIGHTS["mnemonic"],
                        identified_type=curve_type,
                        reasoning=f"Exact match: {mnemonic} = {std_mnemonic}",
                        passed=True
                    ), curve_type, SCORE_WEIGHTS["mnemonic"]
        
        # Check partial match (5 points)
        best_match = None
        best_similarity = 0
        
        for curve_type, config in CURVE_PATTERNS.items():
            for std_mnemonic in config['mnemonics']:
                std_upper = std_mnemonic.upper()
                # Check if one contains the other
                if std_upper in mnemonic_upper or mnemonic_upper in std_upper:
                    similarity = len(std_mnemonic) / max(len(mnemonic), len(std_mnemonic))
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = curve_type
        
        if best_match and best_similarity > 0.4:
            partial_score = SCORE_WEIGHTS["mnemonic"] * 0.5  # 5 pts for partial
            return LayerResult(
                layer_name="Mnemonic Match",
                layer_number=3,
                confidence_contribution=partial_score,
                identified_type=best_match,
                reasoning=f"Partial match: {mnemonic} (similarity: {best_similarity:.0%})",
                passed=True
            ), best_match, partial_score
        
        return LayerResult(
            layer_name="Mnemonic Match",
            layer_number=3,
            confidence_contribution=0,
            identified_type=None,
            reasoning=f"No mnemonic match for: {mnemonic}",
            passed=False
        ), None, 0
    
    def _layer_4_unit_scoring(self, unit: str) -> Tuple[LayerResult, List[CurveType]]:
        """
        Layer 4: Unit validation scoring (+40 pts).
        
        Returns:
            Tuple of (LayerResult, list of CurveTypes with matching units)
        """
        unit_cleaned = _clean_unit(unit)
        
        if not unit_cleaned:
            return LayerResult(
                layer_name="Unit Validation",
                layer_number=4,
                confidence_contribution=0,
                identified_type=None,
                reasoning="No unit specified",
                passed=False
            ), []
        
        matches = []
        
        for curve_type, config in UNIT_PATTERNS.items():
            valid_units = [_clean_unit(u) for u in config['valid_units']]
            # Check if unit matches any valid unit
            for valid_unit in valid_units:
                if valid_unit and (valid_unit in unit_cleaned or unit_cleaned in valid_unit):
                    matches.append(curve_type)
                    break
        
        return LayerResult(
            layer_name="Unit Validation",
            layer_number=4,
            confidence_contribution=SCORE_WEIGHTS["unit"] if matches else 0,
            identified_type=matches[0] if matches else None,
            reasoning=f"Unit '{unit}' matches: {[ct.value for ct in matches]}" if matches else f"Unit '{unit}' not recognized",
            passed=len(matches) > 0
        ), matches
    
    def _layer_5_range_scoring(
        self, data: np.ndarray
    ) -> Tuple[LayerResult, Dict[CurveType, float]]:
        """
        Layer 5: Physical range validation scoring (+25 pts typical, +15 pts physical).
        
        Uses 7th/93rd percentile for robust outlier handling.
        
        Returns:
            Tuple of (LayerResult, dict of CurveType -> score points)
        """
        data = _ensure_array(data)
        pr = _percentile_range(data)
        
        if pr is None:
            return LayerResult(
                layer_name="Physical Range",
                layer_number=5,
                confidence_contribution=0,
                identified_type=None,
                reasoning="Insufficient valid data for range check",
                passed=False
            ), {}
        
        data_low, data_high = pr
        data_mean = np.nanmean(data)
        
        matches = {}
        reasoning_parts = []
        
        for curve_type, ranges in PHYSICAL_RANGES.items():
            phys_min = ranges['min']
            phys_max = ranges['max']
            typ_min = ranges['typical_min']
            typ_max = ranges['typical_max']
            
            # Check if data is within physical limits
            if phys_min <= data_low and data_high <= phys_max:
                # Check if data is within typical range (full points)
                if typ_min <= data_mean <= typ_max:
                    matches[curve_type] = SCORE_WEIGHTS["range"]  # 25 pts
                    reasoning_parts.append(f"{curve_type.value}:TYPICAL")
                else:
                    # Within physical but not typical (partial points)
                    matches[curve_type] = SCORE_WEIGHTS["range"] * 0.6  # 15 pts
                    reasoning_parts.append(f"{curve_type.value}:PHYSICAL")
        
        best_match = max(matches, key=matches.get) if matches else None
        
        return LayerResult(
            layer_name="Physical Range",
            layer_number=5,
            confidence_contribution=matches.get(best_match, 0) if best_match else 0,
            identified_type=best_match,
            reasoning=f"Range [{data_low:.2f}, {data_high:.2f}], mean={data_mean:.2f}. Matches: {reasoning_parts[:3]}",
            passed=len(matches) > 0
        ), matches
    
    def _generate_explanation_with_scores(
        self, 
        layer_results: List[LayerResult],
        identified_type: CurveType,
        raw_score: float,
        score_breakdown: Dict[str, float]
    ) -> str:
        """Generate human-readable explanation with score breakdown."""
        if identified_type == CurveType.UNKNOWN:
            return f"Could not identify curve (score: {raw_score:.0f}/100). Low confidence across all layers."
        
        # Build score breakdown string
        breakdown_parts = []
        for category, pts in score_breakdown.items():
            if pts > 0:
                breakdown_parts.append(f"{category}:{pts:.0f}")
        
        passed_layers = [lr for lr in layer_results if lr.passed]
        key_reasons = [lr.reasoning for lr in passed_layers[:2]]
        
        explanation = f"Identified as {identified_type.value} (score: {raw_score:.0f}/100). "
        if breakdown_parts:
            explanation += f"Points: {', '.join(breakdown_parts)}. "
        if key_reasons:
            explanation += f"Evidence: {'; '.join(key_reasons)}"
        
        return explanation
    
    def _layer_1_header_reading(self, description: str, mnemonic: str) -> LayerResult:
        """Layer 1: Extract human intent from curve description."""
        description_lower = (description or '').lower()
        
        # Map description keywords to curve types
        desc_patterns = {
            CurveType.GR: ['gamma', 'gr ', 'natural radioactivity'],
            CurveType.RES_DEEP: ['deep resistivity', 'true resistivity', 'formation resistivity'],
            CurveType.RES_MED: ['medium resistivity', 'intermediate'],
            CurveType.RES_SHAL: ['shallow resistivity', 'invaded zone', 'flushed'],
            CurveType.DENS: ['bulk density', 'formation density'],
            CurveType.NEUT: ['neutron porosity', 'thermal neutron', 'hydrogen index'],
            CurveType.SONIC: ['sonic', 'acoustic', 'compressional slowness'],
            CurveType.CALIPER: ['caliper', 'borehole size', 'hole diameter'],
            CurveType.SP: ['spontaneous potential', 'self potential'],
            CurveType.PEF: ['photoelectric', 'pe factor'],
        }
        
        identified = None
        for curve_type, patterns in desc_patterns.items():
            for pattern in patterns:
                if pattern in description_lower:
                    identified = curve_type
                    break
            if identified:
                break
        
        return LayerResult(
            layer_name="Header Reading",
            layer_number=1,
            confidence_contribution=self.layer_weights[1] if identified else 0.0,
            identified_type=identified,
            reasoning=f"Description: '{description[:50]}...'" if description else "No description",
            passed=identified is not None
        )
    
    def _layer_2_keyword_nlp(self, mnemonic: str, description: str) -> LayerResult:
        """Layer 2: Keyword/NLP matching."""
        combined_text = f"{mnemonic} {description or ''}".lower()
        
        matches = []
        for curve_type, config in CURVE_PATTERNS.items():
            score = 0
            for keyword in config['keywords']:
                if keyword in combined_text:
                    score += 0.3
            if score > 0:
                matches.append((curve_type, min(score, 1.0)))
        
        return LayerResult(
            layer_name="Keyword / NLP",
            layer_number=2,
            confidence_contribution=self.layer_weights[2],
            identified_type=matches if matches else None,
            reasoning=f"Keyword matches: {len(matches)} types",
            passed=len(matches) > 0
        )
    
    def _layer_3_mnemonic_matching(self, mnemonic: str) -> LayerResult:
        """Layer 3: Industry standard mnemonic matching."""
        mnemonic_upper = mnemonic.upper().strip()
        
        best_match = None
        best_score = 0
        
        for curve_type, config in CURVE_PATTERNS.items():
            for std_mnemonic in config['mnemonics']:
                # Exact match
                if mnemonic_upper == std_mnemonic.upper():
                    return LayerResult(
                        layer_name="Mnemonic Matching",
                        layer_number=3,
                        confidence_contribution=self.layer_weights[3],
                        identified_type=curve_type,
                        reasoning=f"Exact match: {mnemonic} = {std_mnemonic}",
                        passed=True
                    )
                
                # Partial/contains match
                if std_mnemonic.upper() in mnemonic_upper or mnemonic_upper in std_mnemonic.upper():
                    score = len(std_mnemonic) / max(len(mnemonic), len(std_mnemonic))
                    if score > best_score:
                        best_score = score
                        best_match = curve_type
        
        if best_match and best_score > 0.5:
            return LayerResult(
                layer_name="Mnemonic Matching",
                layer_number=3,
                confidence_contribution=self.layer_weights[3] * best_score,
                identified_type=best_match,
                reasoning=f"Partial match: {mnemonic} (score: {best_score:.2f})",
                passed=True
            )
        
        return LayerResult(
            layer_name="Mnemonic Matching",
            layer_number=3,
            confidence_contribution=0.0,
            identified_type=None,
            reasoning=f"No standard mnemonic match for: {mnemonic}",
            passed=False
        )
    
    def _layer_4_unit_validation(
        self, 
        unit: str, 
        current_scores: Dict[CurveType, float]
    ) -> LayerResult:
        """Layer 4: Validate units against expected patterns."""
        unit_lower = (unit or '').lower().strip()
        
        validated_types = []
        for curve_type, config in UNIT_PATTERNS.items():
            if current_scores.get(curve_type, 0) > 0:
                valid_units = [u.lower() for u in config['valid_units']]
                if any(vu in unit_lower or unit_lower in vu for vu in valid_units):
                    validated_types.append(curve_type)
                    current_scores[curve_type] *= 1.2  # Boost confidence
                elif unit_lower:  # Unit present but doesn't match
                    current_scores[curve_type] *= 0.8  # Reduce confidence
        
        return LayerResult(
            layer_name="Unit Validation",
            layer_number=4,
            confidence_contribution=self.layer_weights[4] if validated_types else 0.0,
            identified_type=validated_types[0] if validated_types else None,
            reasoning=f"Unit '{unit}' validated for {len(validated_types)} types",
            passed=len(validated_types) > 0
        )
    
    def _layer_5_physical_range(
        self, 
        data: np.ndarray, 
        current_scores: Dict[CurveType, float]
    ) -> LayerResult:
        """Layer 5: Apply laws of physics - validate data ranges."""
        valid_data = data[~np.isnan(data)]
        if len(valid_data) == 0:
            return LayerResult(
                layer_name="Physical Range",
                layer_number=5,
                confidence_contribution=0.0,
                identified_type=None,
                reasoning="No valid data for range check",
                passed=False
            )
        
        data_min = np.percentile(valid_data, 1)  # Avoid extreme outliers
        data_max = np.percentile(valid_data, 99)
        data_mean = np.mean(valid_data)
        
        validated_types = []
        for curve_type, ranges in PHYSICAL_RANGES.items():
            if current_scores.get(curve_type, 0) > 0:
                # Check if data falls within physical limits
                if ranges['min'] <= data_min and data_max <= ranges['max']:
                    # Check if data is in typical range
                    if ranges['typical_min'] <= data_mean <= ranges['typical_max']:
                        validated_types.append((curve_type, 1.0))
                        current_scores[curve_type] *= 1.3  # Strong boost
                    else:
                        validated_types.append((curve_type, 0.7))
                        current_scores[curve_type] *= 1.1  # Mild boost
                else:
                    current_scores[curve_type] *= 0.5  # Strong penalty
        
        # Try to identify unknown curves by range alone
        if not validated_types:
            for curve_type, ranges in PHYSICAL_RANGES.items():
                if ranges['typical_min'] <= data_mean <= ranges['typical_max']:
                    if ranges['min'] <= data_min and data_max <= ranges['max']:
                        current_scores[curve_type] += 0.1
                        validated_types.append((curve_type, 0.3))
        
        best_type = validated_types[0][0] if validated_types else None
        
        return LayerResult(
            layer_name="Physical Range",
            layer_number=5,
            confidence_contribution=self.layer_weights[5] if validated_types else 0.0,
            identified_type=best_type,
            reasoning=f"Data range [{data_min:.2f}, {data_max:.2f}], mean={data_mean:.2f}",
            passed=len(validated_types) > 0
        )
    
    def _layer_6_statistical_shape(
        self, 
        data: np.ndarray,
        current_scores: Dict[CurveType, float]
    ) -> LayerResult:
        """Layer 6: Identify curve behavior patterns using statistics."""
        data = _ensure_array(data)  # Fix scalar bug
        # Replace null values before analysis
        data = data.astype(float)
        for null_val in NULL_VALUES:
            data = np.where(data == null_val, np.nan, data)
        valid_data = data[~np.isnan(data)]
        if len(valid_data) < 10:
            return LayerResult(
                layer_name="Statistical Shape",
                layer_number=6,
                confidence_contribution=0.0,
                identified_type=None,
                reasoning="Insufficient data for statistical analysis",
                passed=False
            )
        
        # Calculate statistical features
        stats = {
            'mean': np.mean(valid_data),
            'std': np.std(valid_data),
            'skewness': self._calculate_skewness(valid_data),
            'kurtosis': self._calculate_kurtosis(valid_data),
            'cv': np.std(valid_data) / np.mean(valid_data) if np.mean(valid_data) != 0 else 0,
        }
        
        insights = []
        
        # GR: Often right-skewed (shale spikes), high variability
        if current_scores.get(CurveType.GR, 0) > 0:
            if stats['skewness'] > 0 and stats['cv'] > 0.3:
                current_scores[CurveType.GR] *= 1.2
                insights.append("GR-like: positive skew, high variability")
        
        # Density: Usually tight distribution around 2.4-2.65 g/cc
        if current_scores.get(CurveType.DENS, 0) > 0:
            if stats['cv'] < 0.15 and 2.0 < stats['mean'] < 2.8:
                current_scores[CurveType.DENS] *= 1.2
                insights.append("DENS-like: tight distribution")
        
        # Resistivity: Often log-normal (high variability, right-skewed)
        for res_type in [CurveType.RES_DEEP, CurveType.RES_MED, CurveType.RES_SHAL]:
            if current_scores.get(res_type, 0) > 0:
                if stats['skewness'] > 1 and stats['cv'] > 0.5:
                    current_scores[res_type] *= 1.15
                    insights.append(f"{res_type.value}-like: log-normal pattern")
        
        # Neutron: Moderate variability, can be negative (gas effect)
        if current_scores.get(CurveType.NEUT, 0) > 0:
            if np.min(valid_data) < 0:
                current_scores[CurveType.NEUT] *= 1.3
                insights.append("NEUT-like: has negative values (gas effect)")
        
        return LayerResult(
            layer_name="Statistical Shape",
            layer_number=6,
            confidence_contribution=self.layer_weights[6] if insights else 0.0,
            identified_type=None,
            reasoning=f"Stats: CV={stats['cv']:.2f}, skew={stats['skewness']:.2f}. {'; '.join(insights)}",
            passed=len(insights) > 0
        )
    
    def _layer_7_detect_tool_context(self, las) -> ToolContext:
        """Layer 7: Detect logging tool context from file."""
        all_mnemonics = ' '.join([c.mnemonic.upper() for c in las.curves])
        all_descriptions = ' '.join([c.descr.lower() for c in las.curves if c.descr])
        
        # Check for LWD indicators
        lwd_patterns = TOOL_CONTEXT_PATTERNS[ToolContext.LWD]
        lwd_score = sum(1 for m in lwd_patterns['mnemonics'] if m in all_mnemonics)
        lwd_score += sum(1 for k in lwd_patterns['keywords'] if k in all_descriptions)
        
        # Check for wireline indicators
        wl_patterns = TOOL_CONTEXT_PATTERNS[ToolContext.WIRELINE]
        wl_score = sum(1 for m in wl_patterns['mnemonics'] if m in all_mnemonics)
        wl_score += sum(1 for k in wl_patterns['keywords'] if k in all_descriptions)
        
        if lwd_score > wl_score and lwd_score > 0:
            return ToolContext.LWD
        elif wl_score > 0:
            return ToolContext.WIRELINE
        return ToolContext.UNKNOWN
    
    def _layer_7_tool_context_adjustment(
        self, 
        mnemonic: str,
        current_scores: Dict[CurveType, float],
        tool_context: ToolContext
    ) -> LayerResult:
        """Layer 7: Adjust scores based on tool context."""
        mnemonic_upper = mnemonic.upper()
        adjustments = []
        
        if tool_context == ToolContext.LWD:
            prefs = TOOL_CONTEXT_PATTERNS[ToolContext.LWD]['resistivity_preference']
            if any(p in mnemonic_upper for p in prefs):
                for res_type in [CurveType.RES_DEEP, CurveType.RES_MED]:
                    if current_scores.get(res_type, 0) > 0:
                        current_scores[res_type] *= 1.2
                        adjustments.append(f"LWD boost for {res_type.value}")
        
        elif tool_context == ToolContext.WIRELINE:
            prefs = TOOL_CONTEXT_PATTERNS[ToolContext.WIRELINE]['resistivity_preference']
            if any(p in mnemonic_upper for p in prefs):
                for res_type in [CurveType.RES_DEEP, CurveType.RES_MED]:
                    if current_scores.get(res_type, 0) > 0:
                        current_scores[res_type] *= 1.2
                        adjustments.append(f"Wireline boost for {res_type.value}")
        
        return LayerResult(
            layer_name="Tool Context",
            layer_number=7,
            confidence_contribution=self.layer_weights[7] if adjustments else 0.0,
            identified_type=None,
            reasoning=f"Context: {tool_context.value}. {'; '.join(adjustments) if adjustments else 'No adjustments'}",
            passed=len(adjustments) > 0
        )
    
    def _layer_8_cross_curve_checks(
        self, 
        curve_results: Dict[str, CurveIdentificationResult],
        las
    ) -> List[str]:
        """Layer 8: Cross-curve validation checks."""
        insights = []
        
        # Get identified curves
        gr_curve = None
        res_curve = None
        dens_curve = None
        neut_curve = None
        
        for mnemonic, result in curve_results.items():
            if result.identified_type == CurveType.GR:
                gr_curve = _ensure_array(las[mnemonic].data)
            elif result.identified_type == CurveType.RES_DEEP:
                res_curve = _ensure_array(las[mnemonic].data)
            elif result.identified_type == CurveType.DENS:
                dens_curve = _ensure_array(las[mnemonic].data)
            elif result.identified_type == CurveType.NEUT:
                neut_curve = _ensure_array(las[mnemonic].data)
        
        # Check GR-Resistivity relationship (High GR + Low Rt → Shale)
        if gr_curve is not None and res_curve is not None:
            # Ensure arrays are the same length for correlation
            min_len = min(len(gr_curve), len(res_curve))
            if min_len > 10:
                gr_data = gr_curve[:min_len].astype(float)
                res_data = res_curve[:min_len].astype(float)
                # Create valid mask - ensure it's a proper boolean array
                valid_mask = np.array(~(np.isnan(gr_data) | np.isnan(res_data)), dtype=bool)
                valid_count = int(np.sum(valid_mask))
                if valid_count > 10:
                    # Use np.where for safe extraction
                    valid_indices = np.where(valid_mask)[0]
                    gr_valid = gr_data[valid_indices]
                    res_valid = res_data[valid_indices]
                    correlation = np.corrcoef(gr_valid, res_valid)[0, 1]
                    if not np.isnan(correlation) and correlation < -0.3:
                        insights.append(f"GR-Resistivity inverse correlation ({correlation:.2f}): typical shale response")
        
        # Check Density-Neutron crossover patterns
        if dens_curve is not None and neut_curve is not None:
            # Ensure arrays are the same length
            min_len = min(len(dens_curve), len(neut_curve))
            if min_len > 10:
                dens_data = dens_curve[:min_len].astype(float)
                neut_data = neut_curve[:min_len].astype(float)
                # Create valid mask - ensure it's a proper boolean array
                valid_mask = np.array(~(np.isnan(dens_data) | np.isnan(neut_data)), dtype=bool)
                valid_count = int(np.sum(valid_mask))
                if valid_count > 10:
                    # Use np.where for safe extraction
                    valid_indices = np.where(valid_mask)[0]
                    dens_valid = dens_data[valid_indices]
                    neut_valid = neut_data[valid_indices]
                    # Check for gas effect (crossover)
                    # In gas zones: density reads low, neutron reads very low
                    d_norm = (dens_valid - 2.0) / 0.5  # Normalize
                    n_norm = (neut_valid - 0.15) / 0.15  # Normalize
                    crossover_zones = np.sum(d_norm > n_norm)
                    if crossover_zones > 0.1 * valid_count:
                        insights.append("Density-Neutron crossover detected: possible gas zones")
        
        return insights
    
    def _layer_9_duplicate_resolution(
        self, 
        curve_results: Dict[str, CurveIdentificationResult],
        candidates_by_type: Dict[CurveType, List[Tuple[str, float]]]
    ) -> Tuple[Dict[str, CurveIdentificationResult], List[str]]:
        """
        Layer 9: Choose best curve when duplicates exist.
        
        For resistivity curves, applies DOI-aware classification first:
        - Extracts 2-3 digit DOI suffixes from mnemonics
        - Highest DOI = RES_DEEP, lowest = RES_SHAL
        - Falls back to confidence-based selection if no valid DOI found
        """
        duplicate_warnings = []
        
        # =================================================================
        # DOI-AWARE RESISTIVITY CLASSIFICATION (Pre-step for resistivity)
        # =================================================================
        # Collect all resistivity candidates across types
        resistivity_types = {CurveType.RES_DEEP, CurveType.RES_MED, CurveType.RES_SHAL}
        all_resistivity_candidates = []
        
        for curve_type in resistivity_types:
            if curve_type in candidates_by_type:
                for mnemonic, confidence in candidates_by_type[curve_type]:
                    all_resistivity_candidates.append((mnemonic, confidence, curve_type))
        
        # Apply DOI classification if we have multiple resistivity curves
        if len(all_resistivity_candidates) > 1:
            doi_classification = classify_resistivity_by_doi(all_resistivity_candidates)
            
            if doi_classification:
                # DOI classification succeeded - reclassify curves by DOI ordering
                doi_info = []
                for mnemonic, new_type_str in doi_classification.items():
                    doi_value = extract_valid_doi(mnemonic)
                    new_type = CurveType(new_type_str)  # Convert string to enum
                    
                    # Update the curve result with new classification
                    if mnemonic in curve_results:
                        old_type = curve_results[mnemonic].identified_type
                        curve_results[mnemonic].identified_type = new_type
                        curve_results[mnemonic].explanation += f" [DOI-reclassified: {old_type.value} → {new_type.value} based on DOI={doi_value}]"
                    
                    doi_info.append(f"{mnemonic}(DOI={doi_value})->{new_type.value}")
                
                # Rebuild candidates_by_type with DOI-based assignments
                for res_type in resistivity_types:
                    candidates_by_type[res_type] = []
                
                for mnemonic, new_type_str in doi_classification.items():
                    new_type = CurveType(new_type_str)
                    # Get original confidence
                    orig_confidence = next(
                        (conf for m, conf, _ in all_resistivity_candidates if m == mnemonic), 
                        0.0
                    )
                    candidates_by_type[new_type].append((mnemonic, orig_confidence))
                
                # Add DOI classification note to warnings
                duplicate_warnings.append(
                    f"DOI-aware resistivity classification applied: {', '.join(doi_info)}"
                )
        
        # =================================================================
        # STANDARD DUPLICATE RESOLUTION (for all curve types)
        # =================================================================
        for curve_type, candidates in candidates_by_type.items():
            if len(candidates) > 1:
                # Sort by confidence
                sorted_candidates = sorted(candidates, key=lambda x: -x[1])
                primary = sorted_candidates[0][0]
                
                warning = f"Multiple {curve_type.value} candidates: "
                warning += ", ".join([f"{c[0]} ({c[1]:.2f})" for c in sorted_candidates])
                warning += f". Selected: {primary}"
                duplicate_warnings.append(warning)
                
                # Mark non-primary as duplicates
                for mnemonic, _ in sorted_candidates[1:]:
                    if mnemonic in curve_results:
                        curve_results[mnemonic].is_duplicate = True
                        curve_results[mnemonic].duplicate_of = primary
                        curve_results[mnemonic].selected_as_primary = False
        
        return curve_results, duplicate_warnings
    
    def _generate_explanation(
        self, 
        layer_results: List[LayerResult],
        identified_type: CurveType,
        confidence: float
    ) -> str:
        """Layer 10: Generate human-readable explanation."""
        if identified_type == CurveType.UNKNOWN:
            return "Could not reliably identify curve type. Low confidence across all layers."
        
        passed_layers = [lr for lr in layer_results if lr.passed]
        key_reasons = [lr.reasoning for lr in passed_layers[:3]]
        
        explanation = f"Identified as {identified_type.value} (confidence: {confidence:.0%}). "
        explanation += "Key factors: " + "; ".join(key_reasons)
        
        return explanation
    
    def add_learning_exception(self, mnemonic: str, curve_type: CurveType):
        """Layer 11: Add a learning exception for future identification."""
        self.learning_exceptions[mnemonic] = curve_type
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return (np.sum((data - mean) ** 3) / n) / (std ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return (np.sum((data - mean) ** 4) / n) / (std ** 4) - 3


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def identify_curves_layered(las, file_type: str = "LAS") -> IdentificationReport:
    """
    Convenience function to identify curves using the 12-layer methodology.
    
    Args:
        las: lasio.LASFile object
        file_type: "LAS" or "DLIS"
        
    Returns:
        IdentificationReport with complete identification results
    """
    identifier = CurveIdentifier()
    ft = FileType.LAS if file_type.upper() == "LAS" else FileType.DLIS
    return identifier.identify_curves(las, ft)


def get_curve_mapping_layered(las, file_type: str = "LAS") -> Dict[str, str]:
    """
    Get curve mapping using the 12-layer methodology.
    Drop-in replacement for get_curve_mapping.
    
    Args:
        las: lasio.LASFile object
        file_type: "LAS" or "DLIS"
        
    Returns:
        Dictionary mapping curve types to mnemonics
    """
    report = identify_curves_layered(las, file_type)
    return report.mapping


def format_identification_report(report: IdentificationReport) -> str:
    """Format identification report as human-readable text."""
    lines = [
        "=" * 60,
        "CURVE IDENTIFICATION REPORT",
        "=" * 60,
        f"File Type: {report.file_type.value}",
        f"Tool Context: {report.tool_context.value}",
        f"Overall Confidence: {report.overall_confidence:.0%}",
        "",
        "CURVE MAPPING:",
        "-" * 40,
    ]
    
    for curve_type, mnemonic in report.mapping.items():
        result = report.curve_results.get(mnemonic)
        conf = result.confidence_score if result else 0
        lines.append(f"  {curve_type:12s} → {mnemonic:12s} ({conf:.0%})")
    
    if report.duplicate_warnings:
        lines.extend(["", "DUPLICATE WARNINGS:", "-" * 40])
        for warning in report.duplicate_warnings:
            lines.append(f"  ⚠ {warning}")
    
    if report.cross_curve_insights:
        lines.extend(["", "CROSS-CURVE INSIGHTS:", "-" * 40])
        for insight in report.cross_curve_insights:
            lines.append(f"  💡 {insight}")
    
    lines.extend(["", "DETAILED RESULTS:", "-" * 40])
    for mnemonic, result in report.curve_results.items():
        status = "✓" if result.selected_as_primary else "○"
        dup_note = f" (duplicate of {result.duplicate_of})" if result.is_duplicate else ""
        lines.append(f"  {status} {mnemonic}: {result.identified_type.value} ({result.confidence_score:.0%}){dup_note}")
        lines.append(f"      {result.explanation[:80]}...")
    
    lines.append("=" * 60)
    return "\n".join(lines)

