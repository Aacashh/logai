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
    layer_results: List[LayerResult]
    explanation: str
    alternative_types: List[Tuple[CurveType, float]]  # (type, confidence)
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
    CurveType.DEPTH: {
        'mnemonics': ['DEPT', 'DEPTH', 'DPTH', 'MD', 'TVD', 'TVDSS', 'AHD'],
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
        'valid_units': ['ohmm', 'ohm.m', 'ohm-m', 'ohmm', 'ohm', 'ohmm2/m'],
        'typical_range': (0.1, 10000)  # ohm.m
    },
    CurveType.RES_MED: {
        'valid_units': ['ohmm', 'ohm.m', 'ohm-m', 'ohmm', 'ohm', 'ohmm2/m'],
        'typical_range': (0.1, 10000)
    },
    CurveType.RES_SHAL: {
        'valid_units': ['ohmm', 'ohm.m', 'ohm-m', 'ohmm', 'ohm', 'ohmm2/m'],
        'typical_range': (0.1, 10000)
    },
    CurveType.DENS: {
        'valid_units': ['g/cc', 'g/cm3', 'gm/cc', 'kg/m3', 'g/c3'],
        'typical_range': (1.0, 3.5)  # g/cc
    },
    CurveType.NEUT: {
        'valid_units': ['v/v', 'pu', '%', 'frac', 'dec', 'm3/m3'],
        'typical_range': (-0.15, 0.60)  # v/v or fractional
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
    CurveType.NEUT: {'min': -0.20, 'max': 1.0, 'typical_min': -0.05, 'typical_max': 0.45},
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
        """Identify a single curve using all applicable layers."""
        
        mnemonic = curve.mnemonic
        unit = curve.unit
        description = curve.descr
        data = curve.data
        
        layer_results = []
        type_scores: Dict[CurveType, float] = {ct: 0.0 for ct in CurveType}
        
        # Layer 0: File & Context (already processed)
        layer_results.append(LayerResult(
            layer_name="File & Context",
            layer_number=0,
            confidence_contribution=self.layer_weights[0],
            identified_type=None,
            reasoning=f"File type: {file_context['type']}",
            passed=True
        ))
        
        # Layer 1: Header Reading
        l1_result = self._layer_1_header_reading(description, mnemonic)
        layer_results.append(l1_result)
        if l1_result.identified_type:
            type_scores[l1_result.identified_type] += l1_result.confidence_contribution
        
        # Layer 2: Keyword / NLP
        l2_result = self._layer_2_keyword_nlp(mnemonic, description)
        layer_results.append(l2_result)
        for ct, score in l2_result.identified_type or []:
            type_scores[ct] += score * self.layer_weights[2]
        
        # Layer 3: Mnemonic Matching
        l3_result = self._layer_3_mnemonic_matching(mnemonic)
        layer_results.append(l3_result)
        if l3_result.identified_type:
            type_scores[l3_result.identified_type] += l3_result.confidence_contribution
        
        # Layer 4: Unit Validation
        l4_result = self._layer_4_unit_validation(unit, type_scores)
        layer_results.append(l4_result)
        
        # Layer 5: Physical Range
        l5_result = self._layer_5_physical_range(data, type_scores)
        layer_results.append(l5_result)
        
        # Layer 6: Statistical Shape
        l6_result = self._layer_6_statistical_shape(data, type_scores)
        layer_results.append(l6_result)
        
        # Layer 7: Tool Context adjustment
        l7_result = self._layer_7_tool_context_adjustment(
            mnemonic, type_scores, tool_context
        )
        layer_results.append(l7_result)
        
        # Layer 10: Calculate final confidence
        best_type = max(type_scores, key=type_scores.get)
        best_score = type_scores[best_type]
        
        # Normalize confidence to 0-1 range
        confidence = min(1.0, max(0.0, best_score))
        
        # If confidence is too low, mark as unknown
        if confidence < 0.15:
            best_type = CurveType.UNKNOWN
        
        # Build alternatives
        alternatives = [
            (ct, score) for ct, score in sorted(
                type_scores.items(), key=lambda x: -x[1]
            ) if ct != best_type and score > 0.1
        ][:3]
        
        # Generate explanation
        explanation = self._generate_explanation(layer_results, best_type, confidence)
        
        # Layer 11: Check learning exceptions
        if mnemonic in self.learning_exceptions:
            learned_type = self.learning_exceptions[mnemonic]
            explanation += f" [Override from learned exception: {learned_type.value}]"
            best_type = learned_type
            confidence = max(confidence, 0.9)
        
        return CurveIdentificationResult(
            original_mnemonic=mnemonic,
            identified_type=best_type,
            confidence_score=confidence,
            layer_results=layer_results,
            explanation=explanation,
            alternative_types=alternatives
        )
    
    def _layer_0_file_context(self, las, file_type: FileType) -> dict:
        """Layer 0: Analyze file type and context."""
        context = {
            'type': file_type.value,
            'num_curves': len(las.curves),
            'has_header': bool(las.well),
        }
        return context
    
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
                gr_curve = las[mnemonic].data
            elif result.identified_type == CurveType.RES_DEEP:
                res_curve = las[mnemonic].data
            elif result.identified_type == CurveType.DENS:
                dens_curve = las[mnemonic].data
            elif result.identified_type == CurveType.NEUT:
                neut_curve = las[mnemonic].data
        
        # Check GR-Resistivity relationship (High GR + Low Rt → Shale)
        if gr_curve is not None and res_curve is not None:
            valid_mask = ~(np.isnan(gr_curve) | np.isnan(res_curve))
            if np.sum(valid_mask) > 10:
                correlation = np.corrcoef(gr_curve[valid_mask], res_curve[valid_mask])[0, 1]
                if correlation < -0.3:
                    insights.append(f"GR-Resistivity inverse correlation ({correlation:.2f}): typical shale response")
        
        # Check Density-Neutron crossover patterns
        if dens_curve is not None and neut_curve is not None:
            valid_mask = ~(np.isnan(dens_curve) | np.isnan(neut_curve))
            if np.sum(valid_mask) > 10:
                # Check for gas effect (crossover)
                # In gas zones: density reads low, neutron reads very low
                d_norm = (dens_curve[valid_mask] - 2.0) / 0.5  # Normalize
                n_norm = (neut_curve[valid_mask] - 0.15) / 0.15  # Normalize
                crossover_zones = np.sum(d_norm > n_norm)
                if crossover_zones > 0.1 * np.sum(valid_mask):
                    insights.append("Density-Neutron crossover detected: possible gas zones")
        
        return insights
    
    def _layer_9_duplicate_resolution(
        self, 
        curve_results: Dict[str, CurveIdentificationResult],
        candidates_by_type: Dict[CurveType, List[Tuple[str, float]]]
    ) -> Tuple[Dict[str, CurveIdentificationResult], List[str]]:
        """Layer 9: Choose best curve when duplicates exist."""
        duplicate_warnings = []
        
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

