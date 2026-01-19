"""
Curve Mapping Module
Automatically maps LAS curve mnemonics to standard track types.

Supports two modes:
1. Simple mode: Fast mnemonic-only matching (original behavior)
2. Layered mode: Full 12-layer identification methodology

Use get_curve_mapping() for simple mode (default)
Use get_curve_mapping_advanced() for full layered identification
"""

# Standard curve mnemonics for automatic detection (simple mode)
# NOTE: Order matters for DEPTH - MD is prioritized over TVD/TVDSS
CURVE_MNEMONICS = {
    'DEPTH': ['MD', 'DEPT', 'DEPTH', 'DPTH', 'TVD', 'TVDSS', 'AHD'],

    'GR': ['GR', 'GRC', 'GR_EDTC', 'SGR', 'CGR', 'HCGR', 'ECGR', 'GRD', 'GRS', 'GRAM', 'NGR'],
    'RES_DEEP': ['RT', 'RDEP', 'RLLD', 'RLL3', 'ILD', 'RILD', 'LLD', 'RD', 'RDLA', 'RLA5', 
                 'AT90', 'AHT90', 'R40', 'R36', 'HDRS', 'ATRT', 'RTRUE'],
    'RES_MED': ['RM', 'RMED', 'ILM', 'RILM', 'RLM', 'AT60', 'AHT60', 'R24', 'R20', 'HMRS', 'RLA2'],
    'RES_SHAL': ['RS', 'RSHAL', 'MSFL', 'RXOZ', 'RXO', 'RLL1', 'SFL', 'LLS', 'RSFL', 'AT10', 
                 'AHT10', 'R10', 'LLHR', 'MLL'],
    'DENS': ['RHOB', 'RHOZ', 'DEN', 'ZDEN', 'BDEL', 'DENB', 'DENC', 'RHO', 'RHOF', 'DENS', 'HDENS'],
    'NEUT': ['NPHI', 'TNPH', 'NPHZ', 'CNPOR', 'NPOR', 'TNPHI', 'PHIN', 'NEU', 'NEUT', 
             'NPSS', 'NPLS', 'HNPO', 'APLC'],
    'SONIC': ['DT', 'DTC', 'DTCO', 'AC', 'SONIC', 'DTS', 'DTCS', 'DTSM', 'DT4P', 'DTLN'],
    'CALIPER': ['CALI', 'CAL', 'HCAL', 'DCAL', 'CALS', 'BS', 'C1', 'C2', 'LCAL', 'SCAL'],
    'SP': ['SP', 'SSP', 'PSP', 'SPR'],
    'PEF': ['PEF', 'PE', 'PEFZ', 'PEZ', 'PEFA'],
}

# Track configuration from CSV specification
TRACK_CONFIG = {
    'DEPTH': {
        'track': 0,
        'name': 'Measured Depth',
        'unit': 'm',
        'color': None,
        'min': None,
        'max': None,
        'scale': 'linear',
        'line_style': 'bold',
    },
    'GR': {
        'track': 1,
        'name': 'Gamma Ray',
        'unit': 'gAPI',
        'color': '#00AA00',  # Green
        'min': 0,
        'max': 150,
        'scale': 'linear',
        'line_style': 'solid',
    },
    'RES_DEEP': {
        'track': 2,
        'name': 'Resistivity Deep',
        'unit': 'ohm.m',
        'color': '#0000FF',  # Blue
        'min': 0.2,
        'max': 2000,
        'scale': 'log',
        'line_style': 'solid',
    },
    'NEUT': {
        'track': 3,
        'name': 'Neutron Porosity',
        'unit': 'v/v',
        'color': '#0000FF',  # Blue
        'min': -0.15,
        'max': 0.45,
        'scale': 'linear',
        'line_style': 'dashed',
    },
    'DENS': {
        'track': 3,
        'name': 'Bulk Density',
        'unit': 'g/cc',
        'color': '#FF0000',  # Red
        'min': 1.95,
        'max': 2.95,
        'scale': 'linear',
        'line_style': 'solid',
    },
}


def get_curve_mapping(las):
    """
    Maps LAS curves to standard tracks based on mnemonics.
    
    Args:
        las: lasio.LASFile object
        
    Returns:
        Dictionary mapping track types to curve names
    """
    keys = [k.upper() for k in las.keys()]
    original_keys = list(las.keys())
    
    mapping = {
        'DEPTH': None,
        'GR': None,
        'RES_DEEP': None,
        'RES_MED': None,
        'RES_SHAL': None,
        'DENS': None,
        'NEUT': None,
        'SONIC': None,
        'CALIPER': None,
        'SP': None,
        'PEF': None,
    }
    
    def find_match(mnemonics):
        """Find first matching curve from list of possible mnemonics."""
        for m in mnemonics:
            for i, k in enumerate(keys):
                if k == m.upper():
                    return original_keys[i]
        return None
    
    # Map each curve type
    for curve_type, mnemonics in CURVE_MNEMONICS.items():
        mapping[curve_type] = find_match(mnemonics)
    
    return mapping


def get_track_config(curve_type):
    """
    Get display configuration for a curve type.
    
    Args:
        curve_type: String like 'GR', 'RES_DEEP', etc.
        
    Returns:
        Dictionary with track configuration or None
    """
    return TRACK_CONFIG.get(curve_type)


def create_track_layout(mapping):
    """
    Create track layout structure for the log viewer.
    
    Args:
        mapping: Curve mapping dictionary from get_curve_mapping()
        
    Returns:
        List of track definitions for rendering
    """
    tracks = []
    
    # Track 0: Depth (always present)
    if mapping.get('DEPTH'):
        tracks.append({
            'id': 0,
            'type': 'depth',
            'curves': [{'name': mapping['DEPTH'], 'config': TRACK_CONFIG['DEPTH']}]
        })
    
    # Track 1: Gamma Ray
    if mapping.get('GR'):
        tracks.append({
            'id': 1,
            'type': 'gamma_ray',
            'title': 'GAMMA RAY',
            'curves': [{'name': mapping['GR'], 'config': TRACK_CONFIG['GR']}]
        })
    
    # Track 2: Resistivity
    res_curves = []
    if mapping.get('RES_DEEP'):
        res_curves.append({'name': mapping['RES_DEEP'], 'config': TRACK_CONFIG['RES_DEEP']})
    if mapping.get('RES_MED'):
        # Medium Resistivity -> Red
        med_config = TRACK_CONFIG['RES_DEEP'].copy()
        med_config['name'] = 'Resistivity Medium'
        med_config['color'] = '#FF0000' # Red
        res_curves.append({'name': mapping['RES_MED'], 'config': med_config})
    if mapping.get('RES_SHAL'):
        # Shallow Resistivity -> Orange
        shal_config = TRACK_CONFIG['RES_DEEP'].copy()
        shal_config['name'] = 'Resistivity Shallow'
        shal_config['color'] = '#FFA500' # Orange
        res_curves.append({'name': mapping['RES_SHAL'], 'config': shal_config})
    
    if res_curves:
        tracks.append({
            'id': 2,
            'type': 'resistivity',
            'title': 'RESISTIVITY',
            'curves': res_curves
        })
    
    # Track 3: Density-Neutron
    dn_curves = []
    if mapping.get('DENS'):
        dn_curves.append({'name': mapping['DENS'], 'config': TRACK_CONFIG['DENS']})
    if mapping.get('NEUT'):
        dn_curves.append({'name': mapping['NEUT'], 'config': TRACK_CONFIG['NEUT']})
    
    if dn_curves:
        tracks.append({
            'id': 3,
            'type': 'density_neutron',
            'title': 'DENSITY - NEUTRON',
            'curves': dn_curves
        })
    
    return tracks


# =============================================================================
# ADVANCED CURVE IDENTIFICATION (12-Layer Methodology)
# =============================================================================

def get_curve_mapping_advanced(las, use_layered=True):
    """
    Get curve mapping using the advanced 12-layer methodology.
    
    This provides comprehensive curve identification with:
    - Multi-layer validation (mnemonic, unit, range, statistics)
    - Confidence scoring
    - Duplicate resolution
    - Cross-curve validation
    - Explainable results
    
    Args:
        las: lasio.LASFile object
        use_layered: If True, use full 12-layer method; if False, fall back to simple
        
    Returns:
        tuple: (mapping_dict, identification_report)
            - mapping_dict: Standard mapping dictionary for compatibility
            - identification_report: Full IdentificationReport object with details
    """
    if not use_layered:
        return get_curve_mapping(las), None
    
    try:
        from .curve_identifier import identify_curves_layered, IdentificationReport
        
        report = identify_curves_layered(las, "LAS")
        
        # Convert to standard mapping format for backward compatibility
        mapping = {
            'DEPTH': None,
            'GR': None,
            'RES_DEEP': None,
            'RES_MED': None,
            'RES_SHAL': None,
            'DENS': None,
            'NEUT': None,
            'SONIC': None,
            'CALIPER': None,
            'SP': None,
            'PEF': None,
        }
        
        # Merge the identified mappings
        for curve_type, mnemonic in report.mapping.items():
            if curve_type in mapping:
                mapping[curve_type] = mnemonic
        
        return mapping, report
        
    except ImportError:
        # Fall back to simple mapping if curve_identifier not available
        return get_curve_mapping(las), None


def get_identification_summary(report):
    """
    Get a concise summary of curve identification results.
    
    Args:
        report: IdentificationReport from get_curve_mapping_advanced
        
    Returns:
        dict with summary statistics
    """
    if report is None:
        return None
    
    high_conf = sum(1 for r in report.curve_results.values() 
                   if r.confidence_score >= 0.7)
    med_conf = sum(1 for r in report.curve_results.values() 
                  if 0.4 <= r.confidence_score < 0.7)
    low_conf = sum(1 for r in report.curve_results.values() 
                  if r.confidence_score < 0.4)
    
    return {
        'total_curves': len(report.curve_results),
        'identified_curves': len(report.mapping),
        'high_confidence': high_conf,
        'medium_confidence': med_conf,
        'low_confidence': low_conf,
        'overall_confidence': report.overall_confidence,
        'tool_context': report.tool_context.value,
        'duplicates': len(report.duplicate_warnings),
        'insights': len(report.cross_curve_insights)
    }
