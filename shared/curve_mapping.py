"""
Curve Mapping Module
Automatically maps LAS curve mnemonics to standard track types.
"""

# Standard curve mnemonics for automatic detection
CURVE_MNEMONICS = {
    'DEPTH': ['DEPT', 'DEPTH', 'DPTH', 'MD', 'TVD'],
    'GR': ['GR', 'GRC', 'GR_EDTC', 'SGR', 'CGR', 'HCGR'],
    'RES_DEEP': ['RT', 'RDEP', 'RLLD', 'RLL3', 'ILD', 'RILD', 'LLD', 'RD'],
    'RES_MED': ['RM', 'ILM', 'RILS', 'RLM', 'RILM'],
    'RES_SHAL': ['RS', 'MSFL', 'RXOZ', 'RLL1', 'SFL', 'LLS', 'RSFL'],
    'DENS': ['RHOB', 'RHOZ', 'DEN', 'ZDEN', 'BDEL'],
    'NEUT': ['NPHI', 'TNPH', 'NPHZ', 'CNPOR', 'NPOR', 'TNPHI'],
    'SONIC': ['DT', 'DTC', 'DTCO', 'AC', 'SONIC'],
    'CALIPER': ['CALI', 'CAL', 'HCAL', 'DCAL'],
    'SP': ['SP', 'SSP'],
    'PEF': ['PEF', 'PE', 'PEFZ'],
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
    if mapping['DEPTH']:
        tracks.append({
            'id': 0,
            'type': 'depth',
            'curves': [{'name': mapping['DEPTH'], 'config': TRACK_CONFIG['DEPTH']}]
        })
    
    # Track 1: Gamma Ray
    if mapping['GR']:
        tracks.append({
            'id': 1,
            'type': 'gamma_ray',
            'title': 'GAMMA RAY',
            'curves': [{'name': mapping['GR'], 'config': TRACK_CONFIG['GR']}]
        })
    
    # Track 2: Resistivity
    res_curves = []
    if mapping['RES_DEEP']:
        res_curves.append({'name': mapping['RES_DEEP'], 'config': TRACK_CONFIG['RES_DEEP']})
    if mapping['RES_MED']:
        # Medium Resistivity -> Red
        med_config = TRACK_CONFIG['RES_DEEP'].copy()
        med_config['name'] = 'Resistivity Medium'
        med_config['color'] = '#FF0000' # Red
        res_curves.append({'name': mapping['RES_MED'], 'config': med_config})
    if mapping['RES_SHAL']:
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
    if mapping['DENS']:
        dn_curves.append({'name': mapping['DENS'], 'config': TRACK_CONFIG['DENS']})
    if mapping['NEUT']:
        dn_curves.append({'name': mapping['NEUT'], 'config': TRACK_CONFIG['NEUT']})
    
    if dn_curves:
        tracks.append({
            'id': 3,
            'type': 'density_neutron',
            'title': 'DENSITY - NEUTRON',
            'curves': dn_curves
        })
    
    return tracks
