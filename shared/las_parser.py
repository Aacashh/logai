"""
LAS File Parser
Handles loading and parsing of LAS (Log ASCII Standard) files.
"""

import lasio
import pandas as pd
import io

def load_las(file_obj):
    """
    Loads a LAS file from a file-like object or file path.
    
    Args:
        file_obj: File-like object, bytes, or file path string
        
    Returns:
        lasio.LASFile object
    """
    try:
        if isinstance(file_obj, str):
            # File path
            return lasio.read(file_obj)
        elif isinstance(file_obj, bytes):
            # Raw bytes
            str_data = file_obj.decode("utf-8", errors="ignore")
            return lasio.read(io.StringIO(str_data))
        else:
            # File-like object (e.g., from upload)
            bytes_data = file_obj.read()
            str_data = bytes_data.decode("utf-8", errors="ignore")
            return lasio.read(io.StringIO(str_data))
    except Exception as e:
        raise ValueError(f"Error loading LAS file: {e}")


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
            if curve.mnemonic.upper() in ['DEPT', 'DEPTH']:
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
    import numpy as np
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

