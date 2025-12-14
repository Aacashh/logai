"""
AI Splicing Module
Automated merging of overlapping well log runs using Global Cross-Correlation
and Constrained Dynamic Time Warping (DTW).

This module implements an industry-standard hybrid approach:
1. Global Cross-Correlation for bulk shift detection
2. Sakoe-Chiba Band Constrained DTW for elastic correction
3. Batch auto-splicing with unit conversion for multiple log runs
"""

import numpy as np
import pandas as pd
from scipy import signal
from typing import Tuple, Dict, Optional, NamedTuple, List, Callable
from dataclasses import dataclass, field


# Default parameters
DEFAULT_GRID_STEP = 0.1524  # meters (0.5 feet)
DEFAULT_SEARCH_WINDOW = 20.0  # meters
DEFAULT_DTW_WINDOW = 5.0  # meters


@dataclass
class SplicingResult:
    """Container for splicing operation results."""
    # Final merged data
    merged_depth: np.ndarray
    merged_signal: np.ndarray
    
    # Corrected deep log
    corrected_deep_depth: np.ndarray
    corrected_deep_signal: np.ndarray
    
    # Metrics
    bulk_shift_meters: float
    dtw_cost: float
    overlap_start: float
    overlap_end: float
    splice_point: float
    
    # Correction curve (for QC plotting)
    correction_depth: np.ndarray
    correction_delta: np.ndarray  # original_depth - corrected_depth


@dataclass
class PreprocessedSignal:
    """Container for preprocessed signal data."""
    depth: np.ndarray
    signal_raw: np.ndarray
    signal_normalized: np.ndarray
    signal_for_correlation: np.ndarray  # NaN-filled version
    mean: float
    std: float


@dataclass
class PreprocessedLAS:
    """Container for a preprocessed LAS file with normalized units."""
    filename: str
    original_unit: str  # 'ft' or 'm'
    df: pd.DataFrame  # DataFrame normalized to meters
    start_depth: float  # in meters
    stop_depth: float  # in meters
    step: float  # in meters
    curves: List[str] = field(default_factory=list)
    well_name: str = ''  # Well name from header (sanitized)
    location: str = ''  # Location from LOC header


@dataclass
class WellGroupResult:
    """Container for well grouping operation results."""
    well_groups: Dict[str, List['PreprocessedLAS']]  # {well_name: [files]}
    duplicate_warnings: List[str]  # List of duplicate file warnings
    num_wells: int
    num_files_total: int


@dataclass
class BatchSpliceResult:
    """Container for batch splicing operation results."""
    composite_df: pd.DataFrame
    splice_log: List[str]
    file_summary: List[dict]
    total_depth_range: Tuple[float, float]
    num_files_processed: int
    correlation_curve: str


# =============================================================================
# PREPROCESSING ALGORITHMS
# =============================================================================

def create_common_grid(depth1: np.ndarray, depth2: np.ndarray, 
                       step: float = DEFAULT_GRID_STEP) -> np.ndarray:
    """
    Create a master depth grid covering the extent of both files.
    
    Args:
        depth1: Depth array from first log
        depth2: Depth array from second log
        step: Grid spacing in meters (default 0.1524m / 0.5ft)
        
    Returns:
        Common depth grid array
    """
    # Find overall extent
    min_depth = min(np.nanmin(depth1), np.nanmin(depth2))
    max_depth = max(np.nanmax(depth1), np.nanmax(depth2))
    
    # Round to nice values
    min_depth = np.floor(min_depth / step) * step
    max_depth = np.ceil(max_depth / step) * step
    
    return np.arange(min_depth, max_depth + step/2, step)


def resample_to_grid(depth: np.ndarray, signal: np.ndarray, 
                     target_grid: np.ndarray) -> np.ndarray:
    """
    Interpolate signal onto target depth grid.
    
    Handles NaN values by interpolating only valid data points.
    
    Args:
        depth: Original depth array
        signal: Original signal array
        target_grid: Target depth grid for resampling
        
    Returns:
        Resampled signal on target grid
    """
    # Remove NaN values for interpolation
    valid_mask = ~np.isnan(signal) & ~np.isnan(depth)
    
    if np.sum(valid_mask) < 2:
        # Not enough valid points
        return np.full_like(target_grid, np.nan)
    
    valid_depth = depth[valid_mask]
    valid_signal = signal[valid_mask]
    
    # Sort by depth (required for np.interp)
    sort_idx = np.argsort(valid_depth)
    valid_depth = valid_depth[sort_idx]
    valid_signal = valid_signal[sort_idx]
    
    # Interpolate - set NaN outside data range
    resampled = np.interp(target_grid, valid_depth, valid_signal,
                          left=np.nan, right=np.nan)
    
    return resampled


def zscore_normalize(signal: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Apply Z-Score normalization to handle amplitude differences between tools.
    
    Formula: (x - mean) / std
    
    NaN values are ignored during mean/std calculation.
    
    Args:
        signal: Input signal array
        
    Returns:
        Tuple of (normalized_signal, mean, std)
    """
    # Calculate stats ignoring NaN
    mean = np.nanmean(signal)
    std = np.nanstd(signal)
    
    # Avoid division by zero
    if std == 0 or np.isnan(std):
        std = 1.0
    
    # Normalize (preserves NaN positions)
    normalized = (signal - mean) / std
    
    return normalized, mean, std


def prepare_for_correlation(signal: np.ndarray) -> np.ndarray:
    """
    Prepare signal for correlation/DTW by filling NaN with 0 (mean value post-normalization).
    
    This is done ONLY for the mathematical operations, not the actual data.
    
    Args:
        signal: Z-score normalized signal (should have mean ~0)
        
    Returns:
        Signal with NaN replaced by 0
    """
    result = signal.copy()
    result[np.isnan(result)] = 0.0
    return result


def preprocess_signal(depth: np.ndarray, signal: np.ndarray,
                      target_grid: np.ndarray) -> PreprocessedSignal:
    """
    Full preprocessing pipeline for a single signal.
    
    Args:
        depth: Original depth array
        signal: Original signal array  
        target_grid: Common depth grid
        
    Returns:
        PreprocessedSignal with all variants of the data
    """
    # Resample to common grid
    resampled = resample_to_grid(depth, signal, target_grid)
    
    # Z-score normalize
    normalized, mean, std = zscore_normalize(resampled)
    
    # Prepare for correlation (fill NaN with 0)
    for_correlation = prepare_for_correlation(normalized)
    
    return PreprocessedSignal(
        depth=target_grid,
        signal_raw=resampled,
        signal_normalized=normalized,
        signal_for_correlation=for_correlation,
        mean=mean,
        std=std
    )


# =============================================================================
# STAGE 1: GLOBAL ALIGNMENT (BULK SHIFT)
# =============================================================================

def find_global_shift(reference: np.ndarray, target: np.ndarray,
                      depth_step: float, max_search_meters: float = DEFAULT_SEARCH_WINDOW
                      ) -> Tuple[float, np.ndarray]:
    """
    Use cross-correlation to find optimal global lag between two signals.
    
    Args:
        reference: Reference (shallow) signal, NaN-filled and normalized
        target: Target (deep) signal, NaN-filled and normalized
        depth_step: Depth step of the grid in meters
        max_search_meters: Maximum lag search range in meters (+/-)
        
    Returns:
        Tuple of (shift_in_meters, correlation_array)
        Positive shift means target needs to be shifted UP (shallower)
    """
    # Perform cross-correlation
    correlation = signal.correlate(reference, target, mode='full')
    
    # The lag array: negative means target is ahead of reference
    n = len(reference)
    lags = np.arange(-(n-1), n)
    lag_meters = lags * depth_step
    
    # Limit search to specified window
    max_samples = int(max_search_meters / depth_step)
    center = n - 1  # Zero lag position
    search_start = max(0, center - max_samples)
    search_end = min(len(correlation), center + max_samples + 1)
    
    # Find peak within window
    search_region = correlation[search_start:search_end]
    peak_idx_local = np.argmax(search_region)
    peak_idx_global = search_start + peak_idx_local
    
    # Convert to meters
    shift_samples = lags[peak_idx_global]
    shift_meters = shift_samples * depth_step
    
    return shift_meters, correlation


def apply_bulk_shift(depth: np.ndarray, shift_meters: float) -> np.ndarray:
    """
    Apply bulk depth shift to a log.
    
    Args:
        depth: Original depth array
        shift_meters: Shift to apply (positive = shift shallower)
        
    Returns:
        Shifted depth array
    """
    return depth - shift_meters


# =============================================================================
# STAGE 2: ELASTIC CORRECTION (CONSTRAINED DTW)
# =============================================================================

def constrained_dtw(x: np.ndarray, y: np.ndarray, 
                    window_size: int) -> Tuple[np.ndarray, float]:
    """
    Constrained Dynamic Time Warping with Sakoe-Chiba Band.
    
    Standard DTW is too aggressive and will warp geological features 
    unrealistically. The Sakoe-Chiba constraint forces the warp path 
    to stay near the diagonal.
    
    This is a pure NumPy implementation avoiding C-dependencies like fastdtw.
    
    Args:
        x: Reference signal (1D array)
        y: Target signal (1D array)
        window_size: Maximum deviation from diagonal in samples
        
    Returns:
        Tuple of (cost_matrix, total_cost)
    """
    r, c = len(x), len(y)
    
    # Initialize cost matrix with infinity
    D = np.full((r + 1, c + 1), np.inf)
    D[0, 0] = 0
    
    # Fill cost matrix with Sakoe-Chiba band constraint
    for i in range(1, r + 1):
        j_min = max(1, i - window_size)
        j_max = min(c, i + window_size)
        for j in range(j_min, j_max + 1):
            dist = (x[i-1] - y[j-1]) ** 2
            D[i, j] = dist + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    
    total_cost = D[r, c]
    
    return D, total_cost


def backtrack_dtw_path(D: np.ndarray) -> list:
    """
    Extract optimal warping path from DTW cost matrix via backtracking.
    
    Args:
        D: Cost matrix from constrained_dtw
        
    Returns:
        List of (i, j) tuples representing the optimal path
    """
    r, c = D.shape[0] - 1, D.shape[1] - 1
    path = [(r, c)]
    
    i, j = r, c
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            # Find minimum predecessor
            candidates = [
                (D[i-1, j-1], i-1, j-1),  # diagonal
                (D[i-1, j], i-1, j),       # vertical
                (D[i, j-1], i, j-1),       # horizontal
            ]
            _, i, j = min(candidates, key=lambda x: x[0])
        path.append((i, j))
    
    path.reverse()
    return path


def dtw_path_to_depth_mapping(path: list, 
                               ref_depth: np.ndarray,
                               target_depth: np.ndarray) -> Dict[float, float]:
    """
    Convert DTW path to depth mapping dictionary.
    
    Args:
        path: List of (ref_idx, target_idx) from backtracking
        ref_depth: Reference depth array
        target_depth: Target (shifted) depth array
        
    Returns:
        Dictionary mapping {original_target_depth: corrected_depth}
    """
    mapping = {}
    
    for ref_idx, tgt_idx in path:
        if ref_idx > 0 and tgt_idx > 0:  # Skip (0,0) start
            ref_idx -= 1  # Adjust for 1-indexing in DTW
            tgt_idx -= 1
            
            if ref_idx < len(ref_depth) and tgt_idx < len(target_depth):
                original_depth = target_depth[tgt_idx]
                corrected_depth = ref_depth[ref_idx]
                mapping[original_depth] = corrected_depth
    
    return mapping


def apply_dtw_correction(depth: np.ndarray, signal: np.ndarray,
                         depth_mapping: Dict[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply DTW elastic warp using depth mapping.
    
    Interpolates the sparse DTW mapping to all depth points.
    
    Args:
        depth: Original depth array
        signal: Original signal array
        depth_mapping: {original_depth: corrected_depth} from DTW
        
    Returns:
        Tuple of (corrected_depth, signal)
    """
    if not depth_mapping:
        return depth.copy(), signal.copy()
    
    # Convert mapping to arrays
    orig_depths = np.array(list(depth_mapping.keys()))
    corr_depths = np.array(list(depth_mapping.values()))
    
    # Sort by original depth
    sort_idx = np.argsort(orig_depths)
    orig_depths = orig_depths[sort_idx]
    corr_depths = corr_depths[sort_idx]
    
    # Interpolate correction for all depth points
    corrected_depth = np.interp(depth, orig_depths, corr_depths,
                                 left=depth[0] - (orig_depths[0] - corr_depths[0]),
                                 right=depth[-1] - (orig_depths[-1] - corr_depths[-1]))
    
    return corrected_depth, signal.copy()


# =============================================================================
# FINAL MERGING
# =============================================================================

def find_overlap_region(depth1: np.ndarray, depth2: np.ndarray
                        ) -> Tuple[float, float, float]:
    """
    Find the overlapping depth region between two logs.
    
    Args:
        depth1: Depth array from first (shallow) log
        depth2: Depth array from second (deep) log
        
    Returns:
        Tuple of (overlap_start, overlap_end, midpoint)
    """
    # Valid depth ranges
    min1, max1 = np.nanmin(depth1), np.nanmax(depth1)
    min2, max2 = np.nanmin(depth2), np.nanmax(depth2)
    
    # Overlap region
    overlap_start = max(min1, min2)
    overlap_end = min(max1, max2)
    
    if overlap_start >= overlap_end:
        raise ValueError(f"No overlap found between logs. "
                        f"Shallow: {min1:.1f}-{max1:.1f}m, "
                        f"Deep: {min2:.1f}-{max2:.1f}m")
    
    midpoint = (overlap_start + overlap_end) / 2
    
    return overlap_start, overlap_end, midpoint


def splice_at_midpoint(shallow_depth: np.ndarray, shallow_signal: np.ndarray,
                       deep_depth: np.ndarray, deep_signal: np.ndarray,
                       splice_point: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Concatenate two logs at the splice point.
    
    Top: Shallow Log (Depth < Midpoint)
    Bottom: Deep Log (Depth >= Midpoint)
    
    Args:
        shallow_depth: Shallow log depth array
        shallow_signal: Shallow log signal array
        deep_depth: Corrected deep log depth array
        deep_signal: Deep log signal array
        splice_point: Depth at which to splice
        
    Returns:
        Tuple of (merged_depth, merged_signal)
    """
    # Get shallow portion (above splice point)
    shallow_mask = shallow_depth < splice_point
    merged_depth_top = shallow_depth[shallow_mask]
    merged_signal_top = shallow_signal[shallow_mask]
    
    # Get deep portion (at and below splice point)
    deep_mask = deep_depth >= splice_point
    merged_depth_bottom = deep_depth[deep_mask]
    merged_signal_bottom = deep_signal[deep_mask]
    
    # Concatenate
    merged_depth = np.concatenate([merged_depth_top, merged_depth_bottom])
    merged_signal = np.concatenate([merged_signal_top, merged_signal_bottom])
    
    # Sort by depth
    sort_idx = np.argsort(merged_depth)
    merged_depth = merged_depth[sort_idx]
    merged_signal = merged_signal[sort_idx]
    
    return merged_depth, merged_signal


# =============================================================================
# MASTER ORCHESTRATION
# =============================================================================

def splice_logs(
    shallow_depth: np.ndarray,
    shallow_signal: np.ndarray,
    deep_depth: np.ndarray,
    deep_signal: np.ndarray,
    grid_step: float = DEFAULT_GRID_STEP,
    max_search_meters: float = DEFAULT_SEARCH_WINDOW,
    max_elastic_meters: float = DEFAULT_DTW_WINDOW,
    progress_callback: Optional[callable] = None
) -> SplicingResult:
    """
    Master function to perform full log splicing pipeline.
    
    Pipeline:
    1. Create common depth grid
    2. Resample and normalize both signals
    3. Find global shift via cross-correlation
    4. Apply bulk shift to deep log
    5. Find overlap region
    6. Apply constrained DTW elastic correction
    7. Splice logs at midpoint
    
    Args:
        shallow_depth: Shallow run depth array
        shallow_signal: Shallow run signal array (e.g., GR)
        deep_depth: Deep run depth array
        deep_signal: Deep run signal array
        grid_step: Resampling grid step in meters
        max_search_meters: Max search window for cross-correlation
        max_elastic_meters: Max elastic stretch for DTW (Sakoe-Chiba band)
        progress_callback: Optional callback(step_name, message) for UI updates
        
    Returns:
        SplicingResult with all outputs and metrics
    """
    def report(step, msg):
        if progress_callback:
            progress_callback(step, msg)
    
    # -------------------------------------------------------------------------
    # STEP 1: Preprocessing & Grid Alignment
    # -------------------------------------------------------------------------
    report("preprocessing", f"Creating common grid with {grid_step:.4f}m step...")
    
    common_grid = create_common_grid(shallow_depth, deep_depth, grid_step)
    
    report("preprocessing", f"Resampling logs to {len(common_grid)} points...")
    
    shallow_prep = preprocess_signal(shallow_depth, shallow_signal, common_grid)
    deep_prep = preprocess_signal(deep_depth, deep_signal, common_grid)
    
    report("preprocessing", 
           f"Z-score normalization applied. "
           f"Shallow: μ={shallow_prep.mean:.2f}, σ={shallow_prep.std:.2f}. "
           f"Deep: μ={deep_prep.mean:.2f}, σ={deep_prep.std:.2f}")
    
    # -------------------------------------------------------------------------
    # STEP 2: Global Shift Detection
    # -------------------------------------------------------------------------
    report("global_shift", f"Running cross-correlation (±{max_search_meters}m window)...")
    
    bulk_shift, correlation = find_global_shift(
        shallow_prep.signal_for_correlation,
        deep_prep.signal_for_correlation,
        grid_step,
        max_search_meters
    )
    
    shift_direction = "deeper" if bulk_shift > 0 else "shallower"
    report("global_shift", 
           f"Detected bulk shift: {abs(bulk_shift):.3f}m ({shift_direction})")
    
    # Apply bulk shift to deep log depths
    deep_depth_shifted = apply_bulk_shift(deep_depth, bulk_shift)
    
    # -------------------------------------------------------------------------
    # STEP 3: Find Overlap Region
    # -------------------------------------------------------------------------
    overlap_start, overlap_end, splice_point = find_overlap_region(
        shallow_depth, deep_depth_shifted
    )
    
    report("overlap", 
           f"Overlap region: {overlap_start:.1f}m to {overlap_end:.1f}m "
           f"({overlap_end - overlap_start:.1f}m). "
           f"Splice point: {splice_point:.1f}m")
    
    # -------------------------------------------------------------------------
    # STEP 4: Elastic Correction (DTW) on Overlap Region
    # -------------------------------------------------------------------------
    report("dtw", "Extracting overlap region for DTW...")
    
    # Re-preprocess with shifted deep depth for DTW
    deep_prep_shifted = preprocess_signal(deep_depth_shifted, deep_signal, common_grid)
    
    # Extract overlap indices
    overlap_mask = (common_grid >= overlap_start) & (common_grid <= overlap_end)
    shallow_overlap = shallow_prep.signal_for_correlation[overlap_mask]
    deep_overlap = deep_prep_shifted.signal_for_correlation[overlap_mask]
    depth_overlap = common_grid[overlap_mask]
    
    # DTW window in samples
    dtw_window_samples = int(max_elastic_meters / grid_step)
    dtw_window_samples = max(1, dtw_window_samples)  # At least 1
    
    report("dtw", 
           f"Running constrained DTW (Sakoe-Chiba band: ±{dtw_window_samples} samples / "
           f"±{max_elastic_meters}m)...")
    
    # Run DTW
    D, dtw_cost = constrained_dtw(shallow_overlap, deep_overlap, dtw_window_samples)
    
    # Backtrack to get path
    path = backtrack_dtw_path(D)
    
    report("dtw", 
           f"DTW complete. Total cost: {dtw_cost:.2f}. "
           f"Path length: {len(path)} points")
    
    # Convert path to depth mapping
    depth_mapping = dtw_path_to_depth_mapping(path, depth_overlap, depth_overlap)
    
    # -------------------------------------------------------------------------
    # STEP 5: Apply Corrections and Merge
    # -------------------------------------------------------------------------
    report("merge", "Applying elastic correction to overlap region...")
    
    # For overlap region: apply DTW correction
    # For non-overlap region: just use bulk-shifted depth
    
    # Resample original signals to common grid for merging
    shallow_on_grid = resample_to_grid(shallow_depth, shallow_signal, common_grid)
    deep_on_grid = resample_to_grid(deep_depth_shifted, deep_signal, common_grid)
    
    # Initialize corrected arrays
    corrected_deep_depth = common_grid.copy()
    corrected_deep_signal = deep_on_grid.copy()
    
    # Create correction delta array for QC plotting
    correction_delta = np.zeros_like(common_grid)
    
    if depth_mapping:
        # Convert depth_mapping to arrays for interpolation
        map_orig_depths = np.array(list(depth_mapping.keys()))
        map_corr_depths = np.array(list(depth_mapping.values()))
        
        # Sort by original depth
        sort_idx = np.argsort(map_orig_depths)
        map_orig_depths = map_orig_depths[sort_idx]
        map_corr_depths = map_corr_depths[sort_idx]
        
        # Get overlap region indices and depths
        overlap_idx = np.where(overlap_mask)[0]
        overlap_depths = common_grid[overlap_mask]
        
        if len(overlap_idx) > 0 and len(map_orig_depths) > 1:
            # Interpolate the depth correction for all points in overlap region
            # This gives us: for each grid point, what ORIGINAL deep depth should we sample from?
            # map_corr_depths = reference depths (what we want to align to)
            # map_orig_depths = original deep depths (where the values came from)
            
            # For each point in overlap: find what original depth maps to this corrected depth
            # We need the inverse mapping: given a target depth, what original depth to sample
            warped_source_depths = np.interp(
                overlap_depths,
                map_corr_depths,  # corrected/reference depths
                map_orig_depths,  # original deep depths
                left=map_orig_depths[0],
                right=map_orig_depths[-1]
            )
            
            # Now resample the deep signal at these warped source depths
            # Get deep signal values on the shifted depth grid
            deep_shifted_depths = deep_depth_shifted
            deep_shifted_signal = deep_signal
            
            # Sort for interpolation
            valid_mask = ~np.isnan(deep_shifted_signal)
            if np.sum(valid_mask) >= 2:
                valid_depths = deep_shifted_depths[valid_mask]
                valid_signal = deep_shifted_signal[valid_mask]
                sort_idx = np.argsort(valid_depths)
                valid_depths = valid_depths[sort_idx]
                valid_signal = valid_signal[sort_idx]
                
                # Resample deep signal at the warped depths
                warped_signal = np.interp(
                    warped_source_depths,
                    valid_depths,
                    valid_signal,
                    left=np.nan,
                    right=np.nan
                )
                
                # Apply the warped signal to the overlap region
                corrected_deep_signal[overlap_idx] = warped_signal
            
            # Compute correction delta for visualization
            correction_delta[overlap_idx] = overlap_depths - warped_source_depths
    
    report("merge", f"Splicing at {splice_point:.1f}m...")
    
    # Final splice
    merged_depth, merged_signal = splice_at_midpoint(
        common_grid, shallow_on_grid,
        common_grid, corrected_deep_signal,
        splice_point
    )
    
    report("complete", "Splicing complete!")
    
    return SplicingResult(
        merged_depth=merged_depth,
        merged_signal=merged_signal,
        corrected_deep_depth=common_grid,
        corrected_deep_signal=corrected_deep_signal,
        bulk_shift_meters=bulk_shift,
        dtw_cost=dtw_cost,
        overlap_start=overlap_start,
        overlap_end=overlap_end,
        splice_point=splice_point,
        correction_depth=common_grid,
        correction_delta=correction_delta
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def find_common_curves(las1_curves: list, las2_curves: list) -> list:
    """
    Find curves present in both LAS files.
    
    Args:
        las1_curves: List of curve names from first file
        las2_curves: List of curve names from second file
        
    Returns:
        List of common curve names
    """
    set1 = set(c.upper() for c in las1_curves)
    set2 = set(c.upper() for c in las2_curves)
    
    common = set1 & set2
    
    # Return in original case from first file
    return [c for c in las1_curves if c.upper() in common]


def get_recommended_correlation_curve(common_curves: list) -> Optional[str]:
    """
    Get recommended curve for correlation from list of available curves.
    
    Priority: GR > RHOB > NPHI > first available
    
    Args:
        common_curves: List of available curve names
        
    Returns:
        Recommended curve name or None
    """
    # Priority list (case-insensitive)
    priority = ['GR', 'GRC', 'SGR', 'CGR', 'RHOB', 'RHOZ', 'NPHI', 'TNPH']
    
    upper_curves = {c.upper(): c for c in common_curves}
    
    for p in priority:
        if p in upper_curves:
            return upper_curves[p]
    
    # Return first non-depth curve
    depth_names = {'DEPT', 'DEPTH', 'MD', 'TVD'}
    for c in common_curves:
        if c.upper() not in depth_names:
            return c
    
    return None


# =============================================================================
# BATCH AUTO-SPLICING WITH UNIT CONVERSION
# =============================================================================

# Standard null values to strip from top/bottom of logs
NULL_VALUES = [-999.25, -999, -9999, -9999.25, -999.2500, -999.00]

# Conversion factor: feet to meters
FT_TO_M = 0.3048

# Gap threshold for appending vs splicing (meters)
GAP_THRESHOLD = 5.0


def detect_las_units(las) -> str:
    """
    Detect depth units from LAS header.
    
    Checks STRT.FT, STRT.F, STRT.M patterns as specified by client data.
    
    Args:
        las: lasio.LASFile object
        
    Returns:
        'ft' for feet, 'm' for meters
    """
    try:
        if 'STRT' in las.well:
            unit = las.well.STRT.unit.upper()
            # Handle FT, F, FEET variations
            if unit in ['FT', 'F', 'FEET', 'FOOT']:
                return 'ft'
            elif unit in ['M', 'METER', 'METERS', 'METRE', 'METRES']:
                return 'm'
    except (AttributeError, KeyError):
        pass
    
    # Check STEP unit as fallback
    try:
        if 'STEP' in las.well:
            unit = las.well.STEP.unit.upper()
            if unit in ['FT', 'F', 'FEET', 'FOOT']:
                return 'ft'
            elif unit in ['M', 'METER', 'METERS', 'METRE', 'METRES']:
                return 'm'
    except (AttributeError, KeyError):
        pass
    
    # Check depth curve unit
    try:
        for curve in las.curves:
            if curve.mnemonic.upper() in ['DEPT', 'DEPTH', 'MD']:
                unit = curve.unit.upper()
                if unit in ['FT', 'F', 'FEET', 'FOOT']:
                    return 'ft'
                elif unit in ['M', 'METER', 'METERS', 'METRE', 'METRES']:
                    return 'm'
    except (AttributeError, KeyError):
        pass
    
    # Default to feet (common in US data)
    return 'ft'


def strip_null_padding(df: pd.DataFrame, depth_col: str = 'DEPTH') -> pd.DataFrame:
    """
    Strip leading and trailing rows where all non-depth columns are null.
    
    This removes padding that logging tools add at top/bottom of files.
    
    Args:
        df: DataFrame with well log data
        depth_col: Name of depth column
        
    Returns:
        DataFrame with null padding removed
    """
    # Get non-depth columns
    data_cols = [c for c in df.columns if c.upper() != depth_col.upper()]
    
    if not data_cols:
        return df
    
    # Create mask for rows with any valid (non-null) data
    # Replace known null values with NaN first
    df_check = df[data_cols].copy()
    for null_val in NULL_VALUES:
        df_check = df_check.replace(null_val, np.nan)
    
    # Find rows with at least one valid value
    valid_mask = df_check.notna().any(axis=1)
    
    if not valid_mask.any():
        return df
    
    # Find first and last valid indices
    first_valid = valid_mask.idxmax()
    last_valid = valid_mask[::-1].idxmax()
    
    # Slice to valid range
    return df.loc[first_valid:last_valid].reset_index(drop=True)


def convert_las_to_meters(las, df: pd.DataFrame, depth_col: str = 'DEPTH') -> Tuple[pd.DataFrame, float, float, float]:
    """
    Convert LAS data from feet to meters if necessary.
    
    Args:
        las: lasio.LASFile object (for header metadata)
        df: DataFrame with log data
        depth_col: Name of depth column
        
    Returns:
        Tuple of (converted_df, start_depth_m, stop_depth_m, step_m)
    """
    original_unit = detect_las_units(las)
    
    df_converted = df.copy()
    
    if original_unit == 'ft':
        # Convert depth column
        if depth_col in df_converted.columns:
            df_converted[depth_col] = df_converted[depth_col] * FT_TO_M
        
        # Get header values and convert
        try:
            start = float(las.well.STRT.value) * FT_TO_M
            stop = float(las.well.STOP.value) * FT_TO_M
            step = abs(float(las.well.STEP.value)) * FT_TO_M
        except (AttributeError, ValueError, TypeError):
            # Calculate from data if header is problematic
            start = df_converted[depth_col].min()
            stop = df_converted[depth_col].max()
            step = abs(df_converted[depth_col].diff().median())
    else:
        # Already in meters
        try:
            start = float(las.well.STRT.value)
            stop = float(las.well.STOP.value)
            step = abs(float(las.well.STEP.value))
        except (AttributeError, ValueError, TypeError):
            start = df[depth_col].min()
            stop = df[depth_col].max()
            step = abs(df[depth_col].diff().median())
    
    return df_converted, start, stop, step


def preprocess_las_files(
    las_file_objects: List,
    progress_callback: Optional[Callable[[str, str], None]] = None
) -> List[PreprocessedLAS]:
    """
    Preprocess multiple LAS files for batch splicing.
    
    This function:
    1. Loads each LAS file
    2. Detects and normalizes units to Meters
    3. Strips null padding from top/bottom
    4. Returns sorted list by start depth
    
    Args:
        las_file_objects: List of file-like objects or file paths
        progress_callback: Optional callback(step, message) for progress updates
        
    Returns:
        List of PreprocessedLAS objects sorted by start depth (shallowest first)
    """
    import lasio
    import io
    
    def report(step, msg):
        if progress_callback:
            progress_callback(step, msg)
    
    preprocessed = []
    
    report("preprocessing", f"Processing {len(las_file_objects)} files...")
    
    for i, file_obj in enumerate(las_file_objects):
        # Get filename
        if hasattr(file_obj, 'name'):
            filename = file_obj.name
        elif isinstance(file_obj, str):
            filename = file_obj.split('/')[-1]
        else:
            filename = f"File_{i+1}"
        
        report("preprocessing", f"Loading {filename}...")
        
        # Load LAS file
        try:
            if isinstance(file_obj, str):
                las = lasio.read(file_obj)
            elif isinstance(file_obj, bytes):
                str_data = file_obj.decode("utf-8", errors="ignore")
                las = lasio.read(io.StringIO(str_data))
            else:
                # File-like object (e.g., Streamlit UploadedFile)
                file_obj.seek(0)
                bytes_data = file_obj.read()
                str_data = bytes_data.decode("utf-8", errors="ignore")
                las = lasio.read(io.StringIO(str_data))
        except Exception as e:
            report("error", f"Failed to load {filename}: {str(e)}")
            continue
        
        # Detect original units
        original_unit = detect_las_units(las)
        report("preprocessing", f"{filename}: Detected unit = {original_unit.upper()}")
        
        # Convert to DataFrame
        df = las.df().reset_index()
        
        # Standardize depth column name
        depth_col = df.columns[0]  # First column is always depth in lasio
        if depth_col.upper() in ['DEPT', 'DEPTH', 'MD', 'TVD']:
            df = df.rename(columns={depth_col: 'DEPTH'})
        else:
            df = df.rename(columns={depth_col: 'DEPTH'})
        
        # Convert to meters
        df_meters, start_m, stop_m, step_m = convert_las_to_meters(las, df, 'DEPTH')
        
        if original_unit == 'ft':
            report("preprocessing", 
                   f"{filename}: Converted {start_m/FT_TO_M:.1f}-{stop_m/FT_TO_M:.1f} ft → "
                   f"{start_m:.1f}-{stop_m:.1f} m")
        
        # Strip null padding
        df_stripped = strip_null_padding(df_meters, 'DEPTH')
        
        # Check if DataFrame is empty after stripping
        if df_stripped.empty or len(df_stripped) == 0:
            report("warning", f"{filename}: No valid data rows after stripping null padding. Skipping file.")
            continue
        
        # Recalculate bounds after stripping
        actual_start = df_stripped['DEPTH'].min()
        actual_stop = df_stripped['DEPTH'].max()
        
        rows_stripped = len(df_meters) - len(df_stripped)
        if rows_stripped > 0:
            report("preprocessing", 
                   f"{filename}: Stripped {rows_stripped} null padding rows. "
                   f"Valid depth: {actual_start:.1f}-{actual_stop:.1f} m")
        
        # Handle null values in data columns
        for col in df_stripped.columns:
            if col != 'DEPTH':
                for null_val in NULL_VALUES:
                    df_stripped[col] = df_stripped[col].replace(null_val, np.nan)
        
        # Get available curves
        curves = [c for c in df_stripped.columns if c.upper() != 'DEPTH']
        
        preprocessed.append(PreprocessedLAS(
            filename=filename,
            original_unit=original_unit,
            df=df_stripped,
            start_depth=actual_start,
            stop_depth=actual_stop,
            step=step_m,
            curves=curves
        ))
    
    # Check if any files were successfully preprocessed
    if not preprocessed:
        report("error", "No valid LAS files could be preprocessed.")
        return preprocessed
    
    # Sort by start depth (shallowest first)
    preprocessed.sort(key=lambda x: x.start_depth)
    
    report("preprocessing", 
           f"Sorted {len(preprocessed)} files by depth. "
           f"Range: {preprocessed[0].start_depth:.1f}m to {preprocessed[-1].stop_depth:.1f}m")
    
    return preprocessed


def group_files_by_well(
    las_file_objects: List,
    progress_callback: Optional[Callable[[str, str], None]] = None
) -> WellGroupResult:
    """
    Group uploaded LAS files by their well name.
    
    This function:
    1. Loads each LAS file
    2. Extracts and sanitizes well name from header
    3. Detects duplicate files using fingerprint (filename + size + STRT + STOP)
    4. Groups files by well name
    5. Sorts each group by start depth (shallowest first)
    
    Args:
        las_file_objects: List of file-like objects or file paths
        progress_callback: Optional callback(step, message) for progress updates
        
    Returns:
        WellGroupResult with grouped files and duplicate warnings
    """
    import lasio
    import io
    import hashlib
    
    def report(step, msg):
        if progress_callback:
            progress_callback(step, msg)
    
    well_groups: Dict[str, List[PreprocessedLAS]] = {}
    duplicate_warnings: List[str] = []
    seen_fingerprints: Dict[str, set] = {}  # {well_name: {fingerprints}}
    
    report("grouping", f"Scanning {len(las_file_objects)} files for well identification...")
    
    for i, file_obj in enumerate(las_file_objects):
        # Get filename and file size
        if hasattr(file_obj, 'name'):
            filename = file_obj.name
        elif isinstance(file_obj, str):
            filename = file_obj.split('/')[-1]
        else:
            filename = f"File_{i+1}"
        
        # Get file size for fingerprint
        if hasattr(file_obj, 'size'):
            file_size = file_obj.size
        elif hasattr(file_obj, 'seek') and hasattr(file_obj, 'tell'):
            file_obj.seek(0, 2)  # Seek to end
            file_size = file_obj.tell()
            file_obj.seek(0)  # Reset to beginning
        else:
            file_size = 0
        
        report("grouping", f"Processing {filename}...")
        
        # Load LAS file
        try:
            if isinstance(file_obj, str):
                las = lasio.read(file_obj)
            elif isinstance(file_obj, bytes):
                str_data = file_obj.decode("utf-8", errors="ignore")
                las = lasio.read(io.StringIO(str_data))
            else:
                # File-like object (e.g., Streamlit UploadedFile)
                file_obj.seek(0)
                bytes_data = file_obj.read()
                str_data = bytes_data.decode("utf-8", errors="ignore")
                las = lasio.read(io.StringIO(str_data))
                # Reset for potential reprocessing
                file_obj.seek(0)
        except Exception as e:
            report("error", f"Failed to load {filename}: {str(e)}")
            continue
        
        # Extract well name (sanitized)
        try:
            if 'WELL' in las.well:
                well_name = str(las.well.WELL.value).strip().upper()
            else:
                well_name = 'UNKNOWN'
        except (AttributeError, KeyError):
            well_name = 'UNKNOWN'
        
        # Handle empty well names
        if not well_name or well_name.isspace():
            well_name = 'UNKNOWN'
        
        # Extract location
        try:
            if 'LOC' in las.well:
                location = str(las.well.LOC.value).strip()
            elif 'LOCATION' in las.well:
                location = str(las.well.LOCATION.value).strip()
            else:
                location = ''
        except (AttributeError, KeyError):
            location = ''
        
        # Get depth values for fingerprint
        try:
            strt = float(las.well.STRT.value) if 'STRT' in las.well else 0
            stop = float(las.well.STOP.value) if 'STOP' in las.well else 0
        except (AttributeError, ValueError, TypeError):
            strt = 0
            stop = 0
        
        # Create fingerprint for duplicate detection
        fingerprint_str = f"{filename}:{file_size}:{strt:.2f}:{stop:.2f}"
        fingerprint = hashlib.md5(fingerprint_str.encode()).hexdigest()
        
        # Check for duplicates within this well
        if well_name not in seen_fingerprints:
            seen_fingerprints[well_name] = set()
        
        if fingerprint in seen_fingerprints[well_name]:
            warning = f"Duplicate detected: {filename} (Well: {well_name}) - skipped"
            duplicate_warnings.append(warning)
            report("warning", warning)
            continue
        
        seen_fingerprints[well_name].add(fingerprint)
        
        # Detect original units
        original_unit = detect_las_units(las)
        
        # Convert to DataFrame
        df = las.df().reset_index()
        
        # Standardize depth column name
        depth_col = df.columns[0]
        if depth_col.upper() in ['DEPT', 'DEPTH', 'MD', 'TVD']:
            df = df.rename(columns={depth_col: 'DEPTH'})
        else:
            df = df.rename(columns={depth_col: 'DEPTH'})
        
        # Convert to meters
        df_meters, start_m, stop_m, step_m = convert_las_to_meters(las, df, 'DEPTH')
        
        # Strip null padding
        df_stripped = strip_null_padding(df_meters, 'DEPTH')
        
        # Check if DataFrame is empty after stripping
        if df_stripped.empty or len(df_stripped) == 0:
            report("warning", f"{filename}: No valid data rows after stripping null padding. Skipping file.")
            continue
        
        # Recalculate bounds after stripping
        actual_start = df_stripped['DEPTH'].min()
        actual_stop = df_stripped['DEPTH'].max()
        
        # Handle null values in data columns
        for col in df_stripped.columns:
            if col != 'DEPTH':
                for null_val in NULL_VALUES:
                    df_stripped[col] = df_stripped[col].replace(null_val, np.nan)
        
        # Get available curves
        curves = [c for c in df_stripped.columns if c.upper() != 'DEPTH']
        
        # Create PreprocessedLAS object
        preprocessed_file = PreprocessedLAS(
            filename=filename,
            original_unit=original_unit,
            df=df_stripped,
            start_depth=actual_start,
            stop_depth=actual_stop,
            step=step_m,
            curves=curves,
            well_name=well_name,
            location=location
        )
        
        # Add to appropriate well group
        if well_name not in well_groups:
            well_groups[well_name] = []
        well_groups[well_name].append(preprocessed_file)
        
        report("grouping", f"{filename} → Well: {well_name} ({actual_start:.1f}m - {actual_stop:.1f}m)")
    
    # Sort each well's files by start depth
    for well_name in well_groups:
        well_groups[well_name].sort(key=lambda x: x.start_depth)
    
    num_files_total = sum(len(files) for files in well_groups.values())
    
    report("grouping", 
           f"Grouped {num_files_total} files into {len(well_groups)} well(s). "
           f"Duplicates skipped: {len(duplicate_warnings)}")
    
    return WellGroupResult(
        well_groups=well_groups,
        duplicate_warnings=duplicate_warnings,
        num_wells=len(well_groups),
        num_files_total=num_files_total
    )


def _merge_dataframes_with_gap(
    composite_df: pd.DataFrame,
    next_df: pd.DataFrame,
    composite_end: float,
    next_start: float,
    step: float
) -> pd.DataFrame:
    """
    Merge two DataFrames with a gap between them.
    
    Creates NaN-filled rows to bridge the gap.
    
    Args:
        composite_df: Current composite DataFrame
        next_df: Next DataFrame to append
        composite_end: End depth of composite
        next_start: Start depth of next file
        step: Depth step for filling gap
        
    Returns:
        Merged DataFrame
    """
    # Create gap fill with NaN values
    gap_depths = np.arange(composite_end + step, next_start, step)
    
    if len(gap_depths) > 0:
        # Create gap DataFrame with NaN for all columns except DEPTH
        gap_data = {'DEPTH': gap_depths}
        for col in composite_df.columns:
            if col != 'DEPTH':
                gap_data[col] = np.nan
        gap_df = pd.DataFrame(gap_data)
        
        # Concatenate: composite + gap + next
        merged = pd.concat([composite_df, gap_df, next_df], ignore_index=True)
    else:
        # Just concatenate directly
        merged = pd.concat([composite_df, next_df], ignore_index=True)
    
    # Sort by depth and remove duplicates
    merged = merged.sort_values('DEPTH').drop_duplicates(subset=['DEPTH']).reset_index(drop=True)
    
    return merged


def _align_dataframe_columns(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensure both DataFrames have the same columns.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        
    Returns:
        Tuple of aligned DataFrames
    """
    all_cols = set(df1.columns) | set(df2.columns)
    
    for col in all_cols:
        if col not in df1.columns:
            df1[col] = np.nan
        if col not in df2.columns:
            df2[col] = np.nan
    
    # Reorder columns consistently
    col_order = ['DEPTH'] + sorted([c for c in all_cols if c != 'DEPTH'])
    df1 = df1[col_order]
    df2 = df2[col_order]
    
    return df1, df2


def batch_splice_pipeline(
    preprocessed_files: List[PreprocessedLAS],
    correlation_curve: str,
    grid_step: float = DEFAULT_GRID_STEP,
    max_search_meters: float = DEFAULT_SEARCH_WINDOW,
    max_elastic_meters: float = DEFAULT_DTW_WINDOW,
    progress_callback: Optional[Callable[[str, str], None]] = None
) -> BatchSpliceResult:
    """
    Chain-splice multiple preprocessed LAS files into a single composite log.
    
    This function implements the "chain splice" algorithm:
    1. Start with the shallowest file as the composite
    2. For each subsequent file:
       - If gap > 5m: Append with NaN fill
       - If overlap: Use correlation + DTW splicing
    3. Return final composite
    
    Args:
        preprocessed_files: List of PreprocessedLAS from preprocess_las_files()
        correlation_curve: Curve name to use for correlation (e.g., 'GR')
        grid_step: Resampling grid step in meters
        max_search_meters: Max search window for cross-correlation
        max_elastic_meters: Max elastic stretch for DTW
        progress_callback: Optional callback(step, message) for progress updates
        
    Returns:
        BatchSpliceResult with composite DataFrame and metadata
    """
    def report(step, msg):
        if progress_callback:
            progress_callback(step, msg)
    
    # Preserve original correlation curve for return value
    # Use a local variable for potential fallback assignments
    active_correlation_curve = correlation_curve
    
    if len(preprocessed_files) == 0:
        raise ValueError("No files to splice")
    
    if len(preprocessed_files) == 1:
        # Single file - just return it
        single = preprocessed_files[0]
        return BatchSpliceResult(
            composite_df=single.df,
            splice_log=["Single file - no splicing required"],
            file_summary=[{
                'filename': single.filename,
                'original_unit': single.original_unit,
                'start_m': single.start_depth,
                'stop_m': single.stop_depth,
                'action': 'Base file'
            }],
            total_depth_range=(single.start_depth, single.stop_depth),
            num_files_processed=1,
            correlation_curve=correlation_curve
        )
    
    splice_log = []
    file_summary = []
    
    # Initialize composite with first (shallowest) file
    first_file = preprocessed_files[0]
    composite_df = first_file.df.copy()
    composite_end = first_file.stop_depth
    composite_step = first_file.step
    
    file_summary.append({
        'filename': first_file.filename,
        'original_unit': first_file.original_unit,
        'start_m': first_file.start_depth,
        'stop_m': first_file.stop_depth,
        'action': 'Base file (shallowest)'
    })
    
    report("splicing", f"Initialized composite with {first_file.filename} "
           f"({first_file.start_depth:.1f}m - {first_file.stop_depth:.1f}m)")
    splice_log.append(f"Run 1: {first_file.filename} initialized as base "
                     f"({first_file.start_depth:.1f}m - {first_file.stop_depth:.1f}m)")
    
    # Process remaining files
    for i, next_file in enumerate(preprocessed_files[1:], start=2):
        report("splicing", f"Processing Run {i}: {next_file.filename}...")
        
        next_start = next_file.start_depth
        next_end = next_file.stop_depth
        
        # Align columns between composite and next file
        composite_df, next_df = _align_dataframe_columns(composite_df, next_file.df.copy())
        
        # Calculate gap/overlap
        gap = next_start - composite_end
        
        if gap > GAP_THRESHOLD:
            # GAP CASE: No overlap, just append with NaN fill
            report("splicing", f"Run {i-1} & Run {i}: Gap detected ({gap:.1f}m). Appending...")
            
            composite_df = _merge_dataframes_with_gap(
                composite_df, next_df,
                composite_end, next_start,
                composite_step
            )
            
            action = f"Appended (gap: {gap:.1f}m)"
            splice_log.append(f"Run {i-1} & Run {i}: Gap detected ({gap:.1f}m). Data appended.")
            
        else:
            # OVERLAP CASE: Use correlation + DTW splicing
            overlap_amount = composite_end - next_start
            
            if overlap_amount <= 0:
                # Edge case: files just touch, treat as small gap
                report("splicing", f"Run {i-1} & Run {i}: Files touch at boundary. Appending...")
                composite_df = pd.concat([composite_df, next_df], ignore_index=True)
                composite_df = composite_df.sort_values('DEPTH').drop_duplicates(
                    subset=['DEPTH']).reset_index(drop=True)
                action = "Appended (no overlap)"
                splice_log.append(f"Run {i-1} & Run {i}: No overlap. Data appended.")
            else:
                # Real overlap - use splicing algorithm
                report("splicing", f"Run {i-1} & Run {i}: Found {overlap_amount:.1f}m overlap. "
                       "Running correlation + DTW...")
                
                # Check if correlation curve exists in both
                # Use local variable for this iteration's curve selection
                iter_correlation_curve = active_correlation_curve
                
                if iter_correlation_curve not in composite_df.columns:
                    report("warning", f"Correlation curve {iter_correlation_curve} not in composite. "
                           "Using first available curve.")
                    available = [c for c in composite_df.columns if c != 'DEPTH']
                    if available:
                        iter_correlation_curve = available[0]
                    else:
                        # No curves to correlate, just append
                        composite_df = pd.concat([composite_df, next_df], ignore_index=True)
                        composite_df = composite_df.sort_values('DEPTH').drop_duplicates(
                            subset=['DEPTH']).reset_index(drop=True)
                        action = "Appended (no curves for correlation)"
                        splice_log.append(f"Run {i-1} & Run {i}: Appended (no correlation curve)")
                        file_summary.append({
                            'filename': next_file.filename,
                            'original_unit': next_file.original_unit,
                            'start_m': next_start,
                            'stop_m': next_end,
                            'action': action
                        })
                        composite_end = max(composite_end, next_end)
                        continue
                
                if iter_correlation_curve not in next_df.columns:
                    # No correlation curve in next file
                    report("warning", f"Correlation curve {iter_correlation_curve} not in {next_file.filename}. "
                           "Appending without alignment.")
                    composite_df = pd.concat([composite_df, next_df], ignore_index=True)
                    composite_df = composite_df.sort_values('DEPTH').drop_duplicates(
                        subset=['DEPTH']).reset_index(drop=True)
                    action = f"Appended (no {iter_correlation_curve} curve)"
                    splice_log.append(f"Run {i-1} & Run {i}: Appended without alignment")
                    file_summary.append({
                        'filename': next_file.filename,
                        'original_unit': next_file.original_unit,
                        'start_m': next_start,
                        'stop_m': next_end,
                        'action': action
                    })
                    composite_end = max(composite_end, next_end)
                    continue
                
                try:
                    # Extract overlap region data for splicing
                    shallow_depth = composite_df['DEPTH'].values
                    shallow_signal = composite_df[iter_correlation_curve].values
                    deep_depth = next_df['DEPTH'].values
                    deep_signal = next_df[iter_correlation_curve].values
                    
                    # Run the splice algorithm
                    result = splice_logs(
                        shallow_depth=shallow_depth,
                        shallow_signal=shallow_signal,
                        deep_depth=deep_depth,
                        deep_signal=deep_signal,
                        grid_step=grid_step,
                        max_search_meters=max_search_meters,
                        max_elastic_meters=max_elastic_meters,
                        progress_callback=None  # Don't forward inner progress
                    )
                    
                    shift_str = f"{abs(result.bulk_shift_meters):.2f}m"
                    shift_dir = "shallower" if result.bulk_shift_meters > 0 else "deeper"
                    
                    report("splicing", f"Run {i-1} & Run {i}: Correlation shift: {shift_str} ({shift_dir})")
                    
                    # Apply the bulk shift to the next file's depth
                    next_df_shifted = next_df.copy()
                    next_df_shifted['DEPTH'] = next_df_shifted['DEPTH'] - result.bulk_shift_meters
                    
                    # Splice at the midpoint
                    splice_point = result.splice_point
                    
                    # Take composite up to splice point
                    composite_upper = composite_df[composite_df['DEPTH'] < splice_point].copy()
                    
                    # Take shifted next file from splice point onwards
                    next_lower = next_df_shifted[next_df_shifted['DEPTH'] >= splice_point].copy()
                    
                    # Merge
                    composite_df = pd.concat([composite_upper, next_lower], ignore_index=True)
                    composite_df = composite_df.sort_values('DEPTH').reset_index(drop=True)
                    
                    action = f"Spliced (shift: {shift_str} {shift_dir}, overlap: {overlap_amount:.1f}m)"
                    splice_log.append(f"Run {i-1} & Run {i}: {overlap_amount:.1f}m overlap. "
                                     f"Shift: {shift_str} ({shift_dir}). "
                                     f"Splice point: {splice_point:.1f}m")
                    
                except Exception as e:
                    # Splicing failed, fall back to simple append
                    report("warning", f"Splicing failed for Run {i}: {str(e)}. Appending instead.")
                    composite_df = pd.concat([composite_df, next_df], ignore_index=True)
                    composite_df = composite_df.sort_values('DEPTH').drop_duplicates(
                        subset=['DEPTH']).reset_index(drop=True)
                    action = f"Appended (splice failed: {str(e)[:30]})"
                    splice_log.append(f"Run {i-1} & Run {i}: Splice failed. Appended instead.")
        
        file_summary.append({
            'filename': next_file.filename,
            'original_unit': next_file.original_unit,
            'start_m': next_start,
            'stop_m': next_end,
            'action': action
        })
        
        # Update composite end depth
        composite_end = composite_df['DEPTH'].max()
    
    # Final cleanup
    composite_df = composite_df.sort_values('DEPTH').reset_index(drop=True)
    
    total_start = composite_df['DEPTH'].min()
    total_end = composite_df['DEPTH'].max()
    
    report("complete", f"Batch splicing complete! "
           f"Composite range: {total_start:.1f}m - {total_end:.1f}m "
           f"({len(composite_df)} samples)")
    
    splice_log.append(f"Final composite: {total_start:.1f}m - {total_end:.1f}m ({len(composite_df)} samples)")
    
    return BatchSpliceResult(
        composite_df=composite_df,
        splice_log=splice_log,
        file_summary=file_summary,
        total_depth_range=(total_start, total_end),
        num_files_processed=len(preprocessed_files),
        correlation_curve=correlation_curve
    )

