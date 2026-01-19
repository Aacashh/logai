import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.ticker import LogLocator, LogFormatter, AutoMinorLocator
import matplotlib.gridspec as gridspec

# Industry-standard colors (Schlumberger/Techlog style)
COLORS = {
    'GR': '#00AA00',           # Green for Gamma Ray
    'GR_FILL': '#90EE90',      # Light green for sand shading
    'CALIPER': '#000000',      # Black for Caliper curve
    'RES_DEEP': '#0066CC',     # Blue for Deep Resistivity
    'RES_MED': '#CC0000',      # Red for Medium Resistivity  
    'RES_SHAL': '#FF8C00',     # Orange for Shallow Resistivity
    'DENS': '#CC0000',         # Red for Density
    'NEUT': '#0066CC',         # Blue for Neutron
    'CROSS_GAS': '#FFFF00',    # Yellow for gas crossover
    'CROSS_SHALE': '#808080',  # Grey for shale crossover
    'GRID': '#CCCCCC',         # Light grey grid
    'TRACK_BG': '#FFFFFF',     # White track background
    'HEADER_BG': '#F2F2F2',    # Light grey header background
    'BORDER': '#333333',       # Dark border
}

# Standard curve ranges per industry guidelines
STANDARD_RANGES = {
    'GR': (0, 150),           # 0-150 API
    'CALIPER': (6, 16),       # 6-16 inches (DCAL caliper scale)
    'RES': (0.2, 2000),       # 0.2-2000 ohm-m (log scale)
    'RHOB': (1.95, 2.95),     # 1.95-2.95 g/cc
    'NPHI': (-0.15, 0.45),    # -0.15-0.45 v/v (reversed for overlay)
}


def create_professional_log_display(
    df,
    mapping,
    header_info=None,
    settings=None,
    show_gr_fill=True,
    show_dn_crossover=True,
    highlight_mask=None,
    highlight_color='red',
    highlight_alpha=0.2
):
    """
    Creates a professional multi-track well log display following industry standards.
    
    Layout follows Schlumberger Techlog / IP (Interactive Petrophysics) style:
    - Track 1: Gamma Ray (GR) with Caliper curve overlay (DCAL, 6-16 inch scale)
    - Track 2: Resistivity curves (Deep, Medium, Shallow) - logarithmic
    - Track 3: Density-Neutron overlay with crossover fill
    
    Note: CNPOR (neutron in %) is automatically converted to v/v by dividing by 100.
    
    Args:
        df: DataFrame with log data (must include 'DEPTH' column)
        mapping: Curve mapping dictionary (e.g., {'GR': 'GR', 'RES_DEEP': 'RT'})
        header_info: Dictionary with well header info (WELL, FIELD, LOC, STRT, STOP, STEP)
        settings: Plot settings (scale_ratio, depth_unit, etc.)
        show_gr_fill: Enable GR sand indication shading (right-fill)
        show_dn_crossover: Enable Density-Neutron crossover fill
        highlight_mask: Optional boolean mask for highlighting regions
        highlight_color: Color for highlighted regions
        highlight_alpha: Alpha for highlighted regions
        
    Returns:
        Matplotlib figure
    """
    if settings is None:
        settings = {}
    
    # Get depth data
    depth = df['DEPTH'].values
    depth_min = np.nanmin(depth)
    depth_max = np.nanmax(depth)
    depth_range = depth_max - depth_min
    
    # Calculate figure dimensions
    scale_ratio = settings.get('scale_ratio', 500)
    depth_unit = settings.get('depth_unit', 'm')
    
    # Height based on depth range and scale (industry standard sizing)
    height_cm = (depth_range * 100) / scale_ratio
    height_in = max(8, min(30, height_cm / 2.54))  # Clamp between 8-30 inches
    
    # Track widths: ~3.5 inches each (≈350px at 100dpi)
    # 3 tracks + depth track + spacing = ~14 inches total
    track_width = 3.5
    depth_track_width = 0.8
    header_height_ratio = 0.08 if header_info else 0.02
    
    fig_width = depth_track_width + (track_width * 3) + 1.5
    
    # Create figure with GridSpec for header + tracks
    fig = plt.figure(figsize=(fig_width, height_in), facecolor='white')
    
    # GridSpec: header row + main tracks row
    if header_info:
        gs = gridspec.GridSpec(2, 4, figure=fig, 
                               height_ratios=[header_height_ratio, 1-header_height_ratio],
                               width_ratios=[depth_track_width, track_width, track_width, track_width],
                               wspace=0.02, hspace=0.02)
        
        # Header spanning all columns
        ax_header = fig.add_subplot(gs[0, :])
        _draw_well_header(ax_header, header_info, depth_unit)
        
        # Track axes
        ax_depth = fig.add_subplot(gs[1, 0])
        ax_gr = fig.add_subplot(gs[1, 1], sharey=ax_depth)
        ax_res = fig.add_subplot(gs[1, 2], sharey=ax_depth)
        ax_dn = fig.add_subplot(gs[1, 3], sharey=ax_depth)
    else:
        gs = gridspec.GridSpec(1, 4, figure=fig,
                               width_ratios=[depth_track_width, track_width, track_width, track_width],
                               wspace=0.02)
        
        ax_depth = fig.add_subplot(gs[0, 0])
        ax_gr = fig.add_subplot(gs[0, 1], sharey=ax_depth)
        ax_res = fig.add_subplot(gs[0, 2], sharey=ax_depth)
        ax_dn = fig.add_subplot(gs[0, 3], sharey=ax_depth)
    
    axes = [ax_depth, ax_gr, ax_res, ax_dn]
    
    # =========================================================================
    # DEPTH TRACK (leftmost)
    # =========================================================================
    _draw_depth_track(ax_depth, depth, depth_min, depth_max, depth_unit)
    
    # =========================================================================
    # TRACK 1: GAMMA RAY
    # =========================================================================
    _draw_gr_track(ax_gr, df, mapping, depth, settings, show_gr_fill, highlight_mask, highlight_color, highlight_alpha)
    
    # =========================================================================
    # TRACK 2: RESISTIVITY (Logarithmic)
    # =========================================================================
    _draw_resistivity_track(ax_res, df, mapping, depth, settings, highlight_mask, highlight_color, highlight_alpha)
    
    # =========================================================================
    # TRACK 3: DENSITY-NEUTRON OVERLAY
    # =========================================================================
    _draw_density_neutron_track(ax_dn, df, mapping, depth, settings, show_dn_crossover, highlight_mask, highlight_color, highlight_alpha)
    
    # Set depth limits (inverted - depth increases downward)
    ax_depth.set_ylim(depth_max, depth_min)
    
    # Hide y-axis labels on non-depth tracks
    for ax in [ax_gr, ax_res, ax_dn]:
        plt.setp(ax.get_yticklabels(), visible=False)
    
    plt.tight_layout()
    
    return fig


def _draw_well_header(ax, header_info, depth_unit='m'):
    """Draw well header panel at top of log display."""
    ax.set_facecolor(COLORS['HEADER_BG'])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Add border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS['BORDER'])
        spine.set_linewidth(1)
    
    # Well name (large, bold, centered)
    well_name = header_info.get('WELL', 'Unknown Well')
    ax.text(0.5, 0.75, well_name, fontsize=14, fontweight='bold',
            ha='center', va='center', transform=ax.transAxes)
    
    # Build metadata line
    metadata_parts = []
    
    if header_info.get('FIELD'):
        metadata_parts.append(f"FIELD: {header_info['FIELD']}")
    
    if header_info.get('LOC'):
        metadata_parts.append(f"LOC: {header_info['LOC']}")
    
    # Depth range
    strt = header_info.get('STRT', '')
    stop = header_info.get('STOP', '')
    if strt and stop:
        metadata_parts.append(f"DEPTH: {strt:.1f} - {stop:.1f} {depth_unit}")
    
    # Step
    step = header_info.get('STEP', '')
    if step:
        metadata_parts.append(f"STEP: {step:.4f} {depth_unit}")
    
    metadata_text = '  |  '.join(metadata_parts)
    ax.text(0.5, 0.35, metadata_text, fontsize=9, ha='center', va='center',
            transform=ax.transAxes, color='#555555')


def _draw_depth_track(ax, depth, depth_min, depth_max, depth_unit='m'):
    """Draw the depth track on the left side."""
    ax.set_facecolor(COLORS['TRACK_BG'])
    ax.set_xlim(0, 1)
    
    # Style
    ax.set_ylabel(f"Depth ({depth_unit})", fontsize=10, fontweight='bold')
    ax.yaxis.set_tick_params(labelsize=8)
    ax.xaxis.set_visible(False)
    
    # Grid lines
    ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color=COLORS['GRID'], alpha=0.7)
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.grid(True, which='minor', linestyle=':', linewidth=0.3, color=COLORS['GRID'], alpha=0.4)
    
    # Border
    for spine in ax.spines.values():
        spine.set_color(COLORS['BORDER'])
        spine.set_linewidth(1)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _draw_gr_track(ax, df, mapping, depth, settings, show_fill=True, 
                   highlight_mask=None, highlight_color='red', highlight_alpha=0.2):
    """Draw Gamma Ray track with Caliper curve overlay (no GR shading)."""
    ax.set_facecolor(COLORS['TRACK_BG'])
    
    # Track header - now includes both GR and Caliper
    ax.set_title("GAMMA RAY / CALIPER\n(API / inch)", fontsize=9, fontweight='bold', pad=8,
                 color=COLORS['GR'])
    
    # Get range from settings or use standard for GR (primary axis)
    gr_min = settings.get('gr_min', STANDARD_RANGES['GR'][0])
    gr_max = settings.get('gr_max', STANDARD_RANGES['GR'][1])
    ax.set_xlim(gr_min, gr_max)
    
    # Grid
    ax.xaxis.grid(True, linestyle='-', linewidth=0.5, color=COLORS['GRID'], alpha=0.7)
    ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color=COLORS['GRID'], alpha=0.7)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.grid(True, which='minor', linestyle=':', linewidth=0.3, color=COLORS['GRID'], alpha=0.4)
    
    # X-axis styling for GR (top axis - green)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xlabel("GR (API)", fontsize=8, color=COLORS['GR'], fontweight='bold')
    ax.tick_params(axis='x', colors=COLORS['GR'], labelsize=7)
    ax.spines['top'].set_color(COLORS['GR'])
    ax.spines['top'].set_linewidth(2)
    
    # Border
    for spine in ['bottom', 'left', 'right']:
        ax.spines[spine].set_color(COLORS['BORDER'])
        ax.spines[spine].set_linewidth(1)
    
    # Legend items
    legend_handles = []
    
    # Plot GR curve
    gr_col = mapping.get('GR')
    if gr_col and gr_col in df.columns:
        gr_data = df[gr_col].values.copy()
        
        # Plot curve (line thickness 1.5px as per guidelines)
        ax.plot(gr_data, depth, color=COLORS['GR'], linewidth=1.5, label='GR')
        legend_handles.append(mpatches.Patch(color=COLORS['GR'], label='GR'))
        
        # NOTE: GR shading removed per client request - Caliper curve shown instead
    else:
        ax.text(0.5, 0.5, "No GR Data", transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='gray')
    
    # Add Caliper curve on secondary axis (DCAL with 6-16 inch scale)
    cal_col = mapping.get('CALIPER')
    if cal_col and cal_col in df.columns:
        # Create secondary x-axis for Caliper
        ax_cal = ax.twiny()
        
        # Caliper range: 6-16 inches (DCAL)
        cal_min = settings.get('cal_min', STANDARD_RANGES['CALIPER'][0])
        cal_max = settings.get('cal_max', STANDARD_RANGES['CALIPER'][1])
        ax_cal.set_xlim(cal_min, cal_max)
        
        # Caliper axis styling (top, offset - black)
        ax_cal.set_xlabel("CALI (inch)", fontsize=8, color=COLORS['CALIPER'], fontweight='bold')
        ax_cal.tick_params(axis='x', colors=COLORS['CALIPER'], labelsize=7)
        ax_cal.spines['top'].set_position(('outward', 30))
        ax_cal.spines['top'].set_color(COLORS['CALIPER'])
        ax_cal.spines['top'].set_linewidth(2)
        
        # Get caliper data
        cal_data = df[cal_col].values.copy()
        
        # Plot caliper curve (black, dashed for distinction)
        ax_cal.plot(cal_data, depth, color=COLORS['CALIPER'], linewidth=1.5, 
                    linestyle='--', label='CALI')
        legend_handles.append(mpatches.Patch(color=COLORS['CALIPER'], label='CALI'))
    
    # Add legend
    if legend_handles:
        ax.legend(handles=legend_handles, loc='lower right', fontsize=6,
                 framealpha=0.9, edgecolor=COLORS['BORDER'])
    
    # Highlight regions if provided
    if highlight_mask is not None:
        _add_highlight_shading(ax, depth, highlight_mask, gr_min, gr_max, 
                               highlight_color, highlight_alpha)


def _draw_resistivity_track(ax, df, mapping, depth, settings,
                            highlight_mask=None, highlight_color='red', highlight_alpha=0.2):
    """Draw Resistivity track with logarithmic scale."""
    ax.set_facecolor(COLORS['TRACK_BG'])
    
    # Track header
    ax.set_title("RESISTIVITY\n(ohm.m)", fontsize=9, fontweight='bold', pad=8)
    
    # Logarithmic scale
    ax.set_xscale('log')
    
    # Get range from settings or use standard
    res_min = settings.get('res_min', STANDARD_RANGES['RES'][0])
    res_max = settings.get('res_max', STANDARD_RANGES['RES'][1])
    ax.set_xlim(res_min, res_max)
    
    # Grid (log scale)
    ax.xaxis.grid(True, which='major', linestyle='-', linewidth=0.5, color=COLORS['GRID'], alpha=0.7)
    ax.xaxis.grid(True, which='minor', linestyle=':', linewidth=0.3, color=COLORS['GRID'], alpha=0.4)
    ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color=COLORS['GRID'], alpha=0.7)
    
    # Log tick locator
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
    
    # X-axis styling
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', labelsize=7)
    
    # Border
    for spine in ax.spines.values():
        spine.set_color(COLORS['BORDER'])
        spine.set_linewidth(1)
    
    # Legend items
    legend_handles = []
    
    # Plot Deep Resistivity (Blue)
    res_deep_col = mapping.get('RES_DEEP')
    if res_deep_col and res_deep_col in df.columns:
        data = df[res_deep_col].values.copy()
        # Filter valid positive values for log scale
        data[data <= 0] = np.nan
        ax.plot(data, depth, color=COLORS['RES_DEEP'], linewidth=1.5, label='Deep')
        legend_handles.append(mpatches.Patch(color=COLORS['RES_DEEP'], label='Deep'))
    
    # Plot Medium Resistivity (Red)
    res_med_col = mapping.get('RES_MED')
    if res_med_col and res_med_col in df.columns:
        data = df[res_med_col].values.copy()
        data[data <= 0] = np.nan
        ax.plot(data, depth, color=COLORS['RES_MED'], linewidth=1.5, label='Med')
        legend_handles.append(mpatches.Patch(color=COLORS['RES_MED'], label='Med'))
    
    # Plot Shallow Resistivity (Orange)
    res_shal_col = mapping.get('RES_SHAL')
    if res_shal_col and res_shal_col in df.columns:
        data = df[res_shal_col].values.copy()
        data[data <= 0] = np.nan
        ax.plot(data, depth, color=COLORS['RES_SHAL'], linewidth=1.5, label='Shallow')
        legend_handles.append(mpatches.Patch(color=COLORS['RES_SHAL'], label='Shallow'))
    
    if legend_handles:
        ax.legend(handles=legend_handles, loc='lower right', fontsize=6,
                 framealpha=0.9, edgecolor=COLORS['BORDER'])
    else:
        ax.text(0.5, 0.5, "No Resistivity Data", transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='gray')
    
    # Highlight regions
    if highlight_mask is not None:
        _add_highlight_shading(ax, depth, highlight_mask, res_min, res_max,
                               highlight_color, highlight_alpha)


def _draw_density_neutron_track(ax, df, mapping, depth, settings, show_crossover=True,
                                highlight_mask=None, highlight_color='red', highlight_alpha=0.2):
    """Draw Density-Neutron overlay track with crossover fill."""
    ax.set_facecolor(COLORS['TRACK_BG'])
    
    # Track header
    ax.set_title("DENSITY-NEUTRON", fontsize=9, fontweight='bold', pad=8)
    
    # Primary axis: Density (RHOB) - Red
    dens_min = settings.get('dens_min', STANDARD_RANGES['RHOB'][0])
    dens_max = settings.get('dens_max', STANDARD_RANGES['RHOB'][1])
    ax.set_xlim(dens_min, dens_max)
    
    # Grid
    ax.xaxis.grid(True, linestyle='-', linewidth=0.5, color=COLORS['GRID'], alpha=0.7)
    ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color=COLORS['GRID'], alpha=0.7)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.grid(True, which='minor', linestyle=':', linewidth=0.3, color=COLORS['GRID'], alpha=0.4)
    
    # Primary x-axis styling (Density - top)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xlabel("RHOB (g/cc)", fontsize=8, color=COLORS['DENS'], fontweight='bold')
    ax.tick_params(axis='x', colors=COLORS['DENS'], labelsize=7)
    ax.spines['top'].set_color(COLORS['DENS'])
    ax.spines['top'].set_linewidth(2)
    
    # Border
    for spine in ['bottom', 'left', 'right']:
        ax.spines[spine].set_color(COLORS['BORDER'])
        ax.spines[spine].set_linewidth(1)
    
    # Secondary axis: Neutron (NPHI) - Blue (reversed scale)
    ax_neut = ax.twiny()
    neut_min = settings.get('neut_min', STANDARD_RANGES['NPHI'][0])
    neut_max = settings.get('neut_max', STANDARD_RANGES['NPHI'][1])
    ax_neut.set_xlim(neut_max, neut_min)  # REVERSED for overlay
    
    ax_neut.set_xlabel("NPHI (v/v)", fontsize=8, color=COLORS['NEUT'], fontweight='bold')
    ax_neut.tick_params(axis='x', colors=COLORS['NEUT'], labelsize=7)
    ax_neut.spines['top'].set_position(('outward', 30))
    ax_neut.spines['top'].set_color(COLORS['NEUT'])
    ax_neut.spines['top'].set_linewidth(2)
    
    # Get data
    dens_col = mapping.get('DENS')
    neut_col = mapping.get('NEUT')
    
    dens_data = None
    neut_data = None
    
    # Plot Density (Red, solid)
    if dens_col and dens_col in df.columns:
        dens_data = df[dens_col].values.copy()
        ax.plot(dens_data, depth, color=COLORS['DENS'], linewidth=1.5, label='RHOB')
    
    # Plot Neutron (Blue, dashed for distinction)
    if neut_col and neut_col in df.columns:
        neut_data = df[neut_col].values.copy()
        
        # Convert CNPOR from % (porosity units PU) to v/v by dividing by 100
        # Detect if data is in % format (values typically > 1, often 0-45%)
        # If max valid value > 1, assume it's in % and convert
        valid_neut = neut_data[~np.isnan(neut_data)]
        if len(valid_neut) > 0 and np.nanmax(valid_neut) > 1:
            neut_data = neut_data / 100.0  # Convert % to v/v
        
        ax_neut.plot(neut_data, depth, color=COLORS['NEUT'], linewidth=1.5, 
                     linestyle='--', label='NPHI')
        
        # Crossover fill between Density and Neutron
        if show_crossover and dens_data is not None:
            # Transform neutron to density axis for overlay comparison
            neut_transformed = _transform_nphi_to_rhob(neut_data, neut_min, neut_max, 
                                                        dens_min, dens_max)
            
            valid = ~np.isnan(dens_data) & ~np.isnan(neut_transformed)
            
            # Gas effect: NPHI crosses left of RHOB (neutron < density on transformed scale)
            ax.fill_betweenx(depth, dens_data, neut_transformed,
                            where=valid & (neut_transformed < dens_data),
                            color=COLORS['CROSS_GAS'], alpha=0.4,
                            label='Gas Effect')
            
            # Shale: RHOB crosses left of NPHI (density < neutron on transformed scale)
            ax.fill_betweenx(depth, dens_data, neut_transformed,
                            where=valid & (dens_data < neut_transformed),
                            color=COLORS['CROSS_SHALE'], alpha=0.3,
                            label='Shale')
    
    # Legend
    legend_handles = []
    if dens_data is not None:
        legend_handles.append(mpatches.Patch(color=COLORS['DENS'], label='RHOB'))
    if neut_data is not None:
        legend_handles.append(mpatches.Patch(color=COLORS['NEUT'], label='NPHI'))
    
    if legend_handles:
        ax.legend(handles=legend_handles, loc='lower right', fontsize=6,
                 framealpha=0.9, edgecolor=COLORS['BORDER'])
    elif dens_data is None and neut_data is None:
        ax.text(0.5, 0.5, "No Density/Neutron Data", transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='gray')
    
    # Highlight regions
    if highlight_mask is not None:
        _add_highlight_shading(ax, depth, highlight_mask, dens_min, dens_max,
                               highlight_color, highlight_alpha)


def _transform_nphi_to_rhob(nphi_data, nphi_min, nphi_max, rhob_min, rhob_max):
    """Transform NPHI values to RHOB axis scale for overlay comparison."""
    # Normalize to 0-1, then reverse (because NPHI scale is inverted)
    normalized = (nphi_data - nphi_min) / (nphi_max - nphi_min)
    reversed_norm = 1 - normalized
    # Scale to RHOB axis
    return rhob_min + reversed_norm * (rhob_max - rhob_min)


def _add_highlight_shading(ax, depth, mask, x_min, x_max, color, alpha):
    """Add vertical shading to highlight specific depth regions."""
    if mask is None or len(mask) != len(depth):
        return
    
    # Find contiguous highlighted regions
    changes = np.diff(mask.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    
    # Handle edge cases
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [len(mask)]])
    
    # Draw shaded rectangles
    for start, end in zip(starts, ends):
        if start < len(depth) and end <= len(depth):
            d_start = depth[start]
            d_end = depth[min(end, len(depth)-1)]
            ax.axhspan(d_start, d_end, alpha=alpha, color=color, zorder=0)


def create_before_after_comparison(
    df_before, 
    df_after, 
    curve_name,
    mapping=None,
    header_info=None,
    settings=None,
    highlight_mask=None,
    title_before="BEFORE",
    title_after="AFTER"
):
    """
    Create a side-by-side before/after comparison of a single curve.
    
    Professional vertical log style with depth on left, curves displayed
    in industry-standard format.
    
    Args:
        df_before: DataFrame with original data
        df_after: DataFrame with processed data
        curve_name: Name of the curve to compare
        mapping: Curve mapping dictionary
        header_info: Well header information
        settings: Plot settings
        highlight_mask: Boolean mask for outliers/noise (applied to before plot)
        title_before: Title for before panel
        title_after: Title for after panel
        
    Returns:
        Matplotlib figure
    """
    if settings is None:
        settings = {}
    
    # Get depth data
    depth_before = df_before['DEPTH'].values
    depth_after = df_after['DEPTH'].values if 'DEPTH' in df_after.columns else depth_before
    
    depth_min = min(np.nanmin(depth_before), np.nanmin(depth_after))
    depth_max = max(np.nanmax(depth_before), np.nanmax(depth_after))
    depth_range = depth_max - depth_min
    
    # Calculate figure dimensions
    scale_ratio = settings.get('scale_ratio', 500)
    height_cm = (depth_range * 100) / scale_ratio
    height_in = max(6, min(18, height_cm / 2.54))
    
    # Create figure
    fig = plt.figure(figsize=(10, height_in), facecolor='white')
    
    # GridSpec: header + 2 tracks
    if header_info:
        gs = gridspec.GridSpec(2, 3, figure=fig, 
                               height_ratios=[0.08, 0.92],
                               width_ratios=[0.6, 3.5, 3.5],
                               wspace=0.05, hspace=0.02)
        
        ax_header = fig.add_subplot(gs[0, :])
        _draw_well_header(ax_header, header_info, settings.get('depth_unit', 'm'))
        
        ax_depth = fig.add_subplot(gs[1, 0])
        ax_before = fig.add_subplot(gs[1, 1], sharey=ax_depth)
        ax_after = fig.add_subplot(gs[1, 2], sharey=ax_depth)
    else:
        gs = gridspec.GridSpec(1, 3, figure=fig,
                               width_ratios=[0.6, 3.5, 3.5],
                               wspace=0.05)
        
        ax_depth = fig.add_subplot(gs[0, 0])
        ax_before = fig.add_subplot(gs[0, 1], sharey=ax_depth)
        ax_after = fig.add_subplot(gs[0, 2], sharey=ax_depth)
    
    # Draw depth track
    _draw_depth_track(ax_depth, depth_before, depth_min, depth_max, 
                      settings.get('depth_unit', 'm'))
    
    # Get curve color based on type
    curve_color = _get_curve_color(curve_name, mapping)
    
    # Get data range for consistent scaling
    data_before = df_before[curve_name].values if curve_name in df_before.columns else None
    data_after = df_after[curve_name].values if curve_name in df_after.columns else None
    
    if data_before is not None and data_after is not None:
        all_data = np.concatenate([data_before[~np.isnan(data_before)], 
                                   data_after[~np.isnan(data_after)]])
        if len(all_data) > 0:
            data_min = np.percentile(all_data, 1)
            data_max = np.percentile(all_data, 99)
            # Add 5% padding
            padding = (data_max - data_min) * 0.05
            data_min -= padding
            data_max += padding
        else:
            data_min, data_max = 0, 100
    else:
        data_min, data_max = 0, 100
    
    # Draw BEFORE track
    _draw_comparison_track(ax_before, df_before, curve_name, depth_before,
                          data_min, data_max, curve_color, title_before,
                          highlight_mask, '#ef4444', 0.3)
    
    # Draw AFTER track
    _draw_comparison_track(ax_after, df_after, curve_name, depth_after,
                          data_min, data_max, '#22c55e', title_after,
                          None, None, None)
    
    # Set depth limits
    ax_depth.set_ylim(depth_max, depth_min)
    
    # Hide y-axis on non-depth tracks
    plt.setp(ax_before.get_yticklabels(), visible=False)
    plt.setp(ax_after.get_yticklabels(), visible=False)
    
    plt.tight_layout()
    return fig


def _get_curve_color(curve_name, mapping):
    """Get industry-standard color for a curve type."""
    curve_upper = curve_name.upper()
    
    if 'GR' in curve_upper:
        return COLORS['GR']
    elif any(r in curve_upper for r in ['RT', 'RLLD', 'RDEP', 'ILD', 'RES']):
        return COLORS['RES_DEEP']
    elif any(r in curve_upper for r in ['RM', 'RLLM', 'ILM']):
        return COLORS['RES_MED']
    elif any(r in curve_upper for r in ['RS', 'RXOZ', 'MSFL']):
        return COLORS['RES_SHAL']
    elif any(r in curve_upper for r in ['RHOB', 'RHOZ', 'DEN']):
        return COLORS['DENS']
    elif any(r in curve_upper for r in ['NPHI', 'TNPH', 'NEU']):
        return COLORS['NEUT']
    else:
        return '#3b82f6'  # Default blue


def _draw_comparison_track(ax, df, curve_name, depth, data_min, data_max, 
                           curve_color, title, highlight_mask=None,
                           highlight_color=None, highlight_alpha=None):
    """Draw a single track for before/after comparison."""
    ax.set_facecolor(COLORS['TRACK_BG'])
    
    # Title with color indicator
    ax.set_title(f"{title}\n{curve_name}", fontsize=10, fontweight='bold', 
                pad=8, color=curve_color)
    
    # Set scale
    ax.set_xlim(data_min, data_max)
    
    # Grid
    ax.xaxis.grid(True, linestyle='-', linewidth=0.5, color=COLORS['GRID'], alpha=0.7)
    ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color=COLORS['GRID'], alpha=0.7)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.grid(True, which='minor', linestyle=':', linewidth=0.3, color=COLORS['GRID'], alpha=0.4)
    
    # X-axis on top
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', colors=curve_color, labelsize=7)
    ax.spines['top'].set_color(curve_color)
    ax.spines['top'].set_linewidth(2)
    
    # Border
    for spine in ['bottom', 'left', 'right']:
        ax.spines[spine].set_color(COLORS['BORDER'])
        ax.spines[spine].set_linewidth(1)
    
    # Plot curve
    if curve_name in df.columns:
        data = df[curve_name].values.copy()
        ax.plot(data, depth, color=curve_color, linewidth=1.5)
        
        # Highlight regions (outliers, noise, etc.)
        if highlight_mask is not None and highlight_color:
            highlight_depths = depth[highlight_mask]
            highlight_values = data[highlight_mask]
            ax.scatter(highlight_values, highlight_depths, color=highlight_color,
                      s=8, alpha=0.7, zorder=5, label='Flagged')
    else:
        ax.text(0.5, 0.5, f"No {curve_name} Data", transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='gray')


def create_multi_file_overlay(
    dataframes,
    curve_name,
    headers=None,
    settings=None,
    colors=None
):
    """
    Create an overlay plot showing the same curve from multiple LAS files.
    
    Useful for pre-splicing visualization to see depth alignment issues.
    
    Args:
        dataframes: List of DataFrames
        curve_name: Curve to plot
        headers: List of header dictionaries (optional)
        settings: Plot settings
        colors: List of colors (optional)
        
    Returns:
        Matplotlib figure
    """
    if settings is None:
        settings = {}
    
    if colors is None:
        colors = ['#22d3ee', '#34d399', '#f472b6', '#fbbf24', '#a78bfa', '#fb7185']
    
    # Get overall depth range
    all_depths = []
    for df in dataframes:
        if 'DEPTH' in df.columns:
            all_depths.extend(df['DEPTH'].dropna().values)
    
    if not all_depths:
        return None
    
    depth_min = min(all_depths)
    depth_max = max(all_depths)
    depth_range = depth_max - depth_min
    
    # Figure size
    scale_ratio = settings.get('scale_ratio', 500)
    height_cm = (depth_range * 100) / scale_ratio
    height_in = max(6, min(18, height_cm / 2.54))
    
    fig = plt.figure(figsize=(8, height_in), facecolor='white')
    
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[0.6, 4], wspace=0.05)
    
    ax_depth = fig.add_subplot(gs[0, 0])
    ax_curve = fig.add_subplot(gs[0, 1], sharey=ax_depth)
    
    # Depth track
    _draw_depth_track(ax_depth, np.array([depth_min, depth_max]), 
                      depth_min, depth_max, settings.get('depth_unit', 'm'))
    
    # Curve track
    ax_curve.set_facecolor(COLORS['TRACK_BG'])
    ax_curve.set_title(f"Overlay: {curve_name}", fontsize=10, fontweight='bold', pad=8)
    
    # Grid
    ax_curve.xaxis.grid(True, linestyle='-', linewidth=0.5, color=COLORS['GRID'], alpha=0.7)
    ax_curve.yaxis.grid(True, linestyle='-', linewidth=0.5, color=COLORS['GRID'], alpha=0.7)
    
    ax_curve.xaxis.set_label_position('top')
    ax_curve.xaxis.tick_top()
    
    for spine in ax_curve.spines.values():
        spine.set_color(COLORS['BORDER'])
        spine.set_linewidth(1)
    
    # Plot each file
    legend_handles = []
    for idx, df in enumerate(dataframes):
        if curve_name in df.columns:
            color = colors[idx % len(colors)]
            depth = df['DEPTH'].values
            data = df[curve_name].values
            
            label = f"File {idx+1}"
            if headers and idx < len(headers):
                well = headers[idx].get('WELL', f'File {idx+1}')
                label = well[:20]
            
            ax_curve.plot(data, depth, color=color, linewidth=1.2, label=label)
            legend_handles.append(mpatches.Patch(color=color, label=label))
    
    if legend_handles:
        ax_curve.legend(handles=legend_handles, loc='lower right', fontsize=7,
                       framealpha=0.9, edgecolor=COLORS['BORDER'])
    
    ax_depth.set_ylim(depth_max, depth_min)
    plt.setp(ax_curve.get_yticklabels(), visible=False)
    
    plt.tight_layout()
    return fig


def create_log_plot(df, mapping, settings, show_gr_fill=False, show_dn_fill=False):
    """
    Creates a professional Matplotlib figure matching Schlumberger Techlog style.
    
    Args:
        df: DataFrame with log data
        mapping: Curve mapping dictionary
        settings: Plot settings dictionary
        show_gr_fill: Enable GR sand indication shading
        show_dn_fill: Enable Density-Neutron crossover fill
    """
    depth_min = df['DEPTH'].min()
    depth_max = df['DEPTH'].max()
    depth_range = depth_max - depth_min
    
    # Calculate figure height based on scale
    scale_ratio = settings.get('scale_ratio', 500)
    height_cm = (depth_range * 100) / scale_ratio
    height_in = max(5, height_cm / 2.54)
    
    # Fixed track width: ~3.5 inches each (≈350px at 100dpi)
    # 3 tracks + spacing = ~12 inches total width
    figsize = (12, height_in)
    
    # Create figure with professional styling
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize, sharey=True)
    fig.patch.set_facecolor('white')
    
    # Adjust spacing between tracks
    plt.subplots_adjust(wspace=0.05, top=0.97, bottom=0.02, left=0.08, right=0.98)
    
    # Track references
    ax_gr = axes[0]
    ax_res = axes[1]
    ax_dn = axes[2]
    
    # Common styling for all tracks
    for ax in axes:
        ax.set_facecolor(COLORS['TRACK_BG'])
        ax.tick_params(axis='both', which='major', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(COLORS['BORDER'])
            spine.set_linewidth(1)
    
    depth = df['DEPTH'].values
    
    # =============================================
    # TRACK 1: GAMMA RAY
    # =============================================
    _plot_gamma_ray(ax_gr, df, mapping, settings, depth, show_gr_fill)
    
    # Set depth axis on first track
    ax_gr.set_ylim(depth_max, depth_min)  # Inverted
    ax_gr.set_ylabel("Depth (m)", fontsize=10, fontweight='bold')
    ax_gr.yaxis.set_tick_params(labelsize=8)
    
    # =============================================
    # TRACK 2: RESISTIVITY
    # =============================================
    _plot_resistivity(ax_res, df, mapping, settings, depth)
    
    # =============================================
    # TRACK 3: DENSITY-NEUTRON
    # =============================================
    _plot_density_neutron(ax_dn, df, mapping, settings, depth, show_dn_fill)
    
    # Apply tight layout for professional appearance
    fig.tight_layout()
    
    return fig


def _plot_gamma_ray(ax, df, mapping, settings, depth, show_fill=False):
    """Plot Gamma Ray track with optional sand shading."""
    
    # Track header
    ax.set_title("GAMMA RAY", fontsize=10, fontweight='bold', pad=10)
    
    # Grid styling
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, color=COLORS['GRID'], alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3, color=COLORS['GRID'], alpha=0.4)
    
    gr_min = settings.get('gr_min', 0)
    gr_max = settings.get('gr_max', 150)
    ax.set_xlim(gr_min, gr_max)
    
    # X-axis styling
    ax.set_xlabel("API", fontsize=9, color=COLORS['GR'], fontweight='bold')
    ax.tick_params(axis='x', colors=COLORS['GR'], labelsize=8)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.spines['top'].set_color(COLORS['GR'])
    ax.spines['top'].set_linewidth(2)
    
    if mapping['GR'] and mapping['GR'] in df.columns:
        gr_data = df[mapping['GR']].values
        
        # Plot GR curve
        ax.plot(gr_data, depth, color=COLORS['GR'], linewidth=1.5, label='GR')
        
        # Optional: Sand shading (fill from curve to right edge)
        if show_fill:
            # Create sand indication: shade right of GR curve
            # Low GR (left) = sand, High GR (right) = shale
            sand_threshold = (gr_min + gr_max) / 2  # Middle point
            ax.fill_betweenx(depth, gr_data, gr_min, 
                            where=~np.isnan(gr_data),
                            color=COLORS['GR_FILL'], alpha=0.3,
                            label='Sand indication')
    else:
        ax.text(0.5, 0.5, "No GR Curve", transform=ax.transAxes, 
                ha='center', va='center', fontsize=12, color='gray')


def _plot_resistivity(ax, df, mapping, settings, depth):
    """Plot Resistivity track with logarithmic scale."""
    
    # Track header
    ax.set_title("RESISTIVITY", fontsize=10, fontweight='bold', pad=10)
    
    # Logarithmic scale
    ax.set_xscale('log')
    
    res_min = settings.get('res_min', 0.2)
    res_max = settings.get('res_max', 2000)
    ax.set_xlim(res_min, res_max)
    
    # Professional log grid
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, color=COLORS['GRID'], alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3, color=COLORS['GRID'], alpha=0.4)
    
    # X-axis styling  
    ax.set_xlabel("ohm.m", fontsize=9, fontweight='bold')
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.tick_params(axis='x', labelsize=7)
    
    # Set log locator for better tick marks
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
    
    plotted = []
    
    # Deep Resistivity
    if mapping['RES_DEEP'] and mapping['RES_DEEP'] in df.columns:
        ax.plot(df[mapping['RES_DEEP']], depth, 
                color=COLORS['RES_DEEP'], linewidth=1.5, label='Deep (RT)')
        plotted.append(mpatches.Patch(color=COLORS['RES_DEEP'], label='Deep'))
    
    # Medium Resistivity
    if mapping['RES_MED'] and mapping['RES_MED'] in df.columns:
        ax.plot(df[mapping['RES_MED']], depth,
                color=COLORS['RES_MED'], linewidth=1.5, label='Med (RM)')
        plotted.append(mpatches.Patch(color=COLORS['RES_MED'], label='Med'))
    
    # Shallow Resistivity
    if mapping['RES_SHAL'] and mapping['RES_SHAL'] in df.columns:
        ax.plot(df[mapping['RES_SHAL']], depth,
                color=COLORS['RES_SHAL'], linewidth=1.5, label='Shallow (RS)')
        plotted.append(mpatches.Patch(color=COLORS['RES_SHAL'], label='Shallow'))
    
    if plotted:
        ax.legend(handles=plotted, loc='upper right', fontsize=7, 
                 framealpha=0.9, edgecolor=COLORS['BORDER'])
    else:
        ax.text(0.5, 0.5, "No Resistivity Curves", transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')


def _plot_density_neutron(ax, df, mapping, settings, depth, show_fill=False):
    """Plot Density-Neutron overlay with dual axes."""
    
    # Track header
    ax.set_title("DENSITY - NEUTRON", fontsize=10, fontweight='bold', pad=10)
    
    # Grid
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, color=COLORS['GRID'], alpha=0.7)
    
    # Primary axis: Density (Red)
    ax_dens = ax
    dens_min = settings.get('dens_min', 1.95)
    dens_max = settings.get('dens_max', 2.95)
    ax_dens.set_xlim(dens_min, dens_max)
    ax_dens.set_xlabel("RHOB (g/cc)", fontsize=9, color=COLORS['DENS'], fontweight='bold')
    ax_dens.tick_params(axis='x', colors=COLORS['DENS'], labelsize=8)
    ax_dens.xaxis.set_label_position('top')
    ax_dens.xaxis.tick_top()
    ax_dens.spines['top'].set_color(COLORS['DENS'])
    ax_dens.spines['top'].set_linewidth(2)
    
    # Secondary axis: Neutron (Blue) - reversed scale
    ax_neut = ax_dens.twiny()
    neut_min = settings.get('neut_min', -0.15)
    neut_max = settings.get('neut_max', 0.45)
    ax_neut.set_xlim(neut_max, neut_min)  # Reversed!
    ax_neut.set_xlabel("NPHI (v/v)", fontsize=9, color=COLORS['NEUT'], fontweight='bold')
    ax_neut.tick_params(axis='x', colors=COLORS['NEUT'], labelsize=8)
    ax_neut.spines['top'].set_position(('outward', 35))
    ax_neut.spines['top'].set_color(COLORS['NEUT'])
    ax_neut.spines['top'].set_linewidth(2)
    
    # Get data
    dens_data = None
    neut_data = None
    
    # Plot Density curve
    if mapping['DENS'] and mapping['DENS'] in df.columns:
        dens_data = df[mapping['DENS']].values
        ax_dens.plot(dens_data, depth, color=COLORS['DENS'], linewidth=1.5, label='RHOB')
    
    # Plot Neutron curve (dashed for distinction)
    if mapping['NEUT'] and mapping['NEUT'] in df.columns:
        neut_data = df[mapping['NEUT']].values
        # Transform neutron to density axis for overlay
        neut_transformed = _transform_neutron_to_density(neut_data, neut_min, neut_max, dens_min, dens_max)
        ax_dens.plot(neut_transformed, depth, color=COLORS['NEUT'], 
                    linewidth=1.5, linestyle='--', label='NPHI')
        
        # Crossover fill if enabled
        if show_fill and dens_data is not None:
            _add_crossover_fill(ax_dens, dens_data, neut_transformed, depth)
    
    # Legend
    if dens_data is not None or neut_data is not None:
        handles = []
        if mapping['DENS'] and mapping['DENS'] in df.columns:
            handles.append(mpatches.Patch(color=COLORS['DENS'], label='RHOB'))
        if mapping['NEUT'] and mapping['NEUT'] in df.columns:
            handles.append(mpatches.Patch(color=COLORS['NEUT'], label='NPHI'))
        ax_dens.legend(handles=handles, loc='upper right', fontsize=7,
                      framealpha=0.9, edgecolor=COLORS['BORDER'])
    else:
        ax.text(0.5, 0.5, "No Density/Neutron Curves", transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')


def _transform_neutron_to_density(neut_data, neut_min, neut_max, dens_min, dens_max):
    """
    Transform neutron values to density axis for overlay.
    Neutron scale is reversed relative to density.
    """
    # Normalize to 0-1, then reverse
    normalized = (neut_data - neut_min) / (neut_max - neut_min)
    reversed_norm = 1 - normalized  # Reverse for overlay
    
    # Scale to density axis
    return dens_min + reversed_norm * (dens_max - dens_min)


def _add_crossover_fill(ax, dens_data, neut_transformed, depth):
    """
    Add crossover fill between density and neutron.
    - Yellow/light when neutron > density (gas effect)
    - Grey when density > neutron (shale)
    """
    # Create masks for valid data points
    valid = ~np.isnan(dens_data) & ~np.isnan(neut_transformed)
    
    # Gas effect: Neutron crosses to the right of density (gas indication)
    ax.fill_betweenx(depth, dens_data, neut_transformed,
                     where=valid & (neut_transformed > dens_data),
                     color=COLORS['CROSS_GAS'], alpha=0.4,
                     label='Gas effect')
    
    # Shale: Density crosses to the right of neutron
    ax.fill_betweenx(depth, dens_data, neut_transformed,
                     where=valid & (dens_data > neut_transformed),
                     color=COLORS['CROSS_SHALE'], alpha=0.3,
                     label='Shale')


def export_plot_to_bytes(fig, format='png', dpi=150):
    """
    Export matplotlib figure to bytes for download.
    
    Args:
        fig: Matplotlib figure
        format: 'png', 'jpg', or 'pdf'
        dpi: Resolution for output
    
    Returns:
        Bytes of the exported image/document
    """
    import io
    buf = io.BytesIO()
    
    if format.lower() == 'pdf':
        fig.savefig(buf, format='pdf', dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    elif format.lower() in ['jpg', 'jpeg']:
        fig.savefig(buf, format='jpeg', dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none', quality=95)
    else:  # PNG default
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
    
    buf.seek(0)
    return buf.getvalue()


# =============================================================================
# TOGGLEABLE MULTI-TRACK PLOTTING
# =============================================================================

# Track definitions for toggleable plots
TRACK_DEFINITIONS = {
    'GR': {
        'name': 'Gamma Ray',
        'unit': 'gAPI',
        'color': '#00AA00',
        'scale': 'linear',
        'default_min': 0,
        'default_max': 150
    },
    'CALIPER': {
        'name': 'Caliper',
        'unit': 'in',
        'color': '#8B4513',
        'scale': 'linear',
        'default_min': 6,
        'default_max': 16
    },
    'SP': {
        'name': 'Spontaneous Potential',
        'unit': 'mV',
        'color': '#FF6600',
        'scale': 'linear',
        'default_min': -100,
        'default_max': 100
    },
    'RES_DEEP': {
        'name': 'Deep Resistivity',
        'unit': 'ohm.m',
        'color': '#0066CC',
        'scale': 'log',
        'default_min': 0.2,
        'default_max': 2000
    },
    'RES_MED': {
        'name': 'Medium Resistivity',
        'unit': 'ohm.m',
        'color': '#CC0000',
        'scale': 'log',
        'default_min': 0.2,
        'default_max': 2000
    },
    'RES_SHAL': {
        'name': 'Shallow Resistivity',
        'unit': 'ohm.m',
        'color': '#FF8C00',
        'scale': 'log',
        'default_min': 0.2,
        'default_max': 2000
    },
    'DENS': {
        'name': 'Density',
        'unit': 'g/cc',
        'color': '#CC0000',
        'scale': 'linear',
        'default_min': 1.95,
        'default_max': 2.95
    },
    'NEUT': {
        'name': 'Neutron Porosity',
        'unit': 'v/v',
        'color': '#0066CC',
        'scale': 'linear',
        'default_min': -0.15,
        'default_max': 0.45
    },
    'SONIC': {
        'name': 'Sonic',
        'unit': 'us/ft',
        'color': '#9900CC',
        'scale': 'linear',
        'default_min': 40,
        'default_max': 140
    },
    'PEF': {
        'name': 'Photoelectric Factor',
        'unit': 'b/e',
        'color': '#00CCCC',
        'scale': 'linear',
        'default_min': 0,
        'default_max': 10
    }
}


def create_toggleable_log_plot(
    df,
    mapping,
    track_visibility,
    settings=None,
    highlight_mask=None,
    highlight_color='red',
    highlight_alpha=0.3,
    title=None
):
    """
    Create a multi-track log plot with user-controlled track visibility.
    
    Args:
        df: DataFrame with log data (must include 'DEPTH' column)
        mapping: Curve mapping dictionary (curve_type -> column_name)
        track_visibility: Dictionary of track_type -> bool (visible/hidden)
        settings: Optional plot settings (limits, scales, etc.)
        highlight_mask: Optional boolean mask to highlight data points
        highlight_color: Color for highlighted regions
        highlight_alpha: Alpha for highlighted regions
        title: Optional main title for the plot
        
    Returns:
        Matplotlib figure
    """
    if settings is None:
        settings = {}
    
    # Determine visible tracks
    visible_tracks = [t for t in TRACK_DEFINITIONS.keys() 
                      if track_visibility.get(t, False) and 
                      mapping.get(t) and 
                      mapping.get(t) in df.columns]
    
    if not visible_tracks:
        # Return empty figure with message
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No tracks selected or no data available", 
                ha='center', va='center', fontsize=12, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        return fig
    
    # Calculate figure dimensions
    depth = df['DEPTH'].values
    depth_min = df['DEPTH'].min()
    depth_max = df['DEPTH'].max()
    depth_range = depth_max - depth_min
    
    scale_ratio = settings.get('scale_ratio', 500)
    height_cm = (depth_range * 100) / scale_ratio
    height_in = max(6, min(30, height_cm / 2.54))
    
    track_width = 3.5
    fig_width = track_width * len(visible_tracks) + 1.5
    
    # Create figure
    fig, axes = plt.subplots(
        nrows=1, 
        ncols=len(visible_tracks), 
        figsize=(fig_width, height_in),
        sharey=True
    )
    
    # Handle single track case
    if len(visible_tracks) == 1:
        axes = [axes]
    
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(wspace=0.05, top=0.93, bottom=0.03, left=0.1, right=0.98)
    
    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98)
    
    # Plot each visible track
    for i, track_type in enumerate(visible_tracks):
        ax = axes[i]
        track_def = TRACK_DEFINITIONS[track_type]
        curve_name = mapping[track_type]
        curve_data = df[curve_name].values
        
        # Apply common styling
        ax.set_facecolor(COLORS['TRACK_BG'])
        ax.grid(True, which='major', linestyle='-', linewidth=0.5, 
                color=COLORS['GRID'], alpha=0.7)
        ax.grid(True, which='minor', linestyle=':', linewidth=0.3, 
                color=COLORS['GRID'], alpha=0.4)
        
        for spine in ax.spines.values():
            spine.set_color(COLORS['BORDER'])
            spine.set_linewidth(1)
        
        # Track header
        ax.set_title(track_def['name'], fontsize=10, fontweight='bold', pad=10)
        
        # Get limits from settings or use defaults
        limit_key_min = f"{track_type.lower()}_min"
        limit_key_max = f"{track_type.lower()}_max"
        x_min = settings.get(limit_key_min, track_def['default_min'])
        x_max = settings.get(limit_key_max, track_def['default_max'])
        
        # Apply scale type
        if track_def['scale'] == 'log':
            ax.set_xscale('log')
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
        
        ax.set_xlim(x_min, x_max)
        
        # X-axis styling
        color = settings.get(f"{track_type.lower()}_color", track_def['color'])
        ax.set_xlabel(f"{track_def['unit']}", fontsize=9, color=color, fontweight='bold')
        ax.tick_params(axis='x', colors=color, labelsize=8)
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        ax.spines['top'].set_color(color)
        ax.spines['top'].set_linewidth(2)
        
        # Plot the curve
        line_style = settings.get(f"{track_type.lower()}_linestyle", '-')
        line_width = settings.get(f"{track_type.lower()}_linewidth", 1.5)
        ax.plot(curve_data, depth, color=color, linewidth=line_width, 
                linestyle=line_style, label=curve_name)
        
        # Add highlight regions if provided
        if highlight_mask is not None and len(highlight_mask) == len(df):
            _add_highlight_regions(ax, depth, highlight_mask, 
                                   highlight_color, highlight_alpha, x_min, x_max)
        
        # Only show y-axis on first track
        if i == 0:
            ax.set_ylabel("Depth (m)", fontsize=10, fontweight='bold')
        
        ax.tick_params(axis='y', labelsize=8)
    
    # Set y-axis limits (inverted for depth)
    axes[0].set_ylim(depth_max, depth_min)
    
    fig.tight_layout()
    return fig


def _add_highlight_regions(ax, depth, mask, color, alpha, x_min, x_max):
    """Add highlighted regions to a track based on a boolean mask."""
    # Find contiguous regions
    changes = np.diff(mask.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    
    # Handle edge cases
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [len(mask)]])
    
    # Draw shaded rectangles for each region
    for start, end in zip(starts, ends):
        if start < len(depth) and end <= len(depth):
            d_start = depth[start]
            d_end = depth[min(end, len(depth)-1)]
            ax.axhspan(d_start, d_end, alpha=alpha, color=color, zorder=0)


def create_comparison_log_plot(
    df_before,
    df_after,
    curve_name,
    depth_col='DEPTH',
    title="Before/After Comparison",
    before_label="Original",
    after_label="Processed",
    settings=None
):
    """
    Create a side-by-side comparison plot for before/after analysis.
    
    Args:
        df_before: DataFrame with original data
        df_after: DataFrame with processed data
        curve_name: Name of the curve column to compare
        depth_col: Name of depth column
        title: Plot title
        before_label: Label for original data
        after_label: Label for processed data
        settings: Optional plot settings
        
    Returns:
        Matplotlib figure
    """
    if settings is None:
        settings = {}
    
    # Get data
    depth_before = df_before[depth_col].values
    depth_after = df_after[depth_col].values
    signal_before = df_before[curve_name].values if curve_name in df_before.columns else None
    signal_after = df_after[curve_name].values if curve_name in df_after.columns else None
    
    if signal_before is None and signal_after is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"Curve '{curve_name}' not found", 
                ha='center', va='center', fontsize=12, color='gray')
        return fig
    
    # Calculate dimensions
    all_depths = np.concatenate([depth_before, depth_after])
    depth_min = np.nanmin(all_depths)
    depth_max = np.nanmax(all_depths)
    depth_range = depth_max - depth_min
    
    scale_ratio = settings.get('scale_ratio', 500)
    height_in = max(6, min(20, (depth_range * 100) / scale_ratio / 2.54))
    
    # Create side-by-side figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, height_in), sharey=True)
    fig.suptitle(title, fontsize=12, fontweight='bold')
    
    # Common styling
    for ax in [ax1, ax2]:
        ax.set_facecolor(COLORS['TRACK_BG'])
        ax.grid(True, which='major', linestyle='-', linewidth=0.5, color=COLORS['GRID'], alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color(COLORS['BORDER'])
            spine.set_linewidth(1)
    
    # Calculate common x limits
    all_values = []
    if signal_before is not None:
        all_values.extend(signal_before[~np.isnan(signal_before)])
    if signal_after is not None:
        all_values.extend(signal_after[~np.isnan(signal_after)])
    
    if all_values:
        x_min = np.percentile(all_values, 1)
        x_max = np.percentile(all_values, 99)
        margin = (x_max - x_min) * 0.1
        x_min -= margin
        x_max += margin
    else:
        x_min, x_max = 0, 1
    
    # Plot before
    ax1.set_title(before_label, fontsize=10, fontweight='bold', pad=10)
    ax1.set_xlim(x_min, x_max)
    ax1.set_xlabel(curve_name, fontsize=9, fontweight='bold')
    ax1.xaxis.set_label_position('top')
    ax1.xaxis.tick_top()
    ax1.set_ylabel("Depth (m)", fontsize=10, fontweight='bold')
    
    if signal_before is not None:
        ax1.plot(signal_before, depth_before, color='#333333', linewidth=1.5)
    
    # Plot after
    ax2.set_title(after_label, fontsize=10, fontweight='bold', pad=10)
    ax2.set_xlim(x_min, x_max)
    ax2.set_xlabel(curve_name, fontsize=9, fontweight='bold')
    ax2.xaxis.set_label_position('top')
    ax2.xaxis.tick_top()
    
    if signal_after is not None:
        ax2.plot(signal_after, depth_after, color='#00AA00', linewidth=1.5)
    
    # Set y limits
    ax1.set_ylim(depth_max, depth_min)
    
    plt.tight_layout()
    return fig


def create_single_curve_plot(
    df,
    curve_name,
    depth_col='DEPTH',
    title=None,
    color='#0066CC',
    settings=None,
    highlight_mask=None,
    highlight_color='red',
    highlight_alpha=0.3,
    annotation_depths=None,
    annotation_labels=None
):
    """
    Create a single curve vertical log plot.
    
    Args:
        df: DataFrame with data
        curve_name: Column name to plot
        depth_col: Name of depth column
        title: Optional plot title
        color: Curve color
        settings: Optional settings dict
        highlight_mask: Optional boolean mask for highlighting
        highlight_color: Color for highlights
        highlight_alpha: Alpha for highlights
        annotation_depths: List of depths for annotations
        annotation_labels: List of labels for annotations
        
    Returns:
        Matplotlib figure
    """
    if settings is None:
        settings = {}
    
    if curve_name not in df.columns:
        fig, ax = plt.subplots(figsize=(4, 6))
        ax.text(0.5, 0.5, f"Curve '{curve_name}' not found", 
                ha='center', va='center', fontsize=12, color='gray')
        return fig
    
    depth = df[depth_col].values
    signal = df[curve_name].values
    
    depth_min = np.nanmin(depth)
    depth_max = np.nanmax(depth)
    depth_range = depth_max - depth_min
    
    scale_ratio = settings.get('scale_ratio', 500)
    height_in = max(6, min(20, (depth_range * 100) / scale_ratio / 2.54))
    
    fig, ax = plt.subplots(figsize=(5, height_in))
    fig.patch.set_facecolor('white')
    
    # Styling
    ax.set_facecolor(COLORS['TRACK_BG'])
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, color=COLORS['GRID'], alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3, color=COLORS['GRID'], alpha=0.4)
    
    for spine in ax.spines.values():
        spine.set_color(COLORS['BORDER'])
        spine.set_linewidth(1)
    
    # Title
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    else:
        ax.set_title(curve_name, fontsize=11, fontweight='bold', pad=10)
    
    # X limits
    valid_signal = signal[~np.isnan(signal)]
    if len(valid_signal) > 0:
        x_min = settings.get('x_min', np.percentile(valid_signal, 1))
        x_max = settings.get('x_max', np.percentile(valid_signal, 99))
        margin = (x_max - x_min) * 0.1
        x_min -= margin
        x_max += margin
    else:
        x_min, x_max = 0, 1
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(depth_max, depth_min)
    
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.spines['top'].set_color(color)
    ax.spines['top'].set_linewidth(2)
    ax.tick_params(axis='x', colors=color, labelsize=8)
    ax.set_ylabel("Depth (m)", fontsize=10, fontweight='bold')
    
    # Plot curve
    ax.plot(signal, depth, color=color, linewidth=1.5)
    
    # Highlight regions
    if highlight_mask is not None and len(highlight_mask) == len(df):
        _add_highlight_regions(ax, depth, highlight_mask, 
                               highlight_color, highlight_alpha, x_min, x_max)
    
    # Annotations
    if annotation_depths is not None and annotation_labels is not None:
        for d, label in zip(annotation_depths, annotation_labels):
            ax.axhline(y=d, color='orange', linestyle='--', linewidth=1, alpha=0.7)
            ax.annotate(label, xy=(x_max, d), fontsize=8, color='orange',
                       va='center', ha='right')
    
    plt.tight_layout()
    return fig


def create_outlier_visualization(
    df,
    curves,
    outlier_mask,
    depth_col='DEPTH',
    settings=None,
    title="Outlier Detection Results"
):
    """
    Create a comprehensive outlier visualization showing detected outliers on each curve.
    
    Args:
        df: DataFrame with log data
        curves: List of curve names analyzed
        outlier_mask: Boolean mask indicating outlier positions
        depth_col: Name of depth column
        settings: Optional plot settings
        title: Main title for the plot
        
    Returns:
        Matplotlib figure
    """
    if settings is None:
        settings = {}
    
    depth = df[depth_col].values
    depth_min = np.nanmin(depth)
    depth_max = np.nanmax(depth)
    depth_range = depth_max - depth_min
    
    n_curves = len(curves)
    if n_curves == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No curves to display", ha='center', va='center')
        return fig
    
    # Calculate figure size
    scale_ratio = settings.get('scale_ratio', 500)
    height_in = max(6, min(20, (depth_range * 100) / scale_ratio / 2.54))
    
    # Add extra width for histogram column
    fig_width = 3.5 * n_curves + 3
    
    # Create figure with curves + histogram
    fig = plt.figure(figsize=(fig_width, height_in), facecolor='white')
    
    # GridSpec: curves tracks + depth distribution histogram
    gs = gridspec.GridSpec(1, n_curves + 1, figure=fig,
                           width_ratios=[3.5] * n_curves + [2],
                           wspace=0.1)
    
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_curves)]
    ax_hist = fig.add_subplot(gs[0, n_curves])
    
    # Share y-axis across all curve tracks
    for ax in axes[1:]:
        ax.sharey(axes[0])
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    # Colors for different curves
    curve_colors = ['#0066CC', '#CC0000', '#00AA00', '#FF8C00', '#9900CC', '#00CCCC']
    
    # Plot each curve with outliers highlighted
    for i, curve_name in enumerate(curves):
        ax = axes[i]
        color = curve_colors[i % len(curve_colors)]
        
        ax.set_facecolor(COLORS['TRACK_BG'])
        ax.grid(True, linestyle='-', linewidth=0.5, color=COLORS['GRID'], alpha=0.7)
        
        for spine in ax.spines.values():
            spine.set_color(COLORS['BORDER'])
            spine.set_linewidth(1)
        
        if curve_name in df.columns:
            data = df[curve_name].values
            
            # Plot the curve
            ax.plot(data, depth, color=color, linewidth=1.2, alpha=0.7, label='Data')
            
            # Plot outliers as red markers
            outlier_depths = depth[outlier_mask]
            outlier_values = data[outlier_mask]
            
            ax.scatter(outlier_values, outlier_depths, 
                      c='#ef4444', s=25, marker='o', zorder=5,
                      edgecolors='white', linewidths=0.5,
                      label=f'Outliers ({np.sum(outlier_mask)})')
            
            # Add shaded regions for outlier zones
            _add_highlight_shading(ax, depth, outlier_mask, 
                                  np.nanmin(data), np.nanmax(data),
                                  '#ef4444', 0.15)
            
            # X limits with padding
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                x_min = np.percentile(valid_data, 1) - (np.ptp(valid_data) * 0.1)
                x_max = np.percentile(valid_data, 99) + (np.ptp(valid_data) * 0.1)
                ax.set_xlim(x_min, x_max)
        
        # Title and styling
        ax.set_title(curve_name, fontsize=10, fontweight='bold', color=color, pad=8)
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        ax.tick_params(axis='x', colors=color, labelsize=7)
        ax.spines['top'].set_color(color)
        ax.spines['top'].set_linewidth(2)
        
        if i == 0:
            ax.set_ylabel("Depth", fontsize=10, fontweight='bold')
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        
        ax.legend(loc='lower right', fontsize=7, framealpha=0.9)
    
    # Set y-axis limits (inverted for depth)
    axes[0].set_ylim(depth_max, depth_min)
    
    # Depth distribution histogram
    ax_hist.set_facecolor('#f8f9fa')
    ax_hist.set_title("Outlier\nDistribution", fontsize=10, fontweight='bold', pad=8)
    
    outlier_depths = depth[outlier_mask]
    if len(outlier_depths) > 0:
        # Create horizontal histogram (depth bins)
        n_bins = min(30, len(outlier_depths) // 2 + 5)
        bins = np.linspace(depth_min, depth_max, n_bins)
        
        counts, bin_edges = np.histogram(outlier_depths, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Horizontal bar chart
        ax_hist.barh(bin_centers, counts, height=(bins[1] - bins[0]) * 0.9,
                    color='#ef4444', alpha=0.7, edgecolor='#dc2626')
        
        ax_hist.set_xlabel("Count", fontsize=9)
        ax_hist.set_ylim(depth_max, depth_min)
        ax_hist.set_xlim(0, max(counts) * 1.2 if max(counts) > 0 else 1)
        ax_hist.tick_params(axis='y', labelleft=False)
        
        # Add total count annotation
        ax_hist.text(0.5, 0.02, f"Total: {np.sum(outlier_mask)}", 
                    transform=ax_hist.transAxes, fontsize=9, fontweight='bold',
                    ha='center', color='#dc2626')
    else:
        ax_hist.text(0.5, 0.5, "No\nOutliers", ha='center', va='center',
                    fontsize=11, color='#22c55e', fontweight='bold',
                    transform=ax_hist.transAxes)
    
    for spine in ax_hist.spines.values():
        spine.set_color(COLORS['BORDER'])
        spine.set_linewidth(1)
    
    plt.tight_layout()
    return fig


def create_noise_visualization(
    df,
    curves,
    noise_mask,
    variance_data=None,
    slope_data=None,
    depth_col='DEPTH',
    settings=None,
    title="Noise Detection Results",
    noise_type="Tool Startup"
):
    """
    Create a comprehensive noise detection visualization.
    
    Shows each curve with noisy regions highlighted and detection metrics.
    
    Args:
        df: DataFrame with log data
        curves: List of curve names analyzed
        noise_mask: Boolean mask indicating noisy positions
        variance_data: Rolling variance array (optional)
        slope_data: Rolling slope array (optional)
        depth_col: Name of depth column
        settings: Optional plot settings
        title: Main title
        noise_type: Type of noise being shown
        
    Returns:
        Matplotlib figure
    """
    if settings is None:
        settings = {}
    
    depth = df[depth_col].values
    depth_min = np.nanmin(depth)
    depth_max = np.nanmax(depth)
    depth_range = depth_max - depth_min
    
    n_curves = len(curves)
    if n_curves == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No curves to display", ha='center', va='center')
        return fig
    
    # Calculate figure size
    scale_ratio = settings.get('scale_ratio', 500)
    height_in = max(6, min(20, (depth_range * 100) / scale_ratio / 2.54))
    
    # Add columns for metrics if available
    has_metrics = variance_data is not None or slope_data is not None
    n_cols = n_curves + (2 if has_metrics else 1)
    
    fig_width = 3 * n_cols + 1
    
    fig = plt.figure(figsize=(fig_width, height_in), facecolor='white')
    
    # Create GridSpec
    if has_metrics:
        width_ratios = [3] * n_curves + [2, 2]
    else:
        width_ratios = [3] * n_curves + [2]
    
    gs = gridspec.GridSpec(1, n_cols, figure=fig, width_ratios=width_ratios, wspace=0.12)
    
    # Create axes
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_curves)]
    
    if has_metrics:
        ax_var = fig.add_subplot(gs[0, n_curves])
        ax_slope = fig.add_subplot(gs[0, n_curves + 1])
    else:
        ax_summary = fig.add_subplot(gs[0, n_curves])
    
    # Share y-axis
    for ax in axes[1:]:
        ax.sharey(axes[0])
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    curve_colors = ['#0066CC', '#CC0000', '#00AA00', '#FF8C00', '#9900CC']
    
    # Find noise depth range
    noise_depths = depth[noise_mask]
    if len(noise_depths) > 0:
        noise_start = np.min(noise_depths)
        noise_end = np.max(noise_depths)
    else:
        noise_start = noise_end = None
    
    # Plot each curve
    for i, curve_name in enumerate(curves):
        ax = axes[i]
        color = curve_colors[i % len(curve_colors)]
        
        ax.set_facecolor(COLORS['TRACK_BG'])
        ax.grid(True, linestyle='-', linewidth=0.5, color=COLORS['GRID'], alpha=0.7)
        
        for spine in ax.spines.values():
            spine.set_color(COLORS['BORDER'])
            spine.set_linewidth(1)
        
        if curve_name in df.columns:
            data = df[curve_name].values
            
            # Plot the full curve
            ax.plot(data, depth, color=color, linewidth=1.2, alpha=0.7)
            
            # Highlight noisy portion with thicker orange line
            if np.any(noise_mask):
                noisy_data = data.copy()
                noisy_data[~noise_mask] = np.nan
                ax.plot(noisy_data, depth, color='#ff8c00', linewidth=3, alpha=0.9,
                       label=f'Noise ({np.sum(noise_mask)} pts)')
                
                # Add shaded region
                _add_highlight_shading(ax, depth, noise_mask,
                                      np.nanmin(data), np.nanmax(data),
                                      '#ff8c00', 0.25)
            
            # Mark noise boundaries with horizontal lines
            if noise_start is not None:
                ax.axhline(y=noise_start, color='#ff8c00', linestyle='--', 
                          linewidth=1.5, alpha=0.8)
                ax.axhline(y=noise_end, color='#ff8c00', linestyle='--',
                          linewidth=1.5, alpha=0.8)
            
            # X limits
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                x_min = np.percentile(valid_data, 1) - (np.ptp(valid_data) * 0.1)
                x_max = np.percentile(valid_data, 99) + (np.ptp(valid_data) * 0.1)
                ax.set_xlim(x_min, x_max)
        
        ax.set_title(curve_name, fontsize=10, fontweight='bold', color=color, pad=8)
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        ax.tick_params(axis='x', colors=color, labelsize=7)
        ax.spines['top'].set_color(color)
        ax.spines['top'].set_linewidth(2)
        
        if i == 0:
            ax.set_ylabel("Depth", fontsize=10, fontweight='bold')
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        
        if np.any(noise_mask):
            ax.legend(loc='lower right', fontsize=7, framealpha=0.9)
    
    axes[0].set_ylim(depth_max, depth_min)
    
    # Plot metrics if available
    if has_metrics:
        # Variance plot
        ax_var.set_facecolor('#f8f9fa')
        ax_var.set_title("Rolling\nVariance", fontsize=9, fontweight='bold', pad=8)
        
        if variance_data is not None:
            ax_var.plot(variance_data, depth, color='#6366f1', linewidth=1)
            # Highlight noisy region
            noisy_var = variance_data.copy() if isinstance(variance_data, np.ndarray) else np.array(variance_data)
            if len(noisy_var) == len(noise_mask):
                noisy_var_plot = noisy_var.copy()
                noisy_var_plot[~noise_mask] = np.nan
                ax_var.plot(noisy_var_plot, depth, color='#ff8c00', linewidth=2)
            
            ax_var.set_xlabel("Variance", fontsize=8)
        
        ax_var.set_ylim(depth_max, depth_min)
        ax_var.tick_params(axis='y', labelleft=False)
        ax_var.xaxis.set_label_position('top')
        ax_var.xaxis.tick_top()
        
        for spine in ax_var.spines.values():
            spine.set_color(COLORS['BORDER'])
            spine.set_linewidth(1)
        
        # Slope plot
        ax_slope.set_facecolor('#f8f9fa')
        ax_slope.set_title("Rolling\nSlope", fontsize=9, fontweight='bold', pad=8)
        
        if slope_data is not None:
            ax_slope.plot(slope_data, depth, color='#10b981', linewidth=1)
            noisy_slope = slope_data.copy() if isinstance(slope_data, np.ndarray) else np.array(slope_data)
            if len(noisy_slope) == len(noise_mask):
                noisy_slope_plot = noisy_slope.copy()
                noisy_slope_plot[~noise_mask] = np.nan
                ax_slope.plot(noisy_slope_plot, depth, color='#ff8c00', linewidth=2)
            
            ax_slope.set_xlabel("Slope", fontsize=8)
        
        ax_slope.set_ylim(depth_max, depth_min)
        ax_slope.tick_params(axis='y', labelleft=False)
        ax_slope.xaxis.set_label_position('top')
        ax_slope.xaxis.tick_top()
        
        for spine in ax_slope.spines.values():
            spine.set_color(COLORS['BORDER'])
            spine.set_linewidth(1)
    else:
        # Summary panel
        ax_summary.set_facecolor('#fff7ed')
        ax_summary.set_title("Noise\nSummary", fontsize=10, fontweight='bold', pad=8)
        ax_summary.axis('off')
        
        n_noise = np.sum(noise_mask)
        pct = (n_noise / len(noise_mask)) * 100 if len(noise_mask) > 0 else 0
        
        summary_text = f"Type: {noise_type}\n\n"
        summary_text += f"Samples: {n_noise}\n"
        summary_text += f"Affected: {pct:.1f}%\n\n"
        if noise_start is not None:
            summary_text += f"Start: {noise_start:.1f}\n"
            summary_text += f"End: {noise_end:.1f}"
        
        ax_summary.text(0.5, 0.5, summary_text, transform=ax_summary.transAxes,
                       fontsize=10, ha='center', va='center',
                       fontweight='bold', color='#c2410c')
    
    plt.tight_layout()
    return fig


def create_spike_visualization(
    df,
    curves,
    spike_mask,
    depth_col='DEPTH',
    settings=None,
    title="Spike Noise Detection"
):
    """
    Create visualization for spike noise detection.
    
    Args:
        df: DataFrame with log data
        curves: List of curve names analyzed
        spike_mask: Boolean mask indicating spike positions
        depth_col: Name of depth column
        settings: Optional plot settings
        title: Main title
        
    Returns:
        Matplotlib figure
    """
    if settings is None:
        settings = {}
    
    depth = df[depth_col].values
    depth_min = np.nanmin(depth)
    depth_max = np.nanmax(depth)
    depth_range = depth_max - depth_min
    
    n_curves = len(curves)
    if n_curves == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No curves to display", ha='center', va='center')
        return fig
    
    scale_ratio = settings.get('scale_ratio', 500)
    height_in = max(6, min(20, (depth_range * 100) / scale_ratio / 2.54))
    
    fig_width = 3.5 * n_curves + 2
    
    fig = plt.figure(figsize=(fig_width, height_in), facecolor='white')
    
    gs = gridspec.GridSpec(1, n_curves + 1, figure=fig,
                           width_ratios=[3.5] * n_curves + [1.5],
                           wspace=0.1)
    
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_curves)]
    ax_count = fig.add_subplot(gs[0, n_curves])
    
    for ax in axes[1:]:
        ax.sharey(axes[0])
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    curve_colors = ['#0066CC', '#CC0000', '#00AA00', '#FF8C00', '#9900CC']
    
    for i, curve_name in enumerate(curves):
        ax = axes[i]
        color = curve_colors[i % len(curve_colors)]
        
        ax.set_facecolor(COLORS['TRACK_BG'])
        ax.grid(True, linestyle='-', linewidth=0.5, color=COLORS['GRID'], alpha=0.7)
        
        for spine in ax.spines.values():
            spine.set_color(COLORS['BORDER'])
            spine.set_linewidth(1)
        
        if curve_name in df.columns:
            data = df[curve_name].values
            
            # Plot curve
            ax.plot(data, depth, color=color, linewidth=1.2, alpha=0.7)
            
            # Mark spikes with red X markers
            spike_depths = depth[spike_mask]
            spike_values = data[spike_mask]
            
            ax.scatter(spike_values, spike_depths,
                      c='#dc3545', s=40, marker='x', linewidths=2,
                      zorder=5, label=f'Spikes ({np.sum(spike_mask)})')
            
            # X limits
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                x_min = np.percentile(valid_data, 1) - (np.ptp(valid_data) * 0.15)
                x_max = np.percentile(valid_data, 99) + (np.ptp(valid_data) * 0.15)
                ax.set_xlim(x_min, x_max)
        
        ax.set_title(curve_name, fontsize=10, fontweight='bold', color=color, pad=8)
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        ax.tick_params(axis='x', colors=color, labelsize=7)
        ax.spines['top'].set_color(color)
        ax.spines['top'].set_linewidth(2)
        
        if i == 0:
            ax.set_ylabel("Depth", fontsize=10, fontweight='bold')
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        
        ax.legend(loc='lower right', fontsize=7, framealpha=0.9)
    
    axes[0].set_ylim(depth_max, depth_min)
    
    # Summary panel
    ax_count.set_facecolor('#fef2f2')
    ax_count.axis('off')
    ax_count.set_title("Summary", fontsize=10, fontweight='bold', pad=8)
    
    n_spikes = np.sum(spike_mask)
    pct = (n_spikes / len(spike_mask)) * 100 if len(spike_mask) > 0 else 0
    
    ax_count.text(0.5, 0.6, f"⚡ {n_spikes}", fontsize=24, fontweight='bold',
                 ha='center', va='center', transform=ax_count.transAxes,
                 color='#dc3545')
    ax_count.text(0.5, 0.4, f"spikes\n({pct:.2f}%)", fontsize=10,
                 ha='center', va='center', transform=ax_count.transAxes,
                 color='#7f1d1d')
    
    for spine in ax_count.spines.values():
        spine.set_color(COLORS['BORDER'])
        spine.set_linewidth(1)
    
    plt.tight_layout()
    return fig


def create_overlay_plot(
    df,
    curve_names,
    depth_col='DEPTH',
    title="Curve Overlay",
    colors=None,
    settings=None,
    normalize=False
):
    """
    Create an overlay plot with multiple curves on the same track.
    
    Useful for comparing curves or showing before/after on same axis.
    
    Args:
        df: DataFrame with data
        curve_names: List of column names to plot
        depth_col: Name of depth column
        title: Plot title
        colors: List of colors (auto-generated if None)
        settings: Optional settings dict
        normalize: If True, normalize all curves to 0-1 range
        
    Returns:
        Matplotlib figure
    """
    if settings is None:
        settings = {}
    
    if colors is None:
        default_colors = ['#0066CC', '#CC0000', '#00AA00', '#FF8C00', '#9900CC', '#00CCCC']
        colors = [default_colors[i % len(default_colors)] for i in range(len(curve_names))]
    
    # Validate curves
    valid_curves = [c for c in curve_names if c in df.columns]
    if not valid_curves:
        fig, ax = plt.subplots(figsize=(5, 6))
        ax.text(0.5, 0.5, "No valid curves found", 
                ha='center', va='center', fontsize=12, color='gray')
        return fig
    
    depth = df[depth_col].values
    depth_min = np.nanmin(depth)
    depth_max = np.nanmax(depth)
    depth_range = depth_max - depth_min
    
    scale_ratio = settings.get('scale_ratio', 500)
    height_in = max(6, min(20, (depth_range * 100) / scale_ratio / 2.54))
    
    fig, ax = plt.subplots(figsize=(6, height_in))
    fig.patch.set_facecolor('white')
    
    # Styling
    ax.set_facecolor(COLORS['TRACK_BG'])
    ax.grid(True, which='major', linestyle='-', linewidth=0.5, color=COLORS['GRID'], alpha=0.7)
    
    for spine in ax.spines.values():
        spine.set_color(COLORS['BORDER'])
        spine.set_linewidth(1)
    
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    
    # Plot each curve
    handles = []
    for i, curve_name in enumerate(valid_curves):
        signal = df[curve_name].values.copy()
        
        if normalize:
            valid_vals = signal[~np.isnan(signal)]
            if len(valid_vals) > 0:
                min_val, max_val = np.nanmin(valid_vals), np.nanmax(valid_vals)
                if max_val - min_val > 0:
                    signal = (signal - min_val) / (max_val - min_val)
        
        color = colors[i] if i < len(colors) else colors[-1]
        line, = ax.plot(signal, depth, color=color, linewidth=1.5, label=curve_name)
        handles.append(line)
    
    ax.set_ylim(depth_max, depth_min)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_ylabel("Depth (m)", fontsize=10, fontweight='bold')
    
    if normalize:
        ax.set_xlabel("Normalized Value", fontsize=9, fontweight='bold')
        ax.set_xlim(-0.1, 1.1)
    
    ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.9)
    
    plt.tight_layout()
    return fig


# =============================================================================
# ROCK CLASSIFICATION VISUALIZATION FUNCTIONS
# =============================================================================

def create_facies_log_display(
    df,
    cluster_labels,
    cluster_interpretations=None,
    mapping=None,
    header_info=None,
    settings=None,
    facies_colors=None
):
    """
    Create a professional log display with facies classification track.
    
    Shows the standard well log curves alongside a color-coded facies/lithology
    track based on clustering results.
    
    Args:
        df: DataFrame with well log data (must include 'DEPTH')
        cluster_labels: Array of cluster assignments for each sample
        cluster_interpretations: Dict mapping cluster ID to rock type name
        mapping: Curve mapping dictionary
        header_info: Well header information
        settings: Plot settings
        facies_colors: Optional list of colors for each cluster
        
    Returns:
        Matplotlib figure
    """
    if settings is None:
        settings = {}
    if mapping is None:
        mapping = {}
    
    # Get depth data
    depth = df['DEPTH'].values
    depth_min = np.nanmin(depth)
    depth_max = np.nanmax(depth)
    depth_range = depth_max - depth_min
    
    # Calculate figure dimensions
    scale_ratio = settings.get('scale_ratio', 500)
    depth_unit = settings.get('depth_unit', 'm')
    height_cm = (depth_range * 100) / scale_ratio
    height_in = max(8, min(25, height_cm / 2.54))
    
    # Create figure: Depth + Facies + GR + D-N tracks
    fig = plt.figure(figsize=(12, height_in), facecolor='white')
    
    # GridSpec layout
    if header_info:
        gs = gridspec.GridSpec(2, 5, figure=fig,
                               height_ratios=[0.06, 0.94],
                               width_ratios=[0.6, 1.2, 2.5, 2.5, 2.5],
                               wspace=0.02, hspace=0.02)
        ax_header = fig.add_subplot(gs[0, :])
        _draw_well_header(ax_header, header_info, depth_unit)
        
        ax_depth = fig.add_subplot(gs[1, 0])
        ax_facies = fig.add_subplot(gs[1, 1], sharey=ax_depth)
        ax_gr = fig.add_subplot(gs[1, 2], sharey=ax_depth)
        ax_dn = fig.add_subplot(gs[1, 3], sharey=ax_depth)
        ax_curves = fig.add_subplot(gs[1, 4], sharey=ax_depth)
    else:
        gs = gridspec.GridSpec(1, 5, figure=fig,
                               width_ratios=[0.6, 1.2, 2.5, 2.5, 2.5],
                               wspace=0.02)
        ax_depth = fig.add_subplot(gs[0, 0])
        ax_facies = fig.add_subplot(gs[0, 1], sharey=ax_depth)
        ax_gr = fig.add_subplot(gs[0, 2], sharey=ax_depth)
        ax_dn = fig.add_subplot(gs[0, 3], sharey=ax_depth)
        ax_curves = fig.add_subplot(gs[0, 4], sharey=ax_depth)
    
    # Draw depth track
    _draw_depth_track(ax_depth, depth, depth_min, depth_max, depth_unit)
    
    # =========================================================================
    # FACIES TRACK (color-coded lithology)
    # =========================================================================
    ax_facies.set_facecolor('#f8f9fa')
    ax_facies.set_title("FACIES\n(Lithology)", fontsize=9, fontweight='bold', pad=8)
    ax_facies.set_xlim(0, 1)
    
    # Get unique clusters
    unique_clusters = np.unique(cluster_labels[cluster_labels >= 0])
    n_clusters = len(unique_clusters)
    
    # Default colors if not provided
    if facies_colors is None:
        if n_clusters == 2:
            facies_colors = ['#FFD700', '#708090']  # Gold (Sand), SlateGray (Shale)
        elif n_clusters == 3:
            facies_colors = ['#FFD700', '#DAA520', '#708090']
        else:
            import matplotlib.cm as cm
            cmap = cm.get_cmap('Spectral')
            facies_colors = [f'#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}' 
                           for c in cmap(np.linspace(0.1, 0.9, n_clusters))]
    
    # Draw facies blocks
    for i in range(len(depth) - 1):
        label = cluster_labels[i]
        if label >= 0 and label < len(facies_colors):
            color = facies_colors[label]
            ax_facies.axhspan(depth[i], depth[i+1], color=color, alpha=0.9)
    
    # Remove x-axis
    ax_facies.xaxis.set_visible(False)
    plt.setp(ax_facies.get_yticklabels(), visible=False)
    
    # Add legend
    legend_handles = []
    for cluster_id in unique_clusters:
        if cluster_id < len(facies_colors):
            label = cluster_interpretations.get(cluster_id, f"Facies {cluster_id+1}") if cluster_interpretations else f"Facies {cluster_id+1}"
            legend_handles.append(mpatches.Patch(color=facies_colors[cluster_id], label=label))
    
    if legend_handles:
        ax_facies.legend(handles=legend_handles, loc='lower center', fontsize=6,
                        framealpha=0.95, edgecolor=COLORS['BORDER'],
                        bbox_to_anchor=(0.5, 0.02))
    
    for spine in ax_facies.spines.values():
        spine.set_color(COLORS['BORDER'])
        spine.set_linewidth(1)
    
    # =========================================================================
    # GR TRACK
    # =========================================================================
    _draw_gr_track(ax_gr, df, mapping, depth, settings, show_fill=False)
    plt.setp(ax_gr.get_yticklabels(), visible=False)
    
    # =========================================================================
    # DENSITY-NEUTRON TRACK
    # =========================================================================
    _draw_density_neutron_track(ax_dn, df, mapping, depth, settings, show_crossover=True)
    plt.setp(ax_dn.get_yticklabels(), visible=False)
    
    # =========================================================================
    # ADDITIONAL CURVES TRACK (show clustering features)
    # =========================================================================
    ax_curves.set_facecolor(COLORS['TRACK_BG'])
    ax_curves.set_title("CLUSTER\nFEATURES", fontsize=9, fontweight='bold', pad=8)
    
    # Plot some common curves if available
    curve_colors = ['#e11d48', '#0ea5e9', '#22c55e', '#f59e0b']
    curves_plotted = []
    for col in ['DT', 'SONIC', 'DTC', 'RT', 'RLLD', 'PE']:
        if col in df.columns and len(curves_plotted) < 3:
            curves_plotted.append(col)
    
    if curves_plotted:
        for i, col in enumerate(curves_plotted):
            data = df[col].values.copy()
            # Normalize for overlay
            valid = ~np.isnan(data)
            if np.sum(valid) > 0:
                data_norm = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data) + 1e-10)
                ax_curves.plot(data_norm, depth, color=curve_colors[i], linewidth=1.2, label=col)
        
        ax_curves.set_xlim(-0.1, 1.1)
        ax_curves.legend(loc='lower right', fontsize=6, framealpha=0.9)
    else:
        ax_curves.text(0.5, 0.5, "No additional\ncurves", transform=ax_curves.transAxes,
                      ha='center', va='center', fontsize=9, color='gray')
    
    ax_curves.xaxis.set_label_position('top')
    ax_curves.xaxis.tick_top()
    ax_curves.set_xlabel("Normalized", fontsize=8, fontweight='bold')
    ax_curves.xaxis.grid(True, linestyle='-', linewidth=0.5, color=COLORS['GRID'], alpha=0.7)
    ax_curves.yaxis.grid(True, linestyle='-', linewidth=0.5, color=COLORS['GRID'], alpha=0.7)
    plt.setp(ax_curves.get_yticklabels(), visible=False)
    
    for spine in ax_curves.spines.values():
        spine.set_color(COLORS['BORDER'])
        spine.set_linewidth(1)
    
    # Set depth limits
    ax_depth.set_ylim(depth_max, depth_min)
    
    plt.tight_layout()
    return fig


def create_cluster_crossplot(
    df,
    x_column,
    y_column,
    cluster_labels,
    cluster_interpretations=None,
    facies_colors=None,
    title=None,
    show_density=True
):
    """
    Create a crossplot with cluster coloring.
    
    Args:
        df: DataFrame with well log data
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        cluster_labels: Array of cluster assignments
        cluster_interpretations: Dict mapping cluster ID to rock type name
        facies_colors: Optional list of colors
        title: Plot title
        show_density: Whether to show density contours
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    
    # Get data
    x_data = df[x_column].values if x_column in df.columns else None
    y_data = df[y_column].values if y_column in df.columns else None
    
    if x_data is None or y_data is None:
        ax.text(0.5, 0.5, "Missing data columns", ha='center', va='center', 
               transform=ax.transAxes, fontsize=12)
        return fig
    
    # Filter valid data
    valid_mask = ~np.isnan(x_data) & ~np.isnan(y_data) & (cluster_labels >= 0)
    x_valid = x_data[valid_mask]
    y_valid = y_data[valid_mask]
    labels_valid = cluster_labels[valid_mask]
    
    unique_clusters = np.unique(labels_valid)
    n_clusters = len(unique_clusters)
    
    # Default colors
    if facies_colors is None:
        if n_clusters == 2:
            facies_colors = ['#FFD700', '#708090']
        elif n_clusters == 3:
            facies_colors = ['#FFD700', '#DAA520', '#708090']
        else:
            import matplotlib.cm as cm
            cmap = cm.get_cmap('Spectral')
            facies_colors = [f'#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}' 
                           for c in cmap(np.linspace(0.1, 0.9, n_clusters))]
    
    # Plot each cluster
    scatter_handles = []
    for cluster_id in unique_clusters:
        mask = labels_valid == cluster_id
        color = facies_colors[int(cluster_id)] if int(cluster_id) < len(facies_colors) else '#808080'
        label = cluster_interpretations.get(cluster_id, f"Facies {cluster_id+1}") if cluster_interpretations else f"Facies {cluster_id+1}"
        
        scatter = ax.scatter(x_valid[mask], y_valid[mask], c=color, s=20, alpha=0.6,
                           edgecolors='white', linewidths=0.3, label=label)
        scatter_handles.append(scatter)
        
        # Add cluster centroid marker
        centroid_x = np.mean(x_valid[mask])
        centroid_y = np.mean(y_valid[mask])
        ax.scatter(centroid_x, centroid_y, c=color, s=200, marker='*',
                  edgecolors='black', linewidths=1.5, zorder=10)
    
    # Styling
    ax.set_xlabel(x_column, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_column, fontsize=12, fontweight='bold')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    else:
        ax.set_title(f"Rock Type Classification: {x_column} vs {y_column}", 
                    fontsize=14, fontweight='bold', pad=15)
    
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    
    # Add statistics annotation
    stats_text = f"n = {len(x_valid):,} samples\n{n_clusters} rock types identified"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    return fig


def create_3d_cluster_scatter(
    df,
    feature_columns,
    cluster_labels,
    cluster_interpretations=None,
    facies_colors=None,
    title="3D Rock Type Classification"
):
    """
    Create an interactive 3D scatter plot using Plotly.
    
    Args:
        df: DataFrame with well log data
        feature_columns: List of 3 column names for x, y, z axes
        cluster_labels: Array of cluster assignments
        cluster_interpretations: Dict mapping cluster ID to rock type name
        facies_colors: Optional list of colors
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    
    if len(feature_columns) < 3:
        return None
    
    x_col, y_col, z_col = feature_columns[:3]
    
    # Get data
    if x_col not in df.columns or y_col not in df.columns or z_col not in df.columns:
        return None
    
    x_data = df[x_col].values
    y_data = df[y_col].values
    z_data = df[z_col].values
    
    # Filter valid
    valid_mask = ~np.isnan(x_data) & ~np.isnan(y_data) & ~np.isnan(z_data) & (cluster_labels >= 0)
    x_valid = x_data[valid_mask]
    y_valid = y_data[valid_mask]
    z_valid = z_data[valid_mask]
    labels_valid = cluster_labels[valid_mask]
    
    unique_clusters = np.unique(labels_valid)
    n_clusters = len(unique_clusters)
    
    # Default colors
    if facies_colors is None:
        if n_clusters == 2:
            facies_colors = ['#FFD700', '#708090']
        elif n_clusters == 3:
            facies_colors = ['#FFD700', '#DAA520', '#708090']
        else:
            import matplotlib.cm as cm
            cmap = cm.get_cmap('Spectral')
            facies_colors = [mpl_color_to_hex(cmap(i / max(1, n_clusters - 1)))
                           for i in range(n_clusters)]
    
    # Create traces for each cluster
    traces = []
    for cluster_id in unique_clusters:
        mask = labels_valid == cluster_id
        color = facies_colors[int(cluster_id)] if int(cluster_id) < len(facies_colors) else '#808080'
        label = cluster_interpretations.get(cluster_id, f"Facies {cluster_id+1}") if cluster_interpretations else f"Facies {cluster_id+1}"
        
        trace = go.Scatter3d(
            x=x_valid[mask],
            y=y_valid[mask],
            z=z_valid[mask],
            mode='markers',
            name=label,
            marker=dict(
                size=4,
                color=color,
                opacity=0.7,
                line=dict(width=0.5, color='white')
            ),
            hovertemplate=f"<b>{label}</b><br>" +
                         f"{x_col}: %{{x:.2f}}<br>" +
                         f"{y_col}: %{{y:.2f}}<br>" +
                         f"{z_col}: %{{z:.2f}}<extra></extra>"
        )
        traces.append(trace)
    
    # Create figure
    fig = go.Figure(data=traces)
    
    # Update layout for dark theme
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=18, color='#1e293b'),
            x=0.5
        ),
        scene=dict(
            xaxis=dict(title=x_col, backgroundcolor='#f1f5f9', gridcolor='#cbd5e1'),
            yaxis=dict(title=y_col, backgroundcolor='#f1f5f9', gridcolor='#cbd5e1'),
            zaxis=dict(title=z_col, backgroundcolor='#f1f5f9', gridcolor='#cbd5e1'),
            bgcolor='white'
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#64748b',
            borderwidth=1
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor='white',
        height=600
    )
    
    return fig


def mpl_color_to_hex(rgba):
    """Convert matplotlib RGBA tuple to hex string."""
    return f'#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}'


def create_cluster_quality_plot(k_analysis, selected_k=None):
    """
    Create a plot showing cluster quality metrics for K selection.
    
    Args:
        k_analysis: Dictionary from find_optimal_clusters()
        selected_k: The K value that was selected
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')
    
    k_range = k_analysis['k_range']
    inertias = k_analysis['inertias']
    silhouettes = k_analysis['silhouette_scores']
    
    # Elbow plot
    ax1.plot(k_range, inertias, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (K)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Inertia (WCSS)', fontsize=11, fontweight='bold')
    ax1.set_title('Elbow Method', fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    if selected_k:
        ax1.axvline(x=selected_k, color='#ef4444', linestyle='--', linewidth=2, 
                   label=f'Selected K = {selected_k}')
        ax1.legend(fontsize=10)
    
    # Highlight elbow point
    elbow_k = k_analysis.get('optimal_k_elbow', k_range[0])
    elbow_idx = k_range.index(elbow_k) if elbow_k in k_range else 0
    ax1.scatter([elbow_k], [inertias[elbow_idx]], s=150, c='#f59e0b', 
               marker='D', zorder=5, label='Elbow Point')
    
    # Silhouette plot
    colors = ['#22c55e' if s == max(silhouettes) else '#3b82f6' for s in silhouettes]
    bars = ax2.bar(k_range, silhouettes, color=colors, edgecolor='white', linewidth=1.5)
    ax2.set_xlabel('Number of Clusters (K)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Silhouette Score', fontsize=11, fontweight='bold')
    ax2.set_title('Silhouette Analysis', fontsize=13, fontweight='bold', pad=10)
    ax2.grid(True, linestyle='--', alpha=0.4, axis='y')
    
    # Add value labels on bars
    for bar, score in zip(bars, silhouettes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    if selected_k and selected_k in k_range:
        idx = k_range.index(selected_k)
        bars[idx].set_color('#ef4444')
        bars[idx].set_edgecolor('#b91c1c')
        bars[idx].set_linewidth(2)
    
    # Add interpretation
    best_k = k_analysis.get('recommended_k', k_range[np.argmax(silhouettes)])
    ax2.text(0.02, 0.98, f"Recommended K = {best_k}\n(Highest Silhouette)",
            transform=ax2.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#dcfce7', alpha=0.9))
    
    plt.tight_layout()
    return fig

