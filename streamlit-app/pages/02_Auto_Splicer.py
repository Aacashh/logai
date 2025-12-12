"""
WellLog Analyzer Pro - Auto Splicer Page
Multi-well batch auto-splicing with automatic unit conversion.

This is the "Black Box" interface that:
1. Accepts multiple LAS files (potentially from different wells)
2. Groups files by well name
3. Lets user select target well
4. Auto-detects and converts units (Feet → Meters)
5. Chain-splices into a single composite log
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import sys
import os
import io

# Add parent directories to path for shared module access
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.splicing import (
    group_files_by_well,
    batch_splice_pipeline,
    find_common_curves,
    get_recommended_correlation_curve,
    PreprocessedLAS,
    BatchSpliceResult,
    WellGroupResult,
    DEFAULT_GRID_STEP,
    DEFAULT_SEARCH_WINDOW,
    DEFAULT_DTW_WINDOW,
)
from shared.data_processing import export_to_las
from plotting import export_plot_to_bytes

# Page configuration
st.set_page_config(
    page_title="Auto Splicer | WellLog Analyzer Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');
    
    .stApp {
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid #475569;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #06b6d4, #22d3ee, #67e8f9);
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #22d3ee;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-subtitle {
        font-size: 1rem;
        color: #94a3b8;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    .pipeline-step {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-left: 4px solid #22d3ee;
        padding: 1rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
        color: #e2e8f0;
    }
    
    .pipeline-step h4 {
        color: #22d3ee;
        margin: 0 0 0.5rem 0;
        font-weight: 600;
    }
    
    .pipeline-step code {
        font-family: 'JetBrains Mono', monospace;
        background: #334155;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #67e8f9;
    }
    
    .file-table {
        background: #1e293b;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #334155;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #22d3ee;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 0.5rem;
    }
    
    .success-banner {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #34d399;
        margin: 1rem 0;
    }
    
    .success-banner h3 {
        color: #34d399;
        margin: 0;
    }
    
    .success-banner p {
        color: #d1fae5;
        margin: 0.5rem 0 0 0;
    }
    
    .upload-zone {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 2px dashed #475569;
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-zone:hover {
        border-color: #22d3ee;
        background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
    }
    
    .well-selector-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #475569;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .well-info-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .well-stat {
        background: #334155;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    
    .well-stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #22d3ee;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .well-stat-label {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }
    
    .duplicate-warning {
        background: linear-gradient(135deg, #78350f 0%, #92400e 100%);
        border-left: 4px solid #fbbf24;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        color: #fef3c7;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e293b;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #475569;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #64748b;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <div class="main-title">⚡ Auto Splicer</div>
    <div class="main-subtitle">Multi-Well Support • Automatic Unit Conversion • Intelligent Grouping</div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_composite_plot(composite_df: pd.DataFrame, correlation_curve: str,
                          file_summary: list, splice_log: list) -> plt.Figure:
    """
    Create a professional composite log plot showing the full wellbore.
    """
    depth = composite_df['DEPTH'].values
    depth_min = np.nanmin(depth)
    depth_max = np.nanmax(depth)
    depth_range = depth_max - depth_min
    
    # Calculate figure height based on depth range
    height_in = min(20, max(8, depth_range / 80))
    
    # Determine number of tracks based on available curves
    available_curves = [c for c in composite_df.columns if c != 'DEPTH']
    num_tracks = min(4, max(1, len(available_curves)))
    
    fig, axes = plt.subplots(1, num_tracks, figsize=(4 * num_tracks, height_in), sharey=True)
    fig.patch.set_facecolor('#0f172a')
    
    if num_tracks == 1:
        axes = [axes]
    
    plt.subplots_adjust(wspace=0.08, top=0.94, bottom=0.03, left=0.1, right=0.97)
    
    # Colors for different curves
    colors = ['#22d3ee', '#34d399', '#f472b6', '#fbbf24', '#a78bfa', '#fb7185']
    
    # Common styling
    for ax in axes:
        ax.set_facecolor('#1e293b')
        ax.tick_params(axis='both', which='major', labelsize=8, colors='#94a3b8')
        for spine in ax.spines.values():
            spine.set_color('#475569')
            spine.set_linewidth(1)
        ax.grid(True, which='major', linestyle='-', linewidth=0.5, color='#334155', alpha=0.7)
    
    # Plot each available curve
    curve_priority = ['GR', 'RHOB', 'NPHI', 'RT', 'DT', 'CALI', 'SP', 'PEF']
    curves_to_plot = []
    
    # First add priority curves if available
    for pc in curve_priority:
        for ac in available_curves:
            if ac.upper().startswith(pc) or pc in ac.upper():
                if ac not in curves_to_plot:
                    curves_to_plot.append(ac)
                break
    
    # Then add remaining curves
    for ac in available_curves:
        if ac not in curves_to_plot:
            curves_to_plot.append(ac)
    
    # Limit to available tracks
    curves_to_plot = curves_to_plot[:num_tracks]
    
    for i, (ax, curve) in enumerate(zip(axes, curves_to_plot)):
        color = colors[i % len(colors)]
        data = composite_df[curve].values
        
        # Determine if log scale needed
        use_log = 'RT' in curve.upper() or 'RES' in curve.upper() or 'ILD' in curve.upper()
        
        ax.set_title(curve, fontsize=11, fontweight='bold', pad=12, color='#e2e8f0')
        
        if use_log:
            ax.set_xscale('log')
            valid_data = data[data > 0]
            if len(valid_data) > 0:
                ax.set_xlim(0.1, 10000)
        else:
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                dmin, dmax = np.percentile(valid_data, [2, 98])
                margin = (dmax - dmin) * 0.1
                ax.set_xlim(dmin - margin, dmax + margin)
        
        ax.plot(data, depth, color=color, linewidth=1.2)
        
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        ax.tick_params(axis='x', colors=color)
        ax.spines['top'].set_color(color)
        ax.spines['top'].set_linewidth(2)
    
    # Set depth axis
    axes[0].set_ylim(depth_max, depth_min)  # Inverted
    axes[0].set_ylabel("Depth (m)", fontsize=11, fontweight='bold', color='#e2e8f0')
    
    fig.suptitle(f'Composite Log: {depth_min:.1f}m - {depth_max:.1f}m', 
                 fontsize=14, fontweight='bold', y=0.98, color='#22d3ee')
    
    return fig


def create_file_summary_dataframe(file_summary: list) -> pd.DataFrame:
    """Create a pandas DataFrame showing file processing summary."""
    rows = []
    for entry in file_summary:
        unit_badge = '🔵 M' if entry['original_unit'] == 'm' else '🟠 FT'
        rows.append({
            'File Name': entry['filename'],
            'Unit': unit_badge,
            'Start (m)': f"{entry['start_m']:.1f}",
            'End (m)': f"{entry['stop_m']:.1f}",
            'Action': entry['action']
        })
    
    return pd.DataFrame(rows)


def create_well_files_dataframe(files: list) -> pd.DataFrame:
    """Create a pandas DataFrame showing files for a selected well."""
    rows = []
    for f in files:
        unit_badge = '🔵 M' if f.original_unit == 'm' else '🟠 FT'
        rows.append({
            'File Name': f.filename,
            'Unit': unit_badge,
            'Start (m)': f"{f.start_depth:.1f}",
            'End (m)': f"{f.stop_depth:.1f}",
            'Curves': len(f.curves)
        })
    
    return pd.DataFrame(rows)


# =============================================================================
# SIDEBAR CONFIGURATION
# =============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Advanced Settings")
    st.markdown("*Usually no changes needed*")
    
    with st.expander("🔧 Splicing Parameters", expanded=False):
        max_search = st.slider(
            "Max Search Window (m)",
            min_value=5.0,
            max_value=50.0,
            value=DEFAULT_SEARCH_WINDOW,
            step=1.0,
            help="Maximum depth shift to search for correlation."
        )
        
        max_elastic = st.slider(
            "Max Elastic Stretch (m)",
            min_value=1.0,
            max_value=20.0,
            value=DEFAULT_DTW_WINDOW,
            step=0.5,
            help="Maximum local stretch/squeeze allowed by DTW."
        )
        
        grid_step = st.selectbox(
            "Grid Resolution",
            options=[0.0762, 0.1524, 0.3048],
            index=1,
            format_func=lambda x: f"{x:.4f}m ({x/0.3048:.2f}ft)",
            help="Common depth grid step for signal alignment."
        )
    
    st.markdown("---")
    
    st.markdown("""
    ### 📖 How It Works
    
    **Automatic Pipeline:**
    
    1. **Well Detection**
       - Reads WELL header
       - Groups files by well name
    
    2. **Unit Detection**
       - Reads STRT.FT / STRT.M
       - Converts all to Meters
    
    3. **Duplicate Detection**
       - Fingerprints files
       - Skips identical duplicates
    
    4. **Chain Splicing**
       - Gap > 5m: Append
       - Overlap: Correlate + DTW
    """)


# =============================================================================
# MAIN INTERFACE - STEP 1: UPLOAD & GROUP
# =============================================================================

st.markdown("## Step 1: Upload All Logging Runs")
st.markdown("*Drop all your LAS files at once. Files from different wells will be automatically grouped.*")

# File uploader
uploaded_files = st.file_uploader(
    "Upload All Logging Runs",
    type=['las', 'LAS'],
    accept_multiple_files=True,
    help="Upload 2 or more LAS files from different logging runs. "
         "They can be from different wells - the system will group them automatically.",
    label_visibility="collapsed"
)

if uploaded_files:
    if len(uploaded_files) < 2:
        st.warning("⚠️ Please upload at least 2 LAS files for splicing.")
    else:
        # =================================================================
        # SCAN & GROUP FILES BY WELL
        # =================================================================
        
        # Only re-process if files changed
        file_names_hash = hash(tuple(f.name for f in uploaded_files))
        
        if ('well_groups_hash' not in st.session_state or 
            st.session_state.get('well_groups_hash') != file_names_hash):
            
            with st.spinner("🔍 Scanning files and detecting wells..."):
                progress_messages = []
                
                def capture_progress(step, msg):
                    progress_messages.append((step, msg))
                
                group_result = group_files_by_well(
                    uploaded_files,
                    progress_callback=capture_progress
                )
                
                st.session_state['well_groups'] = group_result.well_groups
                st.session_state['duplicate_warnings'] = group_result.duplicate_warnings
                st.session_state['num_wells'] = group_result.num_wells
                st.session_state['num_files_total'] = group_result.num_files_total
                st.session_state['well_groups_hash'] = file_names_hash
                
                # Clear any previous results
                if 'splice_result' in st.session_state:
                    del st.session_state['splice_result']
        
        well_groups = st.session_state['well_groups']
        duplicate_warnings = st.session_state['duplicate_warnings']
        num_wells = st.session_state['num_wells']
        num_files_total = st.session_state['num_files_total']
        
        # Display grouping summary
        st.markdown("---")
        
        if num_wells == 0:
            st.error("❌ No valid LAS files could be processed.")
            st.stop()
        
        # Show duplicate warnings if any
        if duplicate_warnings:
            st.markdown("### ⚠️ Duplicate Files Detected")
            for warning in duplicate_warnings:
                st.markdown(f"""
                <div class="duplicate-warning">
                    {warning}
                </div>
                """, unsafe_allow_html=True)
        
        # Summary metrics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{num_wells}</div>
                <div class="metric-label">Unique Well{'s' if num_wells > 1 else ''} Detected</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{num_files_total}</div>
                <div class="metric-label">Valid Files to Process</div>
            </div>
            """, unsafe_allow_html=True)
        
        # =================================================================
        # STEP 2: WELL SELECTION
        # =================================================================
        
        st.markdown("---")
        st.markdown("## Step 2: Select Target Well")
        
        well_names = list(well_groups.keys())
        
        # Auto-select if only one well, otherwise show selector
        if num_wells == 1:
            selected_well = well_names[0]
            st.info(f"📍 **Single well detected:** {selected_well} - automatically selected")
        else:
            st.markdown(f"*Detected **{num_wells} unique wells** in your upload. Select which one to analyze:*")
            
            selected_well = st.selectbox(
                "Select Target Well",
                options=well_names,
                format_func=lambda x: f"{x} ({len(well_groups[x])} files)",
                help="Choose the well you want to splice. Only files belonging to this well will be processed."
            )
        
        # Display selected well metadata
        if selected_well:
            selected_files = well_groups[selected_well]
            
            # Calculate metadata
            depth_min = min(f.start_depth for f in selected_files)
            depth_max = max(f.stop_depth for f in selected_files)
            total_depth = depth_max - depth_min
            
            # Get location from first file if available
            location = selected_files[0].location if selected_files[0].location else "Not specified"
            
            st.markdown(f"""
            <div class="well-selector-card">
                <h3 style="color: #22d3ee; margin: 0 0 1rem 0;">📊 {selected_well}</h3>
                <div class="well-info-grid">
                    <div class="well-stat">
                        <div class="well-stat-value">{len(selected_files)}</div>
                        <div class="well-stat-label">Files</div>
                    </div>
                    <div class="well-stat">
                        <div class="well-stat-value">{depth_min:.0f}m - {depth_max:.0f}m</div>
                        <div class="well-stat-label">Depth Range</div>
                    </div>
                    <div class="well-stat">
                        <div class="well-stat-value">{location[:20]}{'...' if len(location) > 20 else ''}</div>
                        <div class="well-stat-label">Location</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show files for this well
            with st.expander(f"📁 Files for {selected_well}", expanded=False):
                st.dataframe(
                    create_well_files_dataframe(selected_files),
                    use_container_width=True,
                    hide_index=True
                )
            
            # Store selected well
            st.session_state['selected_well'] = selected_well
            
            # =================================================================
            # STEP 3: EXECUTION
            # =================================================================
            
            st.markdown("---")
            st.markdown("## Step 3: Run Splicing Pipeline")
            
            # Execution button
            if st.button(f"⚡ Analyze '{selected_well}'", type="primary", use_container_width=True):
                
                with st.status("🔄 Processing files...", expanded=True) as status:
                    
                    # Step 3a: Show normalization info
                    st.markdown("""
                    <div class="pipeline-step">
                        <h4>Stage 1: Unit Normalization</h4>
                        <p>Converting all files to meters, stripping null padding...</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Files are already preprocessed in well_groups
                    preprocessed = selected_files
                    
                    st.markdown(f"✅ **{len(preprocessed)} files normalized to meters**")
                    
                    # Step 3b: Find correlation curve
                    st.markdown("""
                    <div class="pipeline-step">
                        <h4>Stage 2: Correlation Analysis</h4>
                        <p>Finding common curves and running cross-correlation...</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Find common curves across all files for this well
                    all_curves = [set(p.curves) for p in preprocessed]
                    common_curves = set.intersection(*all_curves) if all_curves else set()
                    common_curves = list(common_curves)
                    
                    if not common_curves:
                        st.warning("⚠️ No common curves found across all files. Using first available curve.")
                        common_curves = preprocessed[0].curves
                    
                    correlation_curve = get_recommended_correlation_curve(common_curves)
                    
                    if not correlation_curve:
                        correlation_curve = common_curves[0] if common_curves else None
                    
                    if not correlation_curve:
                        st.error("❌ No curves available for correlation.")
                        status.update(label="❌ Error: No correlation curve", state="error")
                        st.stop()
                    
                    st.info(f"📊 Using **{correlation_curve}** for correlation alignment")
                    
                    # Step 3c: DTW Alignment
                    st.markdown("""
                    <div class="pipeline-step">
                        <h4>Stage 3: DTW Elastic Warping</h4>
                        <p>Applying constrained Dynamic Time Warping for fine alignment...</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Progress bar for splicing
                    splice_progress = st.progress(0, text="Initializing splice pipeline...")
                    
                    # Run batch splicing
                    splice_messages = []
                    
                    def splice_progress_callback(step, msg):
                        splice_messages.append(msg)
                        progress_pct = min(0.95, len(splice_messages) / (len(preprocessed) * 3))
                        splice_progress.progress(progress_pct, text=msg)
                    
                    try:
                        result = batch_splice_pipeline(
                            preprocessed_files=preprocessed,
                            correlation_curve=correlation_curve,
                            grid_step=grid_step,
                            max_search_meters=max_search,
                            max_elastic_meters=max_elastic,
                            progress_callback=splice_progress_callback
                        )
                        
                        splice_progress.progress(1.0, text="Splicing complete!")
                        
                        # Display splice operations
                        st.markdown("**Splice Operations:**")
                        for log_entry in result.splice_log:
                            if "Gap" in log_entry:
                                st.write(f"🔗 {log_entry}")
                            elif "overlap" in log_entry.lower():
                                st.write(f"🧬 {log_entry}")
                            else:
                                st.write(f"📝 {log_entry}")
                        
                        status.update(label="✅ Auto-Splicing Complete!", state="complete")
                        
                        # Store results in session state
                        st.session_state['splice_result'] = result
                        st.session_state['correlation_curve'] = correlation_curve
                        st.session_state['processed_well'] = selected_well
                        
                    except Exception as e:
                        status.update(label="❌ Error during processing", state="error")
                        st.error(f"Error: {str(e)}")
                        st.exception(e)
                        st.stop()
            
            # =================================================================
            # RESULTS DISPLAY (if available)
            # =================================================================
            
            if 'splice_result' in st.session_state and st.session_state.get('processed_well') == selected_well:
                result = st.session_state['splice_result']
                correlation_curve = st.session_state['correlation_curve']
                
                st.markdown("---")
                
                # Success banner
                st.markdown(f"""
                <div class="success-banner">
                    <h3>✅ Composite Log Created Successfully</h3>
                    <p><strong>{selected_well}</strong>: {result.num_files_processed} files merged into a single continuous log spanning 
                    {result.total_depth_range[0]:.1f}m to {result.total_depth_range[1]:.1f}m</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{result.num_files_processed}</div>
                        <div class="metric-label">Files Merged</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    total_depth = result.total_depth_range[1] - result.total_depth_range[0]
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{total_depth:.0f}m</div>
                        <div class="metric-label">Total Depth</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(result.composite_df)}</div>
                        <div class="metric-label">Data Points</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    num_curves = len([c for c in result.composite_df.columns if c != 'DEPTH'])
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{num_curves}</div>
                        <div class="metric-label">Curves</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # File summary table
                st.markdown("### 📋 Processing Summary")
                st.dataframe(
                    create_file_summary_dataframe(result.file_summary),
                    use_container_width=True,
                    hide_index=True
                )
                
                st.markdown("---")
                
                # Composite plot
                st.markdown("### 📊 Composite Log Visualization")
                
                fig = create_composite_plot(
                    result.composite_df,
                    correlation_curve,
                    result.file_summary,
                    result.splice_log
                )
                
                st.pyplot(fig)
                
                st.markdown("---")
                
                # Export options
                st.markdown("### 💾 Export Composite Log")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Export as LAS
                    header_info = {
                        'WELL': f'{selected_well}_COMPOSITE',
                        'STRT': result.total_depth_range[0],
                        'STOP': result.total_depth_range[1],
                        'STEP': np.median(np.diff(result.composite_df['DEPTH'].values))
                    }
                    
                    las_data = export_to_las(result.composite_df, header_info, 'm')
                    st.download_button(
                        label="📋 Download Composite LAS",
                        data=las_data,
                        file_name=f"{selected_well.replace(' ', '_')}_composite.las",
                        mime="text/plain",
                        use_container_width=True,
                        type="primary"
                    )
                
                with col2:
                    # Export plot as PNG
                    png_data = export_plot_to_bytes(fig, format='png', dpi=150)
                    st.download_button(
                        label="📷 Download Plot (PNG)",
                        data=png_data,
                        file_name=f"{selected_well.replace(' ', '_')}_composite.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                with col3:
                    # Export as CSV
                    csv_data = result.composite_df.to_csv(index=False)
                    st.download_button(
                        label="📊 Download CSV",
                        data=csv_data,
                        file_name=f"{selected_well.replace(' ', '_')}_composite.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                plt.close(fig)

else:
    # Welcome screen when no files uploaded
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
                border-radius: 16px; padding: 4rem 2rem; text-align: center; 
                border: 2px dashed #475569; margin-top: 2rem;">
        <div style="font-size: 5rem; margin-bottom: 1.5rem;">⚡</div>
        <h2 style="color: #22d3ee; margin-bottom: 1rem; font-size: 1.8rem;">
            Multi-Well Batch Auto-Splicing
        </h2>
        <p style="color: #94a3b8; font-size: 1.1rem; max-width: 600px; margin: 0 auto 2rem auto; line-height: 1.6;">
            Upload LAS files from <strong>any number of wells</strong>. The system automatically 
            groups them by well name, lets you select which well to analyze, and handles 
            unit conversion and intelligent splicing.
        </p>
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; margin-top: 2rem;">
            <div style="text-align: left; color: #e2e8f0;">
                <p style="margin: 0.5rem 0;">✓ Auto-detect wells by name</p>
                <p style="margin: 0.5rem 0;">✓ Skip duplicate files</p>
                <p style="margin: 0.5rem 0;">✓ Convert Feet to Meters</p>
            </div>
            <div style="text-align: left; color: #e2e8f0;">
                <p style="margin: 0.5rem 0;">✓ Handle gaps and overlaps</p>
                <p style="margin: 0.5rem 0;">✓ Correlation + DTW alignment</p>
                <p style="margin: 0.5rem 0;">✓ Export composite LAS</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # How it works section
    st.markdown("---")
    st.markdown("### 🔬 The Algorithm Pipeline")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: #1e293b; padding: 1.5rem; border-radius: 12px; height: 100%;">
            <h4 style="color: #22d3ee; margin-bottom: 1rem;">1️⃣ Scan & Group</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">
                Each file is scanned for the WELL header field. Files are grouped 
                by well name (sanitized to handle variations like "Griffyn #2" vs " griffyn #2 ").
                Duplicate files are automatically detected and skipped.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #1e293b; padding: 1.5rem; border-radius: 12px; height: 100%;">
            <h4 style="color: #22d3ee; margin-bottom: 1rem;">2️⃣ Select Well</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">
                If multiple wells are detected, you choose which one to analyze.
                The system shows metadata for each well: file count, depth range, 
                and location. Single-well uploads are auto-selected.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: #1e293b; padding: 1.5rem; border-radius: 12px; height: 100%;">
            <h4 style="color: #22d3ee; margin-bottom: 1rem;">3️⃣ Splice & Export</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">
                Only the selected well's files are processed through the 
                chain-splice algorithm (correlation + DTW). Export the 
                composite log as LAS, PNG, or CSV.
            </p>
        </div>
        """, unsafe_allow_html=True)
