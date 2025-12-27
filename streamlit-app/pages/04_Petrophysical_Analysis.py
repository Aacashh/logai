"""
WellLog Analyzer Pro - Petrophysical Analysis Page

Comprehensive petrophysical analysis workflow including:
1. Automated Outlier Detection/Despiking (Isolation Forest, ABOD)
2. Tool Startup Noise Removal (Rolling variance + slope check)
3. Log Splicing/Concatenation (Cross-correlation, DTW)
4. Depth Alignment (Correlation-based, Siamese Neural Networks)

All features include toggleable vertical log visualizations.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.las_parser import load_las, extract_header_info, detect_depth_units, get_available_curves
from shared.curve_mapping import get_curve_mapping
from shared.data_processing import process_data, export_to_las
from shared.splicing import group_files_by_well, WellGroupResult

# Page configuration
st.set_page_config(
    page_title="Petrophysical Analysis | WellLog Analyzer Pro",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #334155;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
    }
    
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
        margin: 0;
    }
    
    .main-subtitle {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 0.3rem;
    }
    
    .feature-header {
        background: linear-gradient(90deg, #38bdf8 0%, #0ea5e9 100%);
        color: #0f172a;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 25px 0 15px 0;
    }
    
    .feature-card {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #334155;
        margin-bottom: 15px;
    }
    
    .feature-title {
        color: #38bdf8;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
    
    .feature-desc {
        color: #94a3b8;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #334155;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 5px;
    }
    
    .algorithm-box {
        background: #0f172a;
        border-left: 4px solid #38bdf8;
        padding: 15px 20px;
        border-radius: 0 8px 8px 0;
        margin: 15px 0;
        font-family: 'Inter', sans-serif;
    }
    
    .algorithm-box code {
        font-family: 'JetBrains Mono', monospace;
        background: #1e293b;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 0.85rem;
        color: #38bdf8;
    }
    
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .badge-success { background: #22c55e; color: #0f172a; }
    .badge-warning { background: #f59e0b; color: #0f172a; }
    .badge-error { background: #ef4444; color: white; }
    .badge-info { background: #38bdf8; color: #0f172a; }
    
    .info-panel {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .results-panel {
        background: linear-gradient(180deg, #1e3a5f 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 20px;
        margin-top: 15px;
    }
    
    .step-indicator {
        background: #38bdf8;
        color: #0f172a;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 10px;
    }
    
    .track-toggle {
        background: #1e293b;
        padding: 10px 15px;
        border-radius: 8px;
        margin: 5px 0;
        border: 1px solid #334155;
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
        color: #38bdf8;
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
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <div class="main-title">🔬 Petrophysical Analysis</div>
    <div class="main-subtitle">Comprehensive Well Log Processing • Outlier Detection • Noise Removal • Log Splicing • Depth Alignment</div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTION - SCAN FOLDER FOR LAS FILES AND EXTRACT FIELD/WELL INFO
# =============================================================================

@st.cache_data(show_spinner="Scanning folder for LAS files...")
def scan_folder_for_las_files(folder_path, max_files=100):
    """
    Scan folder for LAS files and extract Field and Well information.
    Returns dict with {filepath: {'FIELD': ..., 'WELL': ...}}
    
    Args:
        folder_path: Path to folder containing LAS files
        max_files: Maximum number of files to scan (to prevent long waits)
    """
    import glob
    las_files_info = {}
    fields = set()
    wells_by_field = {}
    
    # Find all LAS files in folder
    las_pattern = os.path.join(folder_path, "*.las")
    las_files = glob.glob(las_pattern, recursive=False)
    # Also check for .LAS extension
    las_pattern_upper = os.path.join(folder_path, "*.LAS")
    las_files.extend(glob.glob(las_pattern_upper, recursive=False))
    
    # Limit number of files to scan
    if len(las_files) > max_files:
        las_files = las_files[:max_files]
    
    for filepath in las_files:
        try:
            las = load_las(filepath)
            header = extract_header_info(las)
            field = header.get('FIELD', 'Unknown') or 'Unknown'
            well = header.get('WELL', 'Unknown') or 'Unknown'
            
            las_files_info[filepath] = {
                'FIELD': field,
                'WELL': well,
                'FILENAME': os.path.basename(filepath)
            }
            
            fields.add(field)
            if field not in wells_by_field:
                wells_by_field[field] = set()
            wells_by_field[field].add((well, filepath))
        except Exception as e:
            continue
    
    return {
        'files': las_files_info,
        'fields': sorted(list(fields)),
        'wells_by_field': {k: sorted(list(v)) for k, v in wells_by_field.items()}
    }


# =============================================================================
# SIDEBAR CONFIGURATION
# =============================================================================

with st.sidebar:
    st.markdown("### 📁 Data Source")
    
    # Data source mode selection
    upload_mode = st.radio(
        "Data Source",
        ["Select Folder", "Upload File(s)"],
        help="Select folder to browse by Field/Well, or upload files directly"
    )
    
    # Initialize variables
    primary_file = None
    secondary_file = None
    multi_files = None
    selected_las_path = None
    folder_scan_result = None
    
    if upload_mode == "Select Folder":
        st.markdown("#### 📂 Folder Selection")
        
        # Default folder path
        default_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '2025')
        
        folder_path = st.text_input(
            "Folder Path",
            value=default_folder,
            help="Enter the path to folder containing LAS files"
        )
        
        if folder_path and os.path.isdir(folder_path):
            # Scan folder for LAS files
            folder_scan_result = scan_folder_for_las_files(folder_path)
            
            if folder_scan_result['fields']:
                st.success(f"✅ Found {len(folder_scan_result['files'])} LAS files")
                
                # Field dropdown
                selected_field = st.selectbox(
                    "🏭 FIELD",
                    options=folder_scan_result['fields'],
                    help="Select a field to filter wells"
                )
                
                # Well dropdown (filtered by selected field)
                if selected_field and selected_field in folder_scan_result['wells_by_field']:
                    wells_in_field = folder_scan_result['wells_by_field'][selected_field]
                    well_options = [(w[0], w[1]) for w in wells_in_field]  # (well_name, filepath)
                    well_names = [w[0] for w in well_options]
                    
                    selected_well_idx = st.selectbox(
                        "🛢️ WELL",
                        options=range(len(well_names)),
                        format_func=lambda x: well_names[x],
                        help="Select a well to view logs"
                    )
                    
                    if selected_well_idx is not None:
                        selected_las_path = well_options[selected_well_idx][1]
                        st.info(f"📄 **File:** {os.path.basename(selected_las_path)}")
            else:
                st.warning("⚠️ No LAS files found in folder")
        else:
            if folder_path:
                st.error("❌ Invalid folder path")
    
    else:
        # Original upload mode
        upload_type = st.radio(
            "Upload Type",
            ["Single LAS File", "Multiple LAS Files"],
            help="Single file for analysis, multiple for splicing"
        )
        
        if upload_type == "Single LAS File":
            primary_file = st.file_uploader(
                "Upload LAS File",
                type=['las', 'LAS'],
                key='petro_primary',
                help="Upload a well log file for analysis"
            )
        else:
            multi_files = st.file_uploader(
                "Upload Multiple LAS Files",
                type=['las', 'LAS'],
                key='petro_multi',
                accept_multiple_files=True,
                help="Upload multiple files for splicing or alignment"
            )
    
    st.markdown("---")
    
    st.markdown("### 🔧 Track Visibility")
    
    track_visibility = {}
    track_options = ['GR', 'CALIPER', 'RES_DEEP', 'RES_MED', 'DENS', 'NEUT', 'SONIC']
    
    for track in track_options:
        default_value = track in ['GR', 'RES_DEEP', 'DENS']
        track_visibility[track] = st.checkbox(track, value=default_value, key=f"vis_{track}")
    
    st.markdown("---")
    
    st.markdown("### 📐 Display Settings")
    
    scale_option = st.selectbox(
        "Depth Scale",
        ["1:200", "1:500", "1:1000", "1:2000"],
        index=1
    )
    scale_ratio = int(scale_option.split(":")[1])
    
    st.markdown("---")
    
    st.markdown("""
    ### 📚 About This Module
    
    **Processing Features:**
    - 🔍 Outlier Detection (IF, ABOD)
    - 🔧 Noise Removal
    - 🔗 Log Splicing
    - 📐 Depth Alignment (ML/NN)
    
    ---
    
    *Real-time processing with interactive visualizations*
    """)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_available_feature_columns(df, mapping):
    """Get list of available numeric curves for analysis."""
    feature_cols = []
    for curve_type in ['GR', 'RES_DEEP', 'RES_MED', 'RES_SHAL', 'DENS', 'NEUT', 'SONIC', 'CALIPER']:
        if mapping.get(curve_type) and mapping[curve_type] in df.columns:
            feature_cols.append(mapping[curve_type])
    return feature_cols


def display_file_info(header, df, unit, col):
    """Display file information in a styled panel."""
    col.markdown(f"""
    <div class="info-panel">
        <strong style="color: #38bdf8;">Well:</strong> {header.get('WELL', 'Unknown')}<br>
        <span style="color: #94a3b8;">Depth: {df['DEPTH'].min():.1f} - {df['DEPTH'].max():.1f} {unit}</span><br>
        <span style="color: #94a3b8;">Samples: {len(df):,}</span><br>
        <span style="color: #94a3b8;">Curves: {len(df.columns) - 1}</span>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN CONTENT - SINGLE FILE MODE (from folder selection or file upload)
# =============================================================================

# Determine if we have a file to process (either from folder selection or upload)
has_single_file = (upload_mode == "Select Folder" and selected_las_path) or \
                  (upload_mode == "Upload File(s)" and primary_file)

if has_single_file:
    try:
        # Load LAS file from either source
        if upload_mode == "Select Folder" and selected_las_path:
            las = load_las(selected_las_path)
        else:
            las = load_las(primary_file)
        
        unit = detect_depth_units(las)
        mapping = get_curve_mapping(las)
        df = process_data(las, mapping)
        header = extract_header_info(las)
        
        # Display file info
        st.markdown("### 📥 Loaded Data")
        col1, col2, col3 = st.columns(3)
        display_file_info(header, df, unit, col1)
        
        with col2:
            st.markdown(f"""
            <div class="info-panel">
                <strong style="color: #38bdf8;">Available Curves:</strong><br>
                <span style="color: #94a3b8;">{', '.join([c for c in df.columns if c != 'DEPTH'][:5])}</span>
                {f'<br><span style="color: #64748b;">+{len(df.columns) - 6} more</span>' if len(df.columns) > 6 else ''}
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="info-panel">
                <strong style="color: #38bdf8;">Depth Unit:</strong> {unit}<br>
                <strong style="color: #38bdf8;">Depth Step:</strong> {np.median(np.diff(df['DEPTH'].values)):.4f} {unit}
            </div>
            """, unsafe_allow_html=True)
        
        # Get available features
        feature_columns = get_available_feature_columns(df, mapping)
        
        # Add professional log preview following industry standards
        with st.expander("📊 Professional Log Display (Industry Standard)", expanded=True):
            from plotting import create_professional_log_display
            
            st.markdown("""
            <div style="padding: 0.5rem; background: #f8fafc; border-radius: 4px; margin-bottom: 1rem;">
                <small style="color: #64748b;">
                    📋 <strong>Track Layout:</strong> GR (Gamma Ray) | Resistivity (Log Scale) | Density-Neutron Overlay
                </small>
            </div>
            """, unsafe_allow_html=True)
            
            # Prepare header info for display
            header_display = {
                'WELL': header.get('WELL', 'Unknown'),
                'FIELD': header.get('FIELD', header.get('FLD', '')),
                'LOC': header.get('LOC', header.get('LOCATION', '')),
                'STRT': df['DEPTH'].min(),
                'STOP': df['DEPTH'].max(),
                'STEP': np.median(np.diff(df['DEPTH'].values)) if len(df) > 1 else 0
            }
            
            preview_fig = create_professional_log_display(
                df, mapping,
                header_info=header_display,
                settings={'scale_ratio': scale_ratio, 'depth_unit': unit},
                show_gr_fill=True,
                show_dn_crossover=True
            )
            st.pyplot(preview_fig)
            plt.close(preview_fig)
        
        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔍 Outlier Detection",
            "🔧 Noise Removal",
            "🔗 Log Splicing",
            "📐 Depth Alignment"
        ])
        
        # =================================================================
        # TAB 1: OUTLIER DETECTION
        # =================================================================
        with tab1:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-title">🔍 Automated Outlier Detection & Despiking</div>
                <div class="feature-desc">
                    Detect anomalous data points using multiple ML algorithms: Isolation Forest, 
                    Local Outlier Factor (LOF), and Angular-Based Outlier Detection (ABOD).
                    ABOD is particularly effective for high-dimensional data and angular relationships.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Import outlier detection
            from ml_components.outlier_detection import (
                detect_outliers_isolation_forest,
                detect_outliers_lof,
                detect_outliers_abod,
                detect_outliers_ensemble,
                clean_outliers,
                PYOD_AVAILABLE
            )
            from plotting import create_professional_log_display, create_before_after_comparison
            
            # Controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                methods = ['Isolation Forest', 'Local Outlier Factor']
                if PYOD_AVAILABLE:
                    methods.extend(['ABOD (Angular-Based)', 'Ensemble (All Methods)'])
                else:
                    methods.append('Ensemble (IF + LOF)')
                
                outlier_method = st.selectbox("Detection Method", options=methods)
            
            with col2:
                contamination = st.slider(
                    "Contamination (%)",
                    min_value=1, max_value=20, value=5,
                    help="Expected percentage of outliers"
                ) / 100
            
            with col3:
                selected_features = st.multiselect(
                    "Curves to Analyze",
                    options=feature_columns,
                    default=feature_columns[:min(3, len(feature_columns))],
                    help="Select curves for outlier detection"
                )
            
            # Algorithm explanation
            with st.expander("📐 How It Works", expanded=False):
                if 'ABOD' in outlier_method:
                    st.markdown("""
                    <div class="algorithm-box">
                    <strong>Angular-Based Outlier Detection (ABOD):</strong><br><br>
                    ABOD analyzes the <strong>variance of angles</strong> between a point and all pairs of other points.<br><br>
                    <strong>Key Insight:</strong> Points inside a cluster have high angle variance (angles span wide range),
                    while outliers have low angle variance (they're viewed from a consistent direction).<br><br>
                    <strong>Advantages:</strong><br>
                    • Less affected by the "curse of dimensionality"<br>
                    • Works well when outliers are in different angular regions<br>
                    • Robust to varying densities in the data
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="algorithm-box">
                    <strong>Isolation Forest:</strong> Isolates anomalies by random partitioning. 
                    Anomalies are "few and different" → easier to isolate.<br><br>
                    <strong>LOF:</strong> Measures local density deviation. Points with substantially 
                    lower density than neighbors are outliers.
                    </div>
                    """, unsafe_allow_html=True)
            
            if selected_features and st.button("🚀 Run Outlier Detection", type="primary", key="run_outlier"):
                with st.spinner("Detecting outliers..."):
                    try:
                        # Run detection
                        if outlier_method == 'Isolation Forest':
                            result = detect_outliers_isolation_forest(df, selected_features, contamination)
                        elif outlier_method == 'Local Outlier Factor':
                            result = detect_outliers_lof(df, selected_features, contamination=contamination)
                        elif 'ABOD' in outlier_method:
                            result = detect_outliers_abod(df, selected_features, contamination=contamination)
                        else:
                            include_abod = PYOD_AVAILABLE and 'All' in outlier_method
                            result = detect_outliers_ensemble(df, selected_features, contamination, include_abod=include_abod)
                        
                        # Store result in session state
                        st.session_state['outlier_result'] = result
                        st.session_state['outlier_df'] = df
                        
                        # Display results
                        st.markdown('<div class="feature-header">📊 Detection Results</div>', unsafe_allow_html=True)
                        
                        # Metrics
                        m1, m2, m3, m4 = st.columns(4)
                        
                        with m1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{result.num_anomalies}</div>
                                <div class="metric-label">Outliers Detected</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with m2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{result.contamination_actual*100:.1f}%</div>
                                <div class="metric-label">Actual Contamination</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with m3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{result.confidence*100:.0f}%</div>
                                <div class="metric-label">Detection Confidence</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with m4:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{result.method.upper()}</div>
                                <div class="metric-label">Method Used</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Outlier Visualization
                        if result.num_anomalies > 0:
                            st.markdown("### 🎯 Detected Outliers - Per Curve Visualization")
                            st.markdown("""
                            <div style="padding: 0.5rem; background: #fef2f2; border-radius: 4px; margin-bottom: 1rem; border-left: 4px solid #ef4444;">
                                <small style="color: #991b1b;">
                                    🔴 <strong>Red markers show detected outliers on each analyzed curve</strong> - 
                                    {0} anomalies detected ({1:.1f}% of data). Histogram shows depth distribution.
                                </small>
                            </div>
                            """.format(result.num_anomalies, result.contamination_actual*100), unsafe_allow_html=True)
                            
                            from plotting import create_outlier_visualization
                            
                            # Create per-curve outlier visualization
                            outlier_fig = create_outlier_visualization(
                                df,
                                selected_features,
                                result.anomaly_mask,
                                depth_col='DEPTH',
                                settings={'scale_ratio': scale_ratio},
                                title=f"Outlier Detection Results - {outlier_method}"
                            )
                            st.pyplot(outlier_fig)
                            plt.close(outlier_fig)
                            
                            # Outlier depth summary table
                            st.markdown("### 📋 Outlier Depth Summary")
                            outlier_depths = df.loc[result.anomaly_mask, 'DEPTH'].values
                            if len(outlier_depths) > 0:
                                depth_ranges = []
                                start_depth = outlier_depths[0]
                                prev_depth = outlier_depths[0]
                                
                                for d in outlier_depths[1:]:
                                    if d - prev_depth > np.median(np.diff(df['DEPTH'].values)) * 2:
                                        depth_ranges.append((start_depth, prev_depth))
                                        start_depth = d
                                    prev_depth = d
                                depth_ranges.append((start_depth, prev_depth))
                                
                                range_df = pd.DataFrame(depth_ranges, columns=['Start Depth', 'End Depth'])
                                range_df['Interval'] = range_df['End Depth'] - range_df['Start Depth']
                                range_df = range_df.round(2)
                                st.dataframe(range_df, use_container_width=True, hide_index=True)
                            
                            # Expandable: Professional multi-track display
                            with st.expander("📈 Professional Multi-Track Log Display", expanded=False):
                                from plotting import create_professional_log_display
                                
                                header_display = {
                                    'WELL': header.get('WELL', 'Unknown'),
                                    'FIELD': header.get('FIELD', header.get('FLD', '')),
                                    'STRT': df['DEPTH'].min(),
                                    'STOP': df['DEPTH'].max(),
                                    'STEP': np.median(np.diff(df['DEPTH'].values)) if len(df) > 1 else 0
                                }
                                
                                fig = create_professional_log_display(
                                    df, mapping,
                                    header_info=header_display,
                                    settings={'scale_ratio': scale_ratio, 'depth_unit': unit},
                                    show_gr_fill=True,
                                    show_dn_crossover=True,
                                    highlight_mask=result.anomaly_mask,
                                    highlight_color='#ef4444',
                                    highlight_alpha=0.3
                                )
                                st.pyplot(fig)
                                plt.close(fig)
                            
                            # Feature importance
                            if result.feature_importance:
                                st.markdown("### 🎯 Feature Importance")
                                importance_df = pd.DataFrame([
                                    {'Curve': k, 'Importance': v} 
                                    for k, v in result.feature_importance.items()
                                ]).sort_values('Importance', ascending=True)
                                
                                st.bar_chart(importance_df.set_index('Curve')['Importance'])
                            
                            # Cleaning options
                            st.markdown("### 🧹 Clean Outliers")
                            
                            clean_col1, clean_col2 = st.columns(2)
                            
                            with clean_col1:
                                clean_method = st.selectbox(
                                    "Cleaning Method",
                                    ['interpolate', 'median', 'remove'],
                                    help="How to handle detected outliers"
                                )
                            
                            with clean_col2:
                                if st.button("🧹 Apply Cleaning", type="secondary"):
                                    df_cleaned = clean_outliers(df, result, method=clean_method)
                                    st.session_state['cleaned_df'] = df_cleaned
                                    st.session_state['outlier_cleaned'] = True
                                    st.success(f"✅ Cleaned {result.num_anomalies} outliers using {clean_method} method")
                            
                            # Before/After comparison if cleaned
                            if st.session_state.get('outlier_cleaned') and 'cleaned_df' in st.session_state:
                                st.markdown("### 📊 Before / After Comparison (Industry Standard Display)")
                                
                                from plotting import create_before_after_comparison
                                
                                df_cleaned = st.session_state['cleaned_df']
                                
                                # Select a curve to compare
                                compare_curve = st.selectbox(
                                    "Select curve to compare",
                                    options=selected_features,
                                    key="outlier_compare_curve"
                                )
                                
                                # Prepare header info
                                header_display = {
                                    'WELL': header.get('WELL', 'Unknown'),
                                    'STRT': df['DEPTH'].min(),
                                    'STOP': df['DEPTH'].max()
                                }
                                
                                # Use professional before/after comparison
                                comparison_fig = create_before_after_comparison(
                                    df, df_cleaned, compare_curve,
                                    mapping=mapping,
                                    header_info=header_display,
                                    settings={'scale_ratio': scale_ratio, 'depth_unit': unit},
                                    highlight_mask=result.anomaly_mask,
                                    title_before="BEFORE (With Outliers)",
                                    title_after="AFTER (Cleaned)"
                                )
                                st.pyplot(comparison_fig)
                                plt.close(comparison_fig)
                        else:
                            st.info("No outliers detected with current settings.")
                    
                    except Exception as e:
                        st.error(f"Error during outlier detection: {str(e)}")
        
        # =================================================================
        # TAB 2: NOISE REMOVAL
        # =================================================================
        with tab2:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-title">🔧 Tool Startup Noise Removal</div>
                <div class="feature-desc">
                    Detect and remove tool startup noise, which appears as constant/flat values 
                    at the beginning of logging runs. Uses rolling variance and slope analysis 
                    to identify low-variation regions indicating tool warmup periods.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Import noise removal
            from ml_components.noise_removal import (
                detect_tool_startup_noise,
                detect_tool_shutdown_noise,
                remove_noise,
                detect_spike_noise,
                despike_signal,
                get_noise_quality_report
            )
            from plotting import create_professional_log_display, create_before_after_comparison
            
            # Controls
            st.markdown("### ⚙️ Detection Parameters")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                window_size = st.slider("Window Size", min_value=5, max_value=30, value=10)
            
            with col2:
                var_threshold = st.slider(
                    "Variance Threshold",
                    min_value=0.001, max_value=0.1, value=0.01, format="%.3f",
                    help="Maximum variance to be considered flat"
                )
            
            with col3:
                slope_threshold = st.slider(
                    "Slope Threshold",
                    min_value=0.0001, max_value=0.01, value=0.001, format="%.4f",
                    help="Maximum slope to be considered constant"
                )
            
            with col4:
                noise_curves = st.multiselect(
                    "Curves to Analyze",
                    options=feature_columns,
                    default=feature_columns[:min(2, len(feature_columns))]
                )
            
            # Detection mode
            detection_mode = st.radio(
                "Detection Mode",
                ["Tool Startup (Top of Log)", "Tool Shutdown (Bottom of Log)", "Both"],
                horizontal=True
            )
            
            # Algorithm explanation
            with st.expander("📐 How It Works", expanded=False):
                st.markdown("""
                <div class="algorithm-box">
                <strong>Tool Startup Noise Detection:</strong><br><br>
                1. <strong>Rolling Variance:</strong> Calculate variance over a sliding window. 
                   Low variance indicates flat/constant regions.<br><br>
                2. <strong>Rolling Slope:</strong> Calculate local slope using linear regression. 
                   Near-zero slope indicates no change over depth.<br><br>
                3. <strong>Combined Detection:</strong> Regions with both low variance AND low slope 
                   are flagged as tool startup noise.<br><br>
                <strong>Formula:</strong><br>
                <code>noise = (rolling_var < threshold) AND (|slope| < slope_threshold)</code>
                </div>
                """, unsafe_allow_html=True)
            
            if noise_curves and st.button("🔍 Detect Noise", type="primary", key="detect_noise"):
                with st.spinner("Analyzing noise patterns..."):
                    try:
                        results = []
                        
                        if detection_mode in ["Tool Startup (Top of Log)", "Both"]:
                            startup_result = detect_tool_startup_noise(
                                df, noise_curves, 'DEPTH',
                                window=window_size,
                                variance_threshold=var_threshold,
                                slope_threshold=slope_threshold
                            )
                            results.append(('Startup', startup_result))
                        
                        if detection_mode in ["Tool Shutdown (Bottom of Log)", "Both"]:
                            shutdown_result = detect_tool_shutdown_noise(
                                df, noise_curves, 'DEPTH',
                                window=window_size,
                                variance_threshold=var_threshold,
                                slope_threshold=slope_threshold
                            )
                            results.append(('Shutdown', shutdown_result))
                        
                        # Display results for each detection
                        for name, result in results:
                            st.markdown(f'<div class="feature-header">📊 {name} Noise Detection Results</div>', unsafe_allow_html=True)
                            
                            report = get_noise_quality_report(result, df, 'DEPTH')
                            
                            m1, m2, m3, m4 = st.columns(4)
                            
                            with m1:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value">{result.noise_samples}</div>
                                    <div class="metric-label">Noisy Samples</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with m2:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value">{result.noise_percentage:.1f}%</div>
                                    <div class="metric-label">Data Affected</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with m3:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value">{result.noise_depth_start:.1f}</div>
                                    <div class="metric-label">Start Depth ({unit})</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with m4:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value">{result.noise_depth_end:.1f}</div>
                                    <div class="metric-label">End Depth ({unit})</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Store result
                            st.session_state[f'{name.lower()}_noise_result'] = result
                        
                        # Noise Detection Visualization
                        if results:
                            # Combined mask for visualization
                            combined_mask = np.zeros(len(df), dtype=bool)
                            total_noise_points = 0
                            for _, result in results:
                                combined_mask |= result.noise_mask
                                total_noise_points += np.sum(result.noise_mask)
                            
                            st.markdown("### 🔧 Detected Noise - Per Curve Visualization")
                            st.markdown("""
                            <div style="padding: 0.5rem; background: #fff7ed; border-radius: 4px; margin-bottom: 1rem; border-left: 4px solid #ff8c00;">
                                <small style="color: #9a3412;">
                                    🟠 <strong>Orange highlighted regions show detected noise on each analyzed curve</strong> - 
                                    {0} points flagged ({1:.1f}% of data). Dashed lines mark noise boundaries.
                                </small>
                            </div>
                            """.format(total_noise_points, total_noise_points/len(df)*100), unsafe_allow_html=True)
                            
                            from plotting import create_noise_visualization
                            
                            # Get variance and slope data from first result
                            variance_data = results[0][1].rolling_variance if hasattr(results[0][1], 'rolling_variance') else None
                            slope_data = results[0][1].rolling_slope if hasattr(results[0][1], 'rolling_slope') else None
                            
                            # Create per-curve noise visualization
                            noise_fig = create_noise_visualization(
                                df,
                                noise_curves,
                                combined_mask,
                                variance_data=variance_data,
                                slope_data=slope_data,
                                depth_col='DEPTH',
                                settings={'scale_ratio': scale_ratio},
                                title=f"Noise Detection Results - {detection_mode}",
                                noise_type=detection_mode
                            )
                            st.pyplot(noise_fig)
                            plt.close(noise_fig)
                            
                            # Noise depth summary
                            st.markdown("### 📋 Noise Zone Summary")
                            noise_depths = df.loc[combined_mask, 'DEPTH'].values
                            if len(noise_depths) > 0:
                                noise_summary = {
                                    'Metric': ['Start Depth', 'End Depth', 'Interval', 'Samples Affected', 'Percentage'],
                                    'Value': [
                                        f"{np.min(noise_depths):.2f} {unit}",
                                        f"{np.max(noise_depths):.2f} {unit}",
                                        f"{np.max(noise_depths) - np.min(noise_depths):.2f} {unit}",
                                        f"{total_noise_points}",
                                        f"{total_noise_points/len(df)*100:.2f}%"
                                    ]
                                }
                                st.dataframe(pd.DataFrame(noise_summary), use_container_width=True, hide_index=True)
                            
                            # Expandable: Detection metrics plots
                            with st.expander("📉 Detection Metrics (Variance & Slope)", expanded=False):
                                fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                                
                                depth = df['DEPTH'].values
                                
                                # Variance plot with noise region highlighted
                                if variance_data is not None:
                                    ax1.plot(variance_data, depth, 'b-', linewidth=0.8, label='Rolling Variance')
                                    # Highlight noise portion
                                    noisy_var = np.where(combined_mask, variance_data, np.nan)
                                    ax1.plot(noisy_var, depth, color='#ff8c00', linewidth=2, label='Noise Region')
                                ax1.axvline(x=var_threshold, color='r', linestyle='--', linewidth=1.5, label=f'Threshold={var_threshold}')
                                ax1.set_xlabel('Rolling Variance')
                                ax1.set_ylabel('Depth (m)')
                                ax1.set_ylim(depth.max(), depth.min())
                                ax1.set_title('Rolling Variance Profile')
                                ax1.legend(fontsize=8)
                                ax1.grid(True, alpha=0.3)
                                ax1.set_facecolor('#f8f9fa')
                                
                                # Slope plot with noise region highlighted
                                if slope_data is not None:
                                    ax2.plot(slope_data, depth, 'g-', linewidth=0.8, label='Rolling Slope')
                                    noisy_slope = np.where(combined_mask, slope_data, np.nan)
                                    ax2.plot(noisy_slope, depth, color='#ff8c00', linewidth=2, label='Noise Region')
                                ax2.axvline(x=slope_threshold, color='r', linestyle='--', linewidth=1.5, label=f'Threshold={slope_threshold}')
                                ax2.set_xlabel('Rolling Slope')
                                ax2.set_ylim(depth.max(), depth.min())
                                ax2.set_title('Rolling Slope Profile')
                                ax2.legend(fontsize=8)
                                ax2.grid(True, alpha=0.3)
                                ax2.set_facecolor('#f8f9fa')
                                
                                plt.tight_layout()
                                st.pyplot(fig2)
                                plt.close(fig2)
                            
                            # Expandable: Professional multi-track display
                            with st.expander("📈 Professional Multi-Track Log Display", expanded=False):
                                from plotting import create_professional_log_display
                                
                                header_display = {
                                    'WELL': header.get('WELL', 'Unknown'),
                                    'FIELD': header.get('FIELD', header.get('FLD', '')),
                                    'STRT': df['DEPTH'].min(),
                                    'STOP': df['DEPTH'].max(),
                                    'STEP': np.median(np.diff(df['DEPTH'].values)) if len(df) > 1 else 0
                                }
                                
                                fig = create_professional_log_display(
                                    df, mapping,
                                    header_info=header_display,
                                    settings={'scale_ratio': scale_ratio, 'depth_unit': unit},
                                    show_gr_fill=True,
                                    show_dn_crossover=True,
                                    highlight_mask=combined_mask,
                                    highlight_color='#ff8c00',
                                    highlight_alpha=0.4
                                )
                                st.pyplot(fig)
                                plt.close(fig)
                            
                            # Removal options
                            st.markdown("### 🧹 Remove Detected Noise")
                            
                            remove_col1, remove_col2 = st.columns(2)
                            
                            with remove_col1:
                                remove_method = st.selectbox(
                                    "Removal Method",
                                    ['trim', 'interpolate', 'nan'],
                                    help="trim: remove rows, interpolate: replace with interpolated values"
                                )
                            
                            with remove_col2:
                                if st.button("🗑️ Apply Removal", type="secondary"):
                                    df_original = df.copy()
                                    for name, result in results:
                                        removal_result = remove_noise(df, result, method=remove_method)
                                        df = removal_result.cleaned_df
                                    
                                    st.session_state['noise_cleaned_df'] = df
                                    st.session_state['noise_original_df'] = df_original
                                    st.session_state['noise_cleaned'] = True
                                    st.success(f"✅ Noise removed using {remove_method} method")
                            
                            # Before/After comparison if cleaned
                            if st.session_state.get('noise_cleaned') and 'noise_cleaned_df' in st.session_state:
                                st.markdown("### 📊 Before / After Comparison (Industry Standard Display)")
                                
                                from plotting import create_before_after_comparison
                                
                                df_original = st.session_state.get('noise_original_df', df)
                                df_cleaned = st.session_state['noise_cleaned_df']
                                
                                # Select a curve to compare
                                compare_curve = st.selectbox(
                                    "Select curve to compare",
                                    options=noise_curves,
                                    key="noise_compare_curve"
                                )
                                
                                # Create combined noise mask for highlighting
                                combined_mask = np.zeros(len(df_original), dtype=bool)
                                for _, result in results:
                                    combined_mask |= result.noise_mask
                                
                                # Prepare header info
                                header_display = {
                                    'WELL': header.get('WELL', 'Unknown'),
                                    'STRT': df_original['DEPTH'].min(),
                                    'STOP': df_original['DEPTH'].max()
                                }
                                
                                # Use professional before/after comparison
                                comparison_fig = create_before_after_comparison(
                                    df_original, df_cleaned, compare_curve,
                                    mapping=mapping,
                                    header_info=header_display,
                                    settings={'scale_ratio': scale_ratio, 'depth_unit': unit},
                                    highlight_mask=combined_mask,
                                    title_before="BEFORE (With Noise)",
                                    title_after="AFTER (Cleaned)"
                                )
                                st.pyplot(comparison_fig)
                                plt.close(comparison_fig)
                    
                    except Exception as e:
                        st.error(f"Error during noise detection: {str(e)}")
            
            # Spike detection section
            st.markdown("---")
            st.markdown("### ⚡ Spike Noise Detection")
            
            spike_col1, spike_col2 = st.columns(2)
            
            with spike_col1:
                spike_window = st.slider("Spike Window", min_value=3, max_value=15, value=5, key="spike_win")
            
            with spike_col2:
                spike_std = st.slider("Std Threshold", min_value=1.0, max_value=5.0, value=3.0, key="spike_std")
            
            if st.button("⚡ Detect Spikes", key="detect_spikes"):
                with st.spinner("Detecting spike noise..."):
                    spike_mask = detect_spike_noise(df, noise_curves, window=spike_window, std_threshold=spike_std)
                    n_spikes = np.sum(spike_mask)
                    
                    if n_spikes > 0:
                        st.markdown("### ⚡ Detected Spikes - Per Curve Visualization")
                        st.markdown("""
                        <div style="padding: 0.5rem; background: #fef2f2; border-radius: 4px; margin-bottom: 1rem; border-left: 4px solid #dc3545;">
                            <small style="color: #991b1b;">
                                ❌ <strong>Red X markers show detected spike anomalies on each curve</strong> - 
                                {0} spikes detected ({1:.2f}% of data)
                            </small>
                        </div>
                        """.format(n_spikes, n_spikes/len(df)*100), unsafe_allow_html=True)
                        
                        from plotting import create_spike_visualization
                        
                        # Create per-curve spike visualization
                        spike_fig = create_spike_visualization(
                            df,
                            noise_curves,
                            spike_mask,
                            depth_col='DEPTH',
                            settings={'scale_ratio': scale_ratio},
                            title="Spike Noise Detection Results"
                        )
                        st.pyplot(spike_fig)
                        plt.close(spike_fig)
                        
                        # Spike depth summary
                        st.markdown("### 📋 Spike Locations")
                        spike_depths = df.loc[spike_mask, 'DEPTH'].values
                        if len(spike_depths) > 0:
                            spike_df = pd.DataFrame({
                                'Depth': spike_depths[:min(20, len(spike_depths))],
                                **{col: df.loc[spike_mask, col].values[:min(20, len(spike_depths))] for col in noise_curves if col in df.columns}
                            }).round(3)
                            if len(spike_depths) > 20:
                                st.info(f"Showing first 20 of {len(spike_depths)} spikes")
                            st.dataframe(spike_df, use_container_width=True, hide_index=True)
                        
                        # Expandable: Professional display
                        with st.expander("📈 Professional Multi-Track Log Display", expanded=False):
                            from plotting import create_professional_log_display
                            
                            header_display = {
                                'WELL': header.get('WELL', 'Unknown'),
                                'FIELD': header.get('FIELD', header.get('FLD', '')),
                                'STRT': df['DEPTH'].min(),
                                'STOP': df['DEPTH'].max(),
                                'STEP': np.median(np.diff(df['DEPTH'].values)) if len(df) > 1 else 0
                            }
                            
                            fig = create_professional_log_display(
                                df, mapping,
                                header_info=header_display,
                                settings={'scale_ratio': scale_ratio, 'depth_unit': unit},
                                show_gr_fill=True,
                                show_dn_crossover=True,
                                highlight_mask=spike_mask,
                                highlight_color='#dc3545',
                                highlight_alpha=0.5
                            )
                            st.pyplot(fig)
                            plt.close(fig)
                    else:
                        st.success("✅ No spikes detected with current threshold settings.")
        
        # =================================================================
        # TAB 3: LOG SPLICING (info for single file mode)
        # =================================================================
        with tab3:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-title">🔗 Log Splicing & Concatenation</div>
                <div class="feature-desc">
                    Log splicing combines multiple log runs from different depth intervals into a 
                    continuous curve. This feature requires multiple LAS files.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.warning("⚠️ Log splicing requires multiple LAS files. Please select 'Multiple LAS Files' in the upload mode.")
            
            # Show the workflow diagram
            st.markdown("""
            ### 🔄 Splicing Workflow
            
            <div class="algorithm-box">
            <strong>Step-by-step process:</strong><br><br>
            1. <strong>Identify curve segments</strong> - Find matching curves across files<br>
            2. <strong>Sort by depth</strong> - Order files from shallowest to deepest<br>
            3. <strong>Assess data quality</strong> - Check for nulls and gaps<br>
            4. <strong>Detect overlap/gap</strong> - Find where runs connect<br>
            5. <strong>Align overlapping sections</strong> - Cross-correlation + DTW<br>
            6. <strong>Splice using cut or blend</strong> - Merge at optimal points<br>
            7. <strong>Post-splice QC</strong> - Verify continuity<br>
            8. <strong>Output continuous curve</strong> - Export merged result
            </div>
            """, unsafe_allow_html=True)
        
        # =================================================================
        # TAB 4: DEPTH ALIGNMENT (single file mode - limited)
        # =================================================================
        with tab4:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-title">📐 Depth Alignment of Logging Measurements</div>
                <div class="feature-desc">
                    Align measurements from different tools that may have depth discrepancies. 
                    For full alignment capabilities, upload multiple files.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("💡 For tool-to-tool alignment, upload multiple LAS files. Single file mode shows curve correlation analysis.")
            
            # Show correlation between curves in single file
            if len(feature_columns) >= 2:
                st.markdown("### 📊 Curve Correlation Analysis")
                
                corr_col1, corr_col2 = st.columns(2)
                
                with corr_col1:
                    ref_curve = st.selectbox("Reference Curve", options=feature_columns, index=0)
                
                with corr_col2:
                    tgt_curve = st.selectbox(
                        "Target Curve", 
                        options=[c for c in feature_columns if c != ref_curve],
                        index=0 if len(feature_columns) > 1 else 0
                    )
                
                if st.button("📊 Analyze Correlation", type="primary"):
                    from plotting import create_overlay_plot
                    
                    # Calculate correlation
                    valid_mask = ~(df[ref_curve].isna() | df[tgt_curve].isna())
                    if valid_mask.sum() > 10:
                        correlation = np.corrcoef(
                            df.loc[valid_mask, ref_curve],
                            df.loc[valid_mask, tgt_curve]
                        )[0, 1]
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{correlation:.3f}</div>
                            <div class="metric-label">Pearson Correlation</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Overlay plot
                        fig = create_overlay_plot(
                            df, [ref_curve, tgt_curve],
                            title=f"Curve Overlay: {ref_curve} vs {tgt_curve}",
                            normalize=True,
                            settings={'scale_ratio': scale_ratio}
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.warning("Insufficient valid data points for correlation analysis.")
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


# =============================================================================
# MAIN CONTENT - MULTI-FILE MODE WITH WELL GROUPING
# =============================================================================

elif upload_mode == "Upload File(s)" and multi_files and len(multi_files) >= 2:
    try:
        # =================================================================
        # STEP 1: SCAN & GROUP FILES BY WELL
        # =================================================================
        
        # Only re-process if files changed
        file_names_hash = hash(tuple(f.name for f in multi_files))
        
        if ('petro_well_groups_hash' not in st.session_state or 
            st.session_state.get('petro_well_groups_hash') != file_names_hash):
            
            with st.spinner("🔍 Scanning files and detecting wells..."):
                progress_messages = []
                
                def capture_progress(step, msg):
                    progress_messages.append((step, msg))
                
                group_result = group_files_by_well(
                    multi_files,
                    progress_callback=capture_progress
                )
                
                st.session_state['petro_well_groups'] = group_result.well_groups
                st.session_state['petro_duplicate_warnings'] = group_result.duplicate_warnings
                st.session_state['petro_num_wells'] = group_result.num_wells
                st.session_state['petro_num_files_total'] = group_result.num_files_total
                st.session_state['petro_well_groups_hash'] = file_names_hash
                
                # Clear any previous analysis results
                for key in ['petro_selected_well', 'petro_loaded_dfs', 'petro_loaded_mappings']:
                    if key in st.session_state:
                        del st.session_state[key]
        
        well_groups = st.session_state['petro_well_groups']
        duplicate_warnings = st.session_state['petro_duplicate_warnings']
        num_wells = st.session_state['petro_num_wells']
        num_files_total = st.session_state['petro_num_files_total']
        
        # Display grouping summary
        if num_wells == 0:
            st.error("❌ No valid LAS files could be processed.")
            st.stop()
        
        # Show duplicate warnings if any
        if duplicate_warnings:
            st.markdown("### ⚠️ Duplicate Files Detected")
            for warning in duplicate_warnings:
                st.markdown(f"""
                <div class="duplicate-warning">
                    ⚠️ {warning}
                </div>
                """, unsafe_allow_html=True)
        
        # Summary metrics
        st.markdown("### 📊 File Analysis Summary")
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
        st.markdown("### 🎯 Select Target Well")
        
        well_names = list(well_groups.keys())
        
        # Show well selector
        if num_wells == 1:
            selected_well = well_names[0]
            st.info(f"📍 **Single well detected:** {selected_well} - automatically selected")
        else:
            st.markdown(f"*Detected **{num_wells} unique wells** in your upload. Select which one to analyze:*")
            
            selected_well = st.selectbox(
                "Select Target Well",
                options=well_names,
                format_func=lambda x: f"{x} ({len(well_groups[x])} files)",
                help="Choose the well you want to analyze. Only files belonging to this well will be processed.",
                key="petro_well_selector"
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
                <h3 style="color: #38bdf8; margin: 0 0 1rem 0;">📊 {selected_well}</h3>
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
            
            # Show files for this well in an expander
            with st.expander(f"📁 Files for {selected_well} ({len(selected_files)} files)", expanded=False):
                file_data = []
                for f in selected_files:
                    unit_badge = '🔵 M' if f.original_unit == 'm' else '🟠 FT'
                    file_data.append({
                        'File Name': f.filename,
                        'Unit': unit_badge,
                        'Start (m)': f"{f.start_depth:.1f}",
                        'End (m)': f"{f.stop_depth:.1f}",
                        'Curves': len(f.curves)
                    })
                st.dataframe(pd.DataFrame(file_data), use_container_width=True, hide_index=True)
            
            # =================================================================
            # STEP 3: LOAD SELECTED WELL DATA FOR ANALYSIS
            # =================================================================
            
            # Load data for selected well (cache in session state)
            if (st.session_state.get('petro_selected_well') != selected_well):
                with st.spinner(f"Loading data for {selected_well}..."):
                    dataframes = []
                    headers = []
                    mappings = []
                    
                    for preprocessed_file in selected_files:
                        # Use the df from PreprocessedLAS (already normalized to meters)
                        df = preprocessed_file.df.copy()
                        dataframes.append(df)
                        
                        # Create a simple header from preprocessed file info
                        header_info = {
                            'WELL': preprocessed_file.well_name,
                            'LOC': preprocessed_file.location,
                            'STRT': preprocessed_file.start_depth,
                            'STOP': preprocessed_file.stop_depth,
                            'STEP': preprocessed_file.step
                        }
                        headers.append(header_info)
                        
                        # Create mapping from available curves
                        mapping = {}
                        for curve in preprocessed_file.curves:
                            curve_upper = curve.upper()
                            if 'GR' in curve_upper:
                                mapping['GR'] = curve
                            elif 'RT' in curve_upper or 'RDEP' in curve_upper or 'ILD' in curve_upper:
                                mapping['RES_DEEP'] = curve
                            elif 'RM' in curve_upper or 'RMED' in curve_upper or 'ILM' in curve_upper:
                                mapping['RES_MED'] = curve
                            elif 'RS' in curve_upper or 'RSHAL' in curve_upper or 'LLS' in curve_upper:
                                mapping['RES_SHAL'] = curve
                            elif 'RHOB' in curve_upper or 'DEN' in curve_upper:
                                mapping['DENS'] = curve
                            elif 'NPHI' in curve_upper or 'NEU' in curve_upper:
                                mapping['NEUT'] = curve
                            elif 'DT' in curve_upper or 'SONIC' in curve_upper:
                                mapping['SONIC'] = curve
                            elif 'CALI' in curve_upper:
                                mapping['CALIPER'] = curve
                        mappings.append(mapping)
                    
                    st.session_state['petro_selected_well'] = selected_well
                    st.session_state['petro_loaded_dfs'] = dataframes
                    st.session_state['petro_loaded_mappings'] = mappings
                    st.session_state['petro_loaded_headers'] = headers
                    st.session_state['petro_depth_unit'] = 'm'  # PreprocessedLAS is always in meters
            
            # Retrieve loaded data
            dataframes = st.session_state['petro_loaded_dfs']
            mappings = st.session_state['petro_loaded_mappings']
            headers = st.session_state['petro_loaded_headers']
            unit = st.session_state.get('petro_depth_unit', 'm')
            
            # Find common curves across selected well's files
            common_curves = set(dataframes[0].columns)
            for df in dataframes[1:]:
                common_curves &= set(df.columns)
            common_curves = [c for c in common_curves if c != 'DEPTH']
            
            if not common_curves:
                st.error("No common curves found across files for this well!")
                st.stop()
            
            st.success(f"✅ Ready to analyze **{selected_well}** | {len(common_curves)} common curves: {', '.join(common_curves[:5])}")
            
            # Add professional log preview for multi-file mode
            with st.expander("📊 Professional Log Display (Multi-File Overlay)", expanded=True):
                from plotting import create_multi_file_overlay, create_professional_log_display
                
                st.markdown("""
                <div style="padding: 0.5rem; background: #f8fafc; border-radius: 4px; margin-bottom: 1rem;">
                    <small style="color: #64748b;">
                        📋 <strong>Multi-File Overlay:</strong> View all log runs for alignment comparison before splicing
                    </small>
                </div>
                """, unsafe_allow_html=True)
                
                # Select curve to preview
                preview_curve = st.selectbox(
                    "Select curve to preview",
                    options=common_curves,
                    index=common_curves.index('GR') if 'GR' in common_curves else 0,
                    key="multi_preview_curve"
                )
                
                # Use professional overlay function
                overlay_fig = create_multi_file_overlay(
                    dataframes,
                    preview_curve,
                    headers=headers,
                    settings={'scale_ratio': scale_ratio, 'depth_unit': 'm'}
                )
                
                if overlay_fig:
                    st.pyplot(overlay_fig)
                    plt.close(overlay_fig)
                else:
                    st.warning("No data available for overlay display")
                
                st.markdown("---")
                
                # Also show professional display for the first file as reference
                st.markdown("#### 📐 Reference: First File Full Log Display")
                
                first_header = {
                    'WELL': headers[0].get('WELL', selected_well) if headers else selected_well,
                    'FIELD': headers[0].get('FIELD', '') if headers else '',
                    'STRT': dataframes[0]['DEPTH'].min(),
                    'STOP': dataframes[0]['DEPTH'].max(),
                    'STEP': np.median(np.diff(dataframes[0]['DEPTH'].values)) if len(dataframes[0]) > 1 else 0
                }
                
                ref_fig = create_professional_log_display(
                    dataframes[0], mappings[0] if mappings else {},
                    header_info=first_header,
                    settings={'scale_ratio': scale_ratio, 'depth_unit': 'm'},
                    show_gr_fill=True,
                    show_dn_crossover=True
                )
                st.pyplot(ref_fig)
                plt.close(ref_fig)
        
        # Main tabs for multi-file operations
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔍 Outlier Detection",
            "🔧 Noise Removal",
            "🔗 Log Splicing",
            "📐 Depth Alignment"
        ])
        
        # =================================================================
        # TAB 3: LOG SPLICING (multi-file mode) - ML-Based Workflow
        # =================================================================
        with tab3:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-title">🔗 ML-Based Log Splicing</div>
                <div class="feature-desc">
                    Combine multiple log runs using an ML-enhanced workflow: Hampel filter for QC,
                    rolling variance for stability detection, PELT for optimal splice point detection,
                    and weighted blending for smooth transitions.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            from shared.ml_splicing import (
                ml_splice_logs, find_common_curves, get_recommended_correlation_curve,
                find_overlap_region, DEFAULT_GRID_STEP, DEFAULT_HAMPEL_WINDOW,
                DEFAULT_HAMPEL_THRESHOLD, DEFAULT_VARIANCE_WINDOW, DEFAULT_BLEND_ZONE
            )
            from plotting import create_overlay_plot
            
            # Splicing parameters
            st.markdown("### ⚙️ ML Splicing Parameters")
            
            sp_col1, sp_col2 = st.columns(2)
            
            with sp_col1:
                correlation_curve = st.selectbox(
                    "Correlation Curve",
                    options=common_curves,
                    index=common_curves.index('GR') if 'GR' in common_curves else 0,
                    help="Curve to use for splicing alignment"
                )
                
                grid_step = st.selectbox(
                    "Grid Resolution",
                    options=[0.0762, 0.1524, 0.3048],
                    index=1,
                    format_func=lambda x: f"{x:.4f} m ({x/0.3048:.2f} ft)"
                )
            
            with sp_col2:
                blend_zone = st.slider(
                    "Blend Zone Width (m)",
                    min_value=1.0, max_value=10.0, value=DEFAULT_BLEND_ZONE,
                    help="Width of transition zone for smooth blending"
                )
                
                use_pelt = st.checkbox(
                    "Use PELT Change Point Detection",
                    value=True,
                    help="Use PELT algorithm to find optimal splice points"
                )
            
            # Advanced QC parameters
            with st.expander("🔧 Advanced QC Parameters", expanded=False):
                qc_col1, qc_col2 = st.columns(2)
                
                with qc_col1:
                    hampel_window = st.slider(
                        "Hampel Filter Window",
                        min_value=3, max_value=21, value=DEFAULT_HAMPEL_WINDOW, step=2,
                        help="Window size for spike detection (must be odd)"
                    )
                    hampel_threshold = st.slider(
                        "Hampel Threshold (MAD units)",
                        min_value=1.0, max_value=5.0, value=DEFAULT_HAMPEL_THRESHOLD,
                        help="Threshold for spike detection (higher = less sensitive)"
                    )
                
                with qc_col2:
                    variance_window = st.slider(
                        "Variance Window",
                        min_value=5, max_value=51, value=DEFAULT_VARIANCE_WINDOW, step=2,
                        help="Window size for rolling variance calculation"
                    )
            
            # Algorithm explanation
            with st.expander("📐 ML Splicing Workflow", expanded=False):
                st.markdown("""
                <div class="algorithm-box">
                <strong>7-Step ML-Based Splicing Pipeline:</strong><br><br>
                
                <strong>1. Overlap Detection</strong> (Rule-based logic)<br>
                • Identifies common depth region between log runs<br><br>
                
                <strong>2. Resampling</strong> (Linear interpolation)<br>
                • Aligns both logs to a common depth grid<br><br>
                
                <strong>3. QC - Hampel Filter</strong> (Remove spikes)<br>
                • Robust outlier detection using Median Absolute Deviation<br>
                • Formula: <code>spike if |x - median| > threshold × MAD × 1.4826</code><br><br>
                
                <strong>4. Stability Metric</strong> (Rolling variance)<br>
                • Detects noisy/unstable zones in each log<br>
                • Lower variance = more trustworthy data<br><br>
                
                <strong>5. Splice Point Detection</strong> (CPD-PELT)<br>
                • Change Point Detection using PELT algorithm<br>
                • Finds natural transition points in the signal<br><br>
                
                <strong>6. Run Selection</strong> (Trust scoring)<br>
                • Calculates confidence scores for each log run<br>
                • Based on stability, coverage, and consistency<br><br>
                
                <strong>7. Transition</strong> (Weighted blending)<br>
                • Smooth sigmoid-based transition between logs<br>
                • Trust-weighted blending in the transition zone
                </div>
                """, unsafe_allow_html=True)
            
            # Run splicing
            if st.button("🔗 Run ML Splicing", type="primary", key="run_splice"):
                with st.spinner("Running ML-based splicing pipeline..."):
                    try:
                        # Sort by depth (shallowest first)
                        sorted_indices = sorted(
                            range(len(dataframes)),
                            key=lambda i: dataframes[i]['DEPTH'].min()
                        )
                        
                        sorted_dfs = [dataframes[i] for i in sorted_indices]
                        
                        # Progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Splice pairs sequentially
                        current_depth = sorted_dfs[0]['DEPTH'].values
                        current_signal = sorted_dfs[0][correlation_curve].values
                        
                        splice_results = []
                        
                        def update_progress(step, msg):
                            status_text.text(f"[{step.upper()}] {msg}")
                        
                        for i in range(1, len(sorted_dfs)):
                            progress_bar.progress(int((i / len(sorted_dfs)) * 100))
                            
                            target_depth = sorted_dfs[i]['DEPTH'].values
                            target_signal = sorted_dfs[i][correlation_curve].values
                            
                            st.info(f"🔄 Splicing Run {i+1} into composite...")
                            
                            result = ml_splice_logs(
                                shallow_depth=current_depth,
                                shallow_signal=current_signal,
                                deep_depth=target_depth,
                                deep_signal=target_signal,
                                grid_step=grid_step,
                                hampel_window=hampel_window,
                                hampel_threshold=hampel_threshold,
                                variance_window=variance_window,
                                blend_zone=blend_zone,
                                use_pelt=use_pelt,
                                progress_callback=update_progress
                            )
                            
                            splice_results.append(result)
                            current_depth = result.merged_depth
                            current_signal = result.merged_signal
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Splicing complete!")
                        
                        st.session_state['splice_results'] = splice_results
                        st.session_state['spliced_depth'] = current_depth
                        st.session_state['spliced_signal'] = current_signal
                        
                        # Display results
                        st.markdown('<div class="feature-header">📊 ML Splicing Results</div>', unsafe_allow_html=True)
                        
                        for i, result in enumerate(splice_results):
                            st.markdown(f"**Splice {i+1}:**")
                            
                            # Row 1: Main metrics
                            m1, m2, m3, m4 = st.columns(4)
                            
                            with m1:
                                st.metric("Splice Point", f"{result.splice_point:.1f} m")
                            
                            with m2:
                                quality_pct = int(result.splice_quality_score * 100)
                                st.metric("Quality Score", f"{quality_pct}%")
                            
                            with m3:
                                st.metric("Overlap", f"{result.overlap_start:.1f}-{result.overlap_end:.1f} m")
                            
                            with m4:
                                st.metric("Blend Zone", f"{result.blend_start:.1f}-{result.blend_end:.1f} m")
                            
                            # Row 2: Trust and stability
                            t1, t2, t3, t4 = st.columns(4)
                            
                            with t1:
                                trust_shallow_pct = int(result.trust_shallow * 100)
                                st.metric("Shallow Trust", f"{trust_shallow_pct}%")
                            
                            with t2:
                                trust_deep_pct = int(result.trust_deep * 100)
                                st.metric("Deep Trust", f"{trust_deep_pct}%")
                            
                            with t3:
                                stab_shallow_pct = int(result.shallow_stability * 100)
                                st.metric("Shallow Stability", f"{stab_shallow_pct}%")
                            
                            with t4:
                                stab_deep_pct = int(result.deep_stability * 100)
                                st.metric("Deep Stability", f"{stab_deep_pct}%")
                            
                            # Show change points if detected
                            if result.change_points:
                                st.caption(f"📍 Detected change points: {', '.join([f'{cp:.1f}m' for cp in result.change_points])}")
                            
                            st.divider()
                        
                        # Before/After Visualization - Professional Industry Standard
                        st.markdown("### 📈 Before / After Comparison")
                        
                        from plotting import COLORS, STANDARD_RANGES
                        from matplotlib.ticker import AutoMinorLocator
                        import matplotlib.gridspec as gridspec
                        
                        # Calculate dimensions
                        all_depths = np.concatenate([df['DEPTH'].values for df in sorted_dfs])
                        depth_range = all_depths.max() - all_depths.min()
                        height_in = min(14, max(8, depth_range / 80))
                        
                        # Create professional figure with header
                        fig = plt.figure(figsize=(14, height_in), facecolor='white')
                        
                        gs = gridspec.GridSpec(2, 5, figure=fig,
                                             height_ratios=[0.06, 0.94],
                                             width_ratios=[0.4, 2.5, 0.2, 2.5, 0.8],
                                             wspace=0.02, hspace=0.02)
                        
                        # Header
                        ax_header = fig.add_subplot(gs[0, :])
                        ax_header.set_facecolor(COLORS['HEADER_BG'])
                        ax_header.axis('off')
                        ax_header.text(0.5, 0.5, f"ML Log Splicing Results - {selected_well} - {correlation_curve}",
                                      fontsize=12, fontweight='bold', ha='center', va='center')
                        
                        # Depth track
                        ax_depth = fig.add_subplot(gs[1, 0])
                        ax_depth.set_facecolor(COLORS['TRACK_BG'])
                        ax_depth.set_ylabel("Depth (m)", fontsize=10, fontweight='bold')
                        ax_depth.xaxis.set_visible(False)
                        ax_depth.yaxis.grid(True, linestyle='-', linewidth=0.5, color=COLORS['GRID'])
                        ax_depth.set_ylim(all_depths.max(), all_depths.min())
                        for spine in ax_depth.spines.values():
                            spine.set_color(COLORS['BORDER'])
                        
                        # Before track
                        ax_before = fig.add_subplot(gs[1, 1], sharey=ax_depth)
                        ax_before.set_facecolor(COLORS['TRACK_BG'])
                        ax_before.set_title("BEFORE\n(Original Runs)", fontsize=10, fontweight='bold', pad=8)
                        
                        # Plot original segments with different colors
                        segment_colors = ['#0066CC', '#CC0000', '#00AA00', '#FF8C00', '#9933FF']
                        for idx, sdf in enumerate(sorted_dfs):
                            color = segment_colors[idx % len(segment_colors)]
                            ax_before.plot(sdf[correlation_curve].values, sdf['DEPTH'].values,
                                          color=color, linewidth=1.5, label=f'Run {idx+1}')
                        
                        ax_before.xaxis.set_label_position('top')
                        ax_before.xaxis.tick_top()
                        ax_before.set_xlabel(f"{correlation_curve}", fontsize=9, fontweight='bold')
                        ax_before.xaxis.grid(True, linestyle='-', linewidth=0.5, color=COLORS['GRID'])
                        ax_before.yaxis.grid(True, linestyle='-', linewidth=0.5, color=COLORS['GRID'])
                        ax_before.legend(loc='lower right', fontsize=7, framealpha=0.9)
                        plt.setp(ax_before.get_yticklabels(), visible=False)
                        for spine in ax_before.spines.values():
                            spine.set_color(COLORS['BORDER'])
                        
                        # Spacer with arrow
                        ax_spacer = fig.add_subplot(gs[1, 2])
                        ax_spacer.axis('off')
                        ax_spacer.text(0.5, 0.5, '→', fontsize=20, ha='center', va='center', color='#666')
                        
                        # After track (spliced result)
                        ax_after = fig.add_subplot(gs[1, 3], sharey=ax_depth)
                        ax_after.set_facecolor(COLORS['TRACK_BG'])
                        ax_after.set_title("AFTER\n(ML Spliced)", fontsize=10, fontweight='bold', 
                                          pad=8, color='#00AA00')
                        
                        # Plot spliced result
                        ax_after.plot(current_signal, current_depth, color='#00AA00', linewidth=1.5)
                        
                        # Mark splice points and blend zones
                        for i, result in enumerate(splice_results):
                            # Blend zone shading
                            ax_after.axhspan(result.blend_start, result.blend_end, 
                                           alpha=0.2, color='#FFD700', zorder=0)
                            
                            # Splice point line
                            ax_after.axhline(y=result.splice_point, color='#FF8C00', 
                                           linestyle='--', linewidth=2, alpha=0.9)
                            ax_after.text(ax_after.get_xlim()[1] * 0.95, result.splice_point,
                                        f'SP{i+1}', fontsize=8, ha='right', va='bottom',
                                        color='#FF8C00', fontweight='bold',
                                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                            
                            # Change points
                            for cp in result.change_points:
                                ax_after.axhline(y=cp, color='#9933FF', 
                                               linestyle=':', linewidth=1, alpha=0.6)
                        
                        ax_after.xaxis.set_label_position('top')
                        ax_after.xaxis.tick_top()
                        ax_after.set_xlabel(f"{correlation_curve}", fontsize=9, fontweight='bold', color='#00AA00')
                        ax_after.xaxis.grid(True, linestyle='-', linewidth=0.5, color=COLORS['GRID'])
                        ax_after.yaxis.grid(True, linestyle='-', linewidth=0.5, color=COLORS['GRID'])
                        ax_after.spines['top'].set_color('#00AA00')
                        ax_after.spines['top'].set_linewidth(2)
                        plt.setp(ax_after.get_yticklabels(), visible=False)
                        for spine in ['bottom', 'left', 'right']:
                            ax_after.spines[spine].set_color(COLORS['BORDER'])
                        
                        # Quality track (showing trust/stability)
                        ax_quality = fig.add_subplot(gs[1, 4], sharey=ax_depth)
                        ax_quality.set_facecolor('#f0f0f0')
                        ax_quality.set_title("Quality\nMetrics", fontsize=9, fontweight='bold', pad=8)
                        
                        # Create quality visualization
                        for i, result in enumerate(splice_results):
                            # Draw quality bars in overlap region
                            overlap_mid = (result.overlap_start + result.overlap_end) / 2
                            bar_height = result.overlap_end - result.overlap_start
                            
                            # Shallow trust (left bar)
                            ax_quality.barh(overlap_mid, result.trust_shallow * 0.4, 
                                          height=bar_height * 0.4, left=0,
                                          color='#0066CC', alpha=0.7, label='Shallow' if i == 0 else '')
                            
                            # Deep trust (right bar)
                            ax_quality.barh(overlap_mid, result.trust_deep * 0.4, 
                                          height=bar_height * 0.4, left=0.5,
                                          color='#CC0000', alpha=0.7, label='Deep' if i == 0 else '')
                        
                        ax_quality.set_xlim(0, 1)
                        ax_quality.set_xticks([0.2, 0.7])
                        ax_quality.set_xticklabels(['S', 'D'], fontsize=8)
                        ax_quality.xaxis.set_label_position('top')
                        ax_quality.xaxis.tick_top()
                        plt.setp(ax_quality.get_yticklabels(), visible=False)
                        for spine in ax_quality.spines.values():
                            spine.set_color(COLORS['BORDER'])
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        # Legend explanation
                        st.caption("""
                        **Legend:** 🟠 Dashed line = Splice point | 🟡 Yellow zone = Blend transition | 
                        🟣 Dotted lines = Detected change points | Quality bars: S=Shallow trust, D=Deep trust
                        """)
                        
                        # Export option
                        st.markdown("### 📤 Export")
                        export_col1, export_col2 = st.columns(2)
                        
                        with export_col1:
                            if st.button("💾 Export Spliced Data", key="export_splice_csv"):
                                spliced_df = pd.DataFrame({
                                    'DEPTH': current_depth,
                                    correlation_curve: current_signal
                                })
                                
                                csv = spliced_df.to_csv(index=False)
                                st.download_button(
                                    "📥 Download CSV",
                                    csv,
                                    "ml_spliced_log.csv",
                                    "text/csv",
                                    key="download_splice_csv"
                                )
                        
                        with export_col2:
                            if st.button("📋 Export Splice Report", key="export_splice_report"):
                                # Create detailed report
                                report_lines = [
                                    "ML Log Splicing Report",
                                    "=" * 50,
                                    f"Well: {selected_well}",
                                    f"Curve: {correlation_curve}",
                                    f"Grid Step: {grid_step} m",
                                    f"Blend Zone: {blend_zone} m",
                                    "",
                                    "Pipeline Parameters:",
                                    f"  - Hampel Window: {hampel_window}",
                                    f"  - Hampel Threshold: {hampel_threshold}",
                                    f"  - Variance Window: {variance_window}",
                                    f"  - PELT Enabled: {use_pelt}",
                                    "",
                                    "Splice Results:",
                                    "-" * 50,
                                ]
                                
                                for i, result in enumerate(splice_results):
                                    report_lines.extend([
                                        f"\nSplice {i+1}:",
                                        f"  Splice Point: {result.splice_point:.2f} m",
                                        f"  Quality Score: {result.splice_quality_score:.3f}",
                                        f"  Overlap: {result.overlap_start:.2f} - {result.overlap_end:.2f} m",
                                        f"  Blend Zone: {result.blend_start:.2f} - {result.blend_end:.2f} m",
                                        f"  Shallow Trust: {result.trust_shallow:.3f}",
                                        f"  Deep Trust: {result.trust_deep:.3f}",
                                        f"  Shallow Stability: {result.shallow_stability:.3f}",
                                        f"  Deep Stability: {result.deep_stability:.3f}",
                                        f"  Change Points: {result.change_points}",
                                    ])
                                
                                report_text = "\n".join(report_lines)
                                st.download_button(
                                    "📥 Download Report",
                                    report_text,
                                    "ml_splice_report.txt",
                                    "text/plain",
                                    key="download_splice_report"
                                )
                    
                    except Exception as e:
                        st.error(f"Splicing error: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
        
        # =================================================================
        # TAB 4: DEPTH ALIGNMENT (multi-file mode)
        # =================================================================
        with tab4:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-title">📐 Depth Alignment of Logging Measurements</div>
                <div class="feature-desc">
                    Align measurements from different tools using ML algorithms: 
                    correlation-based methods and Siamese Neural Networks for feature matching.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            from ml_components.depth_alignment import (
                align_by_correlation, align_by_siamese, apply_depth_shift, get_model_summary
            )
            from plotting import create_comparison_log_plot, create_overlay_plot
            
            # File selection for alignment
            st.markdown("### 📁 Select Files to Align")
            
            # Build file info for display
            file_display_names = []
            for i, (header, df) in enumerate(zip(headers, dataframes)):
                well_name = header.get('WELL', 'Unknown') if isinstance(header, dict) else 'Unknown'
                depth_min = df['DEPTH'].min()
                depth_max = df['DEPTH'].max()
                file_display_names.append(f"{well_name} ({depth_min:.0f}-{depth_max:.0f}m)")
            
            align_col1, align_col2 = st.columns(2)
            
            with align_col1:
                ref_idx = st.selectbox(
                    "Reference File",
                    options=list(range(len(dataframes))),
                    format_func=lambda i: file_display_names[i] if i < len(file_display_names) else f"File {i+1}"
                )
            
            with align_col2:
                tgt_options = [i for i in range(len(dataframes)) if i != ref_idx]
                tgt_idx = st.selectbox(
                    "Target File",
                    options=tgt_options,
                    format_func=lambda i: file_display_names[i] if i < len(file_display_names) else f"File {i+1}"
                ) if tgt_options else None
            
            if tgt_idx is not None:
                ref_df = dataframes[ref_idx]
                tgt_df = dataframes[tgt_idx]
                
                # Curve selection
                alignment_curve = st.selectbox(
                    "Alignment Curve",
                    options=common_curves,
                    index=common_curves.index('GR') if 'GR' in common_curves else 0
                )
                
                # Method selection
                st.markdown("### 🔧 Alignment Method")
                
                alignment_method = st.radio(
                    "Method",
                    ["Correlation-Based (Fast)", "Siamese Neural Network (ML)"],
                    horizontal=True
                )
                
                # Method-specific parameters
                if "Correlation" in alignment_method:
                    max_shift = st.slider("Max Shift (m)", min_value=5.0, max_value=50.0, value=20.0)
                    
                    with st.expander("📐 Algorithm Details", expanded=False):
                        st.markdown("""
                        <div class="algorithm-box">
                        <strong>Cross-Correlation Alignment:</strong><br><br>
                        1. Resample both signals to a common grid<br>
                        2. Z-score normalize for amplitude independence<br>
                        3. Compute cross-correlation: <code>corr[lag] = ∑ ref[i] × tgt[i+lag]</code><br>
                        4. Find lag that maximizes correlation<br>
                        5. Convert lag to depth shift
                        </div>
                        """, unsafe_allow_html=True)
                
                else:
                    siamese_col1, siamese_col2, siamese_col3 = st.columns(3)
                    
                    with siamese_col1:
                        window_size = st.slider("Window Size", min_value=20, max_value=100, value=50)
                    
                    with siamese_col2:
                        n_pairs = st.slider("Training Pairs", min_value=100, max_value=500, value=200)
                    
                    with siamese_col3:
                        epochs = st.slider("Training Epochs", min_value=10, max_value=50, value=30)
                    
                    with st.expander("📐 Siamese Network Architecture", expanded=False):
                        st.markdown("""
                        <div class="algorithm-box">
                        <strong>Siamese Neural Network for Depth Alignment:</strong><br><br>
                        
                        <strong>Architecture:</strong><br>
                        • Conv1D(1 → 32, k=5) + BatchNorm + ReLU + MaxPool<br>
                        • Conv1D(32 → 64, k=5) + BatchNorm + ReLU + MaxPool<br>
                        • Conv1D(64 → 128, k=3) + BatchNorm + ReLU<br>
                        • AdaptiveAvgPool1d(1)<br>
                        • Linear(128 → 64) + ReLU + Dropout<br><br>
                        
                        <strong>Training:</strong><br>
                        • Generate positive pairs (matching positions)<br>
                        • Generate negative pairs (different positions)<br>
                        • Train with contrastive loss on cosine similarity<br><br>
                        
                        <strong>Inference:</strong><br>
                        • Slide window over target<br>
                        • Find position with maximum similarity
                        </div>
                        """, unsafe_allow_html=True)
                
                # Run alignment
                if st.button("🎯 Run Alignment", type="primary", key="run_align"):
                    with st.spinner("Computing optimal alignment..."):
                        try:
                            ref_depth = ref_df['DEPTH'].values
                            ref_signal = ref_df[alignment_curve].values
                            tgt_depth = tgt_df['DEPTH'].values
                            tgt_signal = tgt_df[alignment_curve].values
                            
                            if "Correlation" in alignment_method:
                                result = align_by_correlation(
                                    ref_depth, ref_signal,
                                    tgt_depth, tgt_signal,
                                    max_shift_meters=max_shift
                                )
                                
                                st.markdown('<div class="feature-header">📊 Alignment Results</div>', unsafe_allow_html=True)
                                
                                m1, m2, m3 = st.columns(3)
                                
                                with m1:
                                    st.markdown(f"""
                                    <div class="metric-card">
                                        <div class="metric-value">{result.optimal_shift:.2f}m</div>
                                        <div class="metric-label">Optimal Shift</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with m2:
                                    st.markdown(f"""
                                    <div class="metric-card">
                                        <div class="metric-value">{result.correlation_coefficient:.3f}</div>
                                        <div class="metric-label">Correlation</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with m3:
                                    st.markdown(f"""
                                    <div class="metric-card">
                                        <div class="metric-value">{result.confidence*100:.0f}%</div>
                                        <div class="metric-label">Confidence</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Correlation curve plot
                                st.markdown("### 📈 Correlation Function")
                                
                                fig, ax = plt.subplots(figsize=(10, 4))
                                ax.plot(result.shift_range, result.correlation_curve, 'b-', linewidth=1)
                                ax.axvline(x=result.optimal_shift, color='r', linestyle='--', 
                                           label=f'Optimal: {result.optimal_shift:.2f}m')
                                ax.set_xlabel('Depth Shift (m)')
                                ax.set_ylabel('Correlation')
                                ax.set_title('Cross-Correlation Function')
                                ax.legend()
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                                plt.close(fig)
                            
                            else:  # Siamese NN
                                progress_container = st.empty()
                                
                                def update_progress(epoch, loss):
                                    progress_container.progress(
                                        epoch / epochs,
                                        text=f"Training epoch {epoch}/{epochs} | Loss: {loss:.4f}"
                                    )
                                
                                result = align_by_siamese(
                                    ref_depth, ref_signal,
                                    tgt_depth, tgt_signal,
                                    window_size=window_size,
                                    n_pairs=n_pairs,
                                    epochs=epochs,
                                    progress_callback=update_progress
                                )
                                
                                progress_container.empty()
                                
                                st.markdown('<div class="feature-header">📊 Siamese Alignment Results</div>', unsafe_allow_html=True)
                                
                                m1, m2, m3 = st.columns(3)
                                
                                with m1:
                                    st.markdown(f"""
                                    <div class="metric-card">
                                        <div class="metric-value">{result.optimal_shift:.2f}m</div>
                                        <div class="metric-label">Optimal Shift</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with m2:
                                    st.markdown(f"""
                                    <div class="metric-card">
                                        <div class="metric-value">{result.similarity_score:.3f}</div>
                                        <div class="metric-label">Similarity Score</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with m3:
                                    st.markdown(f"""
                                    <div class="metric-card">
                                        <div class="metric-value">{result.confidence*100:.0f}%</div>
                                        <div class="metric-label">Confidence</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                # Training loss plot
                                if result.training_loss:
                                    st.markdown("### 📉 Training Progress")
                                    
                                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                                    
                                    ax1.plot(result.training_loss, 'b-', linewidth=1.5)
                                    ax1.set_xlabel('Epoch')
                                    ax1.set_ylabel('Loss')
                                    ax1.set_title('Training Loss')
                                    ax1.grid(True, alpha=0.3)
                                    
                                    ax2.plot(result.similarity_map, 'g-', linewidth=1)
                                    ax2.axhline(y=result.similarity_score, color='r', linestyle='--',
                                               label=f'Best: {result.similarity_score:.3f}')
                                    ax2.set_xlabel('Position')
                                    ax2.set_ylabel('Similarity')
                                    ax2.set_title('Similarity Map')
                                    ax2.legend()
                                    ax2.grid(True, alpha=0.3)
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close(fig)
                            
                            # Apply shift and show comparison
                            st.markdown("### 📊 Before/After Comparison")
                            
                            shift = result.optimal_shift if hasattr(result, 'optimal_shift') else result.optimal_shift
                            tgt_df_shifted = apply_depth_shift(tgt_df.copy(), shift, 'DEPTH')
                            
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
                            
                            # Before
                            ax1.plot(ref_df[alignment_curve], ref_df['DEPTH'], 'b-', linewidth=1.5, label='Reference')
                            ax1.plot(tgt_df[alignment_curve], tgt_df['DEPTH'], 'r--', linewidth=1.5, label='Target (Original)')
                            ax1.set_ylim(max(ref_depth.max(), tgt_depth.max()), min(ref_depth.min(), tgt_depth.min()))
                            ax1.set_xlabel(alignment_curve)
                            ax1.set_ylabel('Depth (m)')
                            ax1.set_title('Before Alignment')
                            ax1.legend()
                            ax1.grid(True, alpha=0.3)
                            
                            # After
                            ax2.plot(ref_df[alignment_curve], ref_df['DEPTH'], 'b-', linewidth=1.5, label='Reference')
                            ax2.plot(tgt_df_shifted[alignment_curve], tgt_df_shifted['DEPTH'], 'g-', linewidth=1.5, label='Target (Aligned)')
                            ax2.set_ylim(max(ref_depth.max(), (tgt_depth + shift).max()), 
                                        min(ref_depth.min(), (tgt_depth + shift).min()))
                            ax2.set_xlabel(alignment_curve)
                            ax2.set_title('After Alignment')
                            ax2.legend()
                            ax2.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                            
                            st.session_state['aligned_df'] = tgt_df_shifted
                            st.session_state['alignment_shift'] = shift
                        
                        except Exception as e:
                            st.error(f"Alignment error: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
        
        # =================================================================
        # TAB 1: OUTLIER DETECTION (multi-file - process all)
        # =================================================================
        with tab1:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-title">🔍 Automated Outlier Detection</div>
                <div class="feature-desc">
                    Detect outliers across all loaded files. Select curves to analyze and 
                    choose from multiple detection algorithms.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            from ml_components.outlier_detection import (
                detect_outliers_isolation_forest,
                detect_outliers_lof,
                detect_outliers_abod,
                PYOD_AVAILABLE
            )
            
            # Build file display names for selection
            outlier_file_names = []
            for i, (header, df) in enumerate(zip(headers, dataframes)):
                well_name = header.get('WELL', 'Unknown') if isinstance(header, dict) else 'Unknown'
                outlier_file_names.append(f"{well_name} ({len(df)} samples)")
            
            file_to_analyze = st.selectbox(
                "Select File to Analyze",
                options=list(range(len(dataframes))),
                format_func=lambda i: outlier_file_names[i] if i < len(outlier_file_names) else f"File {i+1}"
            )
            
            selected_df = dataframes[file_to_analyze]
            selected_mapping = mappings[file_to_analyze]
            feature_cols = get_available_feature_columns(selected_df, selected_mapping)
            
            if feature_cols:
                method = st.selectbox(
                    "Method",
                    ['Isolation Forest', 'LOF', 'ABOD'] if PYOD_AVAILABLE else ['Isolation Forest', 'LOF']
                )
                
                contam = st.slider("Contamination %", 1, 20, 5) / 100
                curves = st.multiselect("Curves", feature_cols, default=feature_cols[:2])
                
                if curves and st.button("Run Detection", key="multi_outlier"):
                    with st.spinner("Detecting..."):
                        if method == 'Isolation Forest':
                            result = detect_outliers_isolation_forest(selected_df, curves, contam)
                        elif method == 'LOF':
                            result = detect_outliers_lof(selected_df, curves, contamination=contam)
                        else:
                            result = detect_outliers_abod(selected_df, curves, contamination=contam)
                        
                        st.metric("Outliers Found", result.num_anomalies)
                        st.metric("Confidence", f"{result.confidence*100:.0f}%")
        
        # =================================================================
        # TAB 2: NOISE REMOVAL (multi-file)
        # =================================================================
        with tab2:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-title">🔧 Tool Noise Removal</div>
                <div class="feature-desc">
                    Detect and remove tool startup/shutdown noise from selected files.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            from ml_components.noise_removal import detect_tool_startup_noise, remove_noise
            
            # Build file display names for noise removal selection
            noise_file_names = []
            for i, (header, df) in enumerate(zip(headers, dataframes)):
                well_name = header.get('WELL', 'Unknown') if isinstance(header, dict) else 'Unknown'
                noise_file_names.append(f"{well_name} ({df['DEPTH'].min():.0f}-{df['DEPTH'].max():.0f}m)")
            
            noise_file_idx = st.selectbox(
                "Select File",
                options=list(range(len(dataframes))),
                format_func=lambda i: noise_file_names[i] if i < len(noise_file_names) else f"File {i+1}",
                key="noise_file_select"
            )
            
            noise_df = dataframes[noise_file_idx]
            noise_mapping = mappings[noise_file_idx]
            noise_feature_cols = get_available_feature_columns(noise_df, noise_mapping)
            
            if noise_feature_cols:
                noise_curves = st.multiselect("Analyze Curves", noise_feature_cols, 
                                               default=noise_feature_cols[:2], key="multi_noise_curves")
                
                if noise_curves and st.button("Detect Noise", key="multi_noise"):
                    result = detect_tool_startup_noise(noise_df, noise_curves, 'DEPTH')
                    st.metric("Noise Samples", result.noise_samples)
                    st.metric("Affected %", f"{result.noise_percentage:.1f}%")
    
    except Exception as e:
        st.error(f"Error loading files: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


# =============================================================================
# NO DATA STATE
# =============================================================================

else:
    st.markdown("""
    <div class="info-panel" style="text-align: center; padding: 40px;">
        <h2 style="color: #38bdf8;">📂 Select a Folder or Upload LAS Files to Begin</h2>
        <p style="color: #94a3b8;">
            Use the sidebar to select a folder or upload well log files for petrophysical analysis.<br><br>
            <strong>Folder Selection:</strong> Browse by Field and Well from existing LAS files<br>
            <strong>Single File Upload:</strong> Outlier detection, noise removal, curve analysis<br>
            <strong>Multi-File Upload:</strong> Log splicing, depth alignment, batch processing
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature overview
    st.markdown('<div class="feature-header">🔧 Available Features</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🔍 Outlier Detection</div>
            <div class="feature-desc">
                <ul style="color: #94a3b8;">
                    <li>Isolation Forest</li>
                    <li>Local Outlier Factor (LOF)</li>
                    <li>Angular-Based Outlier Detection (ABOD)</li>
                    <li>Ensemble methods</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🔧 Noise Removal</div>
            <div class="feature-desc">
                <ul style="color: #94a3b8;">
                    <li>Tool startup noise detection</li>
                    <li>Rolling variance analysis</li>
                    <li>Slope-based detection</li>
                    <li>Spike removal</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">🔗 Log Splicing</div>
            <div class="feature-desc">
                <ul style="color: #94a3b8;">
                    <li>Cross-correlation alignment</li>
                    <li>Dynamic Time Warping</li>
                    <li>Overlap detection</li>
                    <li>Quality assessment</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">📐 Depth Alignment</div>
            <div class="feature-desc">
                <ul style="color: #94a3b8;">
                    <li>Correlation-based methods</li>
                    <li>Siamese Neural Networks</li>
                    <li>Feature matching</li>
                    <li>Confidence estimation</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

