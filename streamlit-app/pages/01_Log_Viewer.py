"""
WellLog Analyzer Pro - Log Viewer Page
Professional LAS file visualization with industry-standard formatting.
"""

import streamlit as st
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for shared module access
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.las_parser import load_las, extract_header_info, detect_depth_units
from shared.curve_mapping import get_curve_mapping
from shared.data_processing import process_data, get_auto_scale, export_to_las

# Import plotting from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from plotting import create_log_plot, export_plot_to_bytes

# Page configuration
st.set_page_config(
    page_title="Log Viewer | WellLog Analyzer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling matching Schlumberger Techlog aesthetic
st.markdown("""
<style>
    /* Import professional font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Main app styling */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1a1d24 0%, #2d3748 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border: 1px solid #3d4852;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #00D4AA;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .main-subtitle {
        font-size: 0.9rem;
        color: #8892a0;
        margin-top: 0.3rem;
    }
    
    /* Well header panel - Industry standard look */
    .well-header {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 20px 25px;
        border-radius: 8px;
        border: 2px solid #dee2e6;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .well-name {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a1d24;
        margin-bottom: 12px;
        padding-bottom: 10px;
        border-bottom: 2px solid #00D4AA;
    }
    
    .well-meta-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 10px;
    }
    
    .well-meta-item {
        font-size: 0.9rem;
        color: #495057;
    }
    
    .well-meta-label {
        font-weight: 600;
        color: #1a1d24;
    }
    
    /* Export buttons styling */
    .export-section {
        background: #1e2530;
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
    }
    
    /* Track info badges */
    .track-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-right: 5px;
        margin-bottom: 5px;
    }
    
    .track-gr { background: #00AA00; color: white; }
    .track-res { background: #0066CC; color: white; }
    .track-dn { background: #CC0000; color: white; }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #1a1d24;
    }
    
    /* Info boxes */
    .info-box {
        background: #2d3748;
        border-left: 4px solid #00D4AA;
        padding: 12px 15px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Plot container */
    .plot-container {
        background: white;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <div class="main-title">📊 Log Viewer</div>
    <div class="main-subtitle">Professional LAS File Visualization • Industry-Standard Log Display</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### 📁 File Upload")
    uploaded_file = st.file_uploader("Choose a LAS file", type=['las', 'LAS'])
    
    st.markdown("---")
    st.markdown("### ⚙️ Plot Settings")
    
    # Scale settings
    scale_option = st.selectbox(
        "Depth Scale",
        ["1:200", "1:500", "1:1000", "Custom"],
        index=1,
        help="Standard well log scales"
    )
    if scale_option == "Custom":
        scale_ratio = st.number_input("Scale Ratio (1:X)", min_value=100, max_value=5000, value=500)
    else:
        scale_ratio = int(scale_option.split(":")[1])
    
    # Smoothing
    st.markdown("#### Data Processing")
    smooth_window = st.slider(
        "Smoothing Window",
        min_value=0, max_value=20, value=0,
        help="Moving average filter (0 = raw data)"
    )
    show_raw = st.checkbox("Show raw data info", value=False)
    
    # Visualization options
    st.markdown("#### 🎨 Visualization")
    show_gr_fill = st.checkbox(
        "GR Sand Shading",
        value=False,
        help="Shade sand indication on Gamma Ray track"
    )
    show_dn_fill = st.checkbox(
        "Density-Neutron Crossover",
        value=True,
        help="Show gas effect and lithology separation"
    )
    
    # Track limits
    st.markdown("#### 📏 Track Limits")
    
    with st.expander("Gamma Ray (Track 1)", expanded=False):
        gr_min = st.number_input("Min API", value=0, key="gr_min")
        gr_max = st.number_input("Max API", value=150, key="gr_max")
        auto_gr = st.checkbox("Auto-scale GR", value=False)
        
    with st.expander("Resistivity (Track 2)", expanded=False):
        res_min = st.number_input("Min Ohm.m", value=0.2, key="res_min", format="%.2f")
        res_max = st.number_input("Max Ohm.m", value=2000.0, key="res_max", format="%.1f")
        
    with st.expander("Density/Neutron (Track 3)", expanded=False):
        dens_min = st.number_input("Density Min (g/cc)", value=1.95, key="dens_min", format="%.2f")
        dens_max = st.number_input("Density Max (g/cc)", value=2.95, key="dens_max", format="%.2f")
        neut_min = st.number_input("Neutron Min (v/v)", value=-0.15, key="neut_min", format="%.2f")
        neut_max = st.number_input("Neutron Max (v/v)", value=0.45, key="neut_max", format="%.2f")

# Build settings dictionary
settings = {
    'scale_ratio': scale_ratio,
    'gr_min': gr_min, 'gr_max': gr_max,
    'res_min': res_min, 'res_max': res_max,
    'dens_min': dens_min, 'dens_max': dens_max,
    'neut_min': neut_min, 'neut_max': neut_max
}

# Main content
if uploaded_file:
    try:
        # Load and process data
        las = load_las(uploaded_file)
        mapping = get_curve_mapping(las)
        depth_unit = detect_depth_units(las)
        df_full = process_data(las, mapping, smooth_window=smooth_window)
        header_info = extract_header_info(las)
        
        # Auto-scaling if enabled
        if auto_gr and mapping['GR']:
            gr_min, gr_max = get_auto_scale(df_full, mapping['GR'], 0, 150)
            settings['gr_min'] = gr_min
            settings['gr_max'] = gr_max
        
        # Depth range for pagination
        min_depth = df_full['DEPTH'].min()
        max_depth = df_full['DEPTH'].max()
        
        # View interval selector in sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📍 View Interval")
        
        default_end = min(min_depth + 500, max_depth)
        depth_range = st.sidebar.slider(
            f"Depth Range ({depth_unit})",
            min_value=float(min_depth),
            max_value=float(max_depth),
            value=(float(min_depth), float(default_end)),
            step=10.0
        )
        start_depth, end_depth = depth_range
        
        # Filter data to selected range
        df = df_full[(df_full['DEPTH'] >= start_depth) & (df_full['DEPTH'] <= end_depth)]
        
        # Well Header Panel
        st.markdown(f"""
        <div class="well-header">
            <div class="well-name">🛢️ WELL: {header_info['WELL']}</div>
            <div class="well-meta-grid">
                <div class="well-meta-item">
                    <span class="well-meta-label">FIELD:</span> {header_info['FIELD'] or 'N/A'}
                </div>
                <div class="well-meta-item">
                    <span class="well-meta-label">LOCATION:</span> {header_info['LOC'] or 'N/A'}
                </div>
                <div class="well-meta-item">
                    <span class="well-meta-label">OPERATOR:</span> {header_info['COMP'] or 'N/A'}
                </div>
                <div class="well-meta-item">
                    <span class="well-meta-label">DEPTH:</span> {header_info['STRT']} {depth_unit} – {header_info['STOP']} {depth_unit}
                </div>
                <div class="well-meta-item">
                    <span class="well-meta-label">STEP:</span> {header_info['STEP']} {depth_unit}
                </div>
                <div class="well-meta-item">
                    <span class="well-meta-label">DATE:</span> {header_info.get('DATE', 'N/A') or 'N/A'}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Curve mapping info
        with st.expander("📊 Detected Curves & Mapping", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Track 1: Gamma Ray**")
                st.markdown(f"• GR: `{mapping['GR'] or 'Not found'}`")
            
            with col2:
                st.markdown("**Track 2: Resistivity**")
                st.markdown(f"• Deep: `{mapping['RES_DEEP'] or 'Not found'}`")
                st.markdown(f"• Medium: `{mapping['RES_MED'] or 'Not found'}`")
                st.markdown(f"• Shallow: `{mapping['RES_SHAL'] or 'Not found'}`")
            
            with col3:
                st.markdown("**Track 3: Density-Neutron**")
                st.markdown(f"• Density: `{mapping['DENS'] or 'Not found'}`")
                st.markdown(f"• Neutron: `{mapping['NEUT'] or 'Not found'}`")
            
            if show_raw:
                st.markdown("---")
                st.markdown("**Raw Data Sample:**")
                st.dataframe(df.head(20), width='stretch')
        
        # Calculate estimated plot height
        est_height_in = ((end_depth - start_depth) * 100) / scale_ratio / 2.54
        est_pixels = est_height_in * 100
        
        # Plot display
        st.markdown("### 📈 Well Log Display")
        
        if est_pixels > 65000:
            st.warning(f"⚠️ Selected range is too large for 1:{scale_ratio} scale (Height: {int(est_pixels)} px). Please reduce the depth range or change the scale.")
        else:
            # Create the plot
            fig = create_log_plot(df, mapping, settings, show_gr_fill=show_gr_fill, show_dn_fill=show_dn_fill)
            
            # Display plot
            st.pyplot(fig)
            
            # Export Section
            st.markdown("---")
            st.markdown("### 💾 Export Options")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # PNG Export
                png_data = export_plot_to_bytes(fig, format='png', dpi=150)
                st.download_button(
                    label="📷 Download PNG",
                    data=png_data,
                    file_name=f"{header_info['WELL']}_log.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            with col2:
                # High-res PNG
                png_hires = export_plot_to_bytes(fig, format='png', dpi=300)
                st.download_button(
                    label="📷 PNG (High-Res)",
                    data=png_hires,
                    file_name=f"{header_info['WELL']}_log_hires.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            with col3:
                # PDF Export
                pdf_data = export_plot_to_bytes(fig, format='pdf', dpi=150)
                st.download_button(
                    label="📄 Download PDF",
                    data=pdf_data,
                    file_name=f"{header_info['WELL']}_log.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            
            with col4:
                # LAS Export
                las_data = export_to_las(df, header_info, depth_unit)
                st.download_button(
                    label="📋 Export LAS",
                    data=las_data,
                    file_name=f"{header_info['WELL']}_export.las",
                    mime="text/plain",
                    use_container_width=True
                )
            
            # Close the figure to free memory
            plt.close(fig)
    
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        st.exception(e)

else:
    # Welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px;">
        <div style="font-size: 4rem; margin-bottom: 20px;">📊</div>
        <h2 style="color: #00D4AA; margin-bottom: 10px;">Welcome to Log Viewer</h2>
        <p style="color: #8892a0; font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
            Upload a LAS file to visualize well logs with industry-standard formatting.
            <br><br>
            <strong>Features:</strong><br>
            • Professional Schlumberger Techlog-style visualization<br>
            • Gamma Ray, Resistivity, and Density-Neutron tracks<br>
            • Optional sand shading and crossover fills<br>
            • Export to PNG, PDF, and LAS formats
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Track color legend
    st.markdown("---")
    st.markdown("### 🎨 Standard Log Colors")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Track 1: Gamma Ray**
        - 🟢 GR → Green
        """)
    
    with col2:
        st.markdown("""
        **Track 2: Resistivity**
        - 🔵 Deep → Blue
        - 🔴 Medium → Red
        - 🟠 Shallow → Orange
        """)
    
    with col3:
        st.markdown("""
        **Track 3: Density-Neutron**
        - 🔴 Density (RHOB) → Red
        - 🔵 Neutron (NPHI) → Blue (dashed)
        """)

