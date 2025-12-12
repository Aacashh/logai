"""
WellLog Analyzer Pro - Main Entry Point
Professional LAS file analysis and visualization suite.
"""

import streamlit as st

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="WellLog Analyzer Pro",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1a1d24 0%, #2d3748 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #3d4852;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00D4AA;
        margin: 0;
    }
    
    .main-subtitle {
        font-size: 1rem;
        color: #8892a0;
        margin-top: 0.5rem;
    }
    
    .feature-card {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 25px;
        border-radius: 10px;
        border: 2px solid #dee2e6;
        height: 100%;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 15px;
    }
    
    .feature-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a1d24;
        margin-bottom: 10px;
    }
    
    .feature-desc {
        font-size: 0.95rem;
        color: #495057;
        line-height: 1.5;
    }
    
    .badge-new {
        background: #00D4AA;
        color: #1a1d24;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-left: 8px;
        vertical-align: middle;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Main Header
st.markdown("""
<div class="main-header">
    <div class="main-title">🛢️ WellLog Analyzer Pro</div>
    <div class="main-subtitle">Professional Well Log Analysis Suite • LAS File Visualization & Processing</div>
</div>
""", unsafe_allow_html=True)

# Welcome message
st.markdown("""
### Welcome to WellLog Analyzer Pro

Select a tool from the sidebar to get started. This suite provides professional-grade well log 
analysis capabilities with industry-standard visualization and processing tools.
""")

st.markdown("---")

# Feature cards
st.markdown("### 🧰 Available Tools")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <div class="feature-title">Log Viewer</div>
        <div class="feature-desc">
            Professional LAS file visualization with industry-standard Techlog-style formatting.
            <br><br>
            <strong>Features:</strong>
            <ul>
                <li>Gamma Ray, Resistivity, Density-Neutron tracks</li>
                <li>Configurable scales and track limits</li>
                <li>Sand shading and crossover fills</li>
                <li>Export to PNG, PDF, and LAS</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🔗</div>
        <div class="feature-title">AI Splicing <span class="badge-new">NEW</span></div>
        <div class="feature-desc">
            Automated merging of overlapping logging runs using AI-powered alignment.
            <br><br>
            <strong>Features:</strong>
            <ul>
                <li>Global cross-correlation for bulk shift detection</li>
                <li>Constrained DTW for elastic correction</li>
                <li>Educational "glass box" step-by-step display</li>
                <li>Professional QC plotting</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Quick start guide
st.markdown("""
### 🚀 Quick Start

1. **Use the sidebar** to navigate between tools
2. **Upload LAS files** using the file upload widgets
3. **Configure settings** using the sidebar controls
4. **Export results** in your preferred format

### 📚 Supported File Formats

- **Input:** LAS 2.0 files (.las, .LAS)
- **Output:** PNG, PDF, LAS

### 💡 Tips

- For best visualization results, use the 1:500 scale for detailed analysis
- Enable "Density-Neutron Crossover" to highlight gas effects
- Use the AI Splicing tool when merging shallow and deep logging runs
""")

# Sidebar info
with st.sidebar:
    st.markdown("### 🛢️ WellLog Analyzer Pro")
    st.markdown("---")
    st.markdown("""
    **Navigation**
    
    Use the pages above to access different tools.
    
    ---
    
    **About**
    
    Professional well log analysis suite for petrophysicists and geoscientists.
    
    Version: 1.1.0
    """)
