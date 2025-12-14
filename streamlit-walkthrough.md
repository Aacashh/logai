# WellLog Analyzer Pro - Complete Technical Documentation

## Table of Contents

1. [Application Overview](#1-application-overview)
2. [Architecture & File Structure](#2-architecture--file-structure)
3. [Main Entry Point (`app.py`)](#3-main-entry-point-apppy)
4. [Log Viewer Page](#4-log-viewer-page)
5. [AI Splicing Page](#5-ai-splicing-page)
6. [Auto Splicer Page](#6-auto-splicer-page)
7. [Shared Modules](#7-shared-modules)
8. [Plotting Engine](#8-plotting-engine)
9. [Technical Algorithms](#9-technical-algorithms)
10. [Dependencies](#10-dependencies)

---

## 1. Application Overview

**WellLog Analyzer Pro** is a professional-grade well log analysis suite built with Streamlit, designed for petrophysicists and geoscientists. It provides:

| Feature | Description |
|---------|-------------|
| **Log Viewer** | Industry-standard Schlumberger Techlog-style LAS file visualization |
| **AI Splicing** | Educational "glass box" splicing with step-by-step algorithm explanation |
| **Auto Splicer** | Multi-well batch processing with automatic unit conversion |

**Version:** 1.1.0  
**Input Format:** LAS 2.0 files (`.las`, `.LAS`)  
**Output Formats:** PNG, PDF, LAS, CSV

---

## 2. Architecture & File Structure

```
streamlit-app/
├── app.py                    # Main entry point with welcome screen
├── plotting.py               # Matplotlib plotting engine
├── requirements.txt          # Python dependencies
├── run.sh                    # Startup script
└── pages/
    ├── 01_Log_Viewer.py      # Log visualization page
    ├── 02_AI_Splicing.py     # Educational two-file splicing
    └── 02_Auto_Splicer.py    # Multi-file batch splicing

shared/
├── __init__.py
├── las_parser.py             # LAS file loading & parsing
├── curve_mapping.py          # Automatic curve mnemonic mapping
├── data_processing.py        # Data transformation & export
└── splicing.py               # Core splicing algorithms (Cross-Correlation + DTW)
```

---

## 3. Main Entry Point (`app.py`)

### Purpose
The landing page that welcomes users and provides navigation to the available tools.

### Features
- **Professional UI Styling**: Custom CSS with Inter font family, gradient backgrounds, hover effects
- **Feature Cards**: Visual descriptions of Log Viewer and AI Splicing capabilities
- **Quick Start Guide**: Step-by-step instructions for new users
- **Sidebar Navigation**: Version info and tool descriptions

### Page Configuration
```python
st.set_page_config(
    page_title="WellLog Analyzer Pro",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

### Styling Highlights
- Gradient header: `linear-gradient(135deg, #1a1d24 0%, #2d3748 100%)`
- Accent color: `#00D4AA` (teal/cyan)
- Hidden Streamlit branding for professional appearance

---

## 4. Log Viewer Page

**File:** `pages/01_Log_Viewer.py`

### Capabilities

#### 4.1 File Upload & Parsing
- Accepts LAS 2.0 format files via `st.file_uploader`
- Uses `lasio` library for robust LAS parsing
- Handles encoding issues with UTF-8 fallback

#### 4.2 Automatic Curve Detection
The system automatically maps curve mnemonics to standard track types:

| Track Type | Detected Mnemonics |
|------------|-------------------|
| **GR** (Gamma Ray) | `GR`, `GRC`, `GR_EDTC`, `SGR`, `CGR`, `HCGR` |
| **RES_DEEP** (Deep Resistivity) | `RT`, `RDEP`, `RLLD`, `RLL3`, `ILD`, `RILD`, `LLD`, `RD` |
| **RES_MED** (Medium Resistivity) | `RM`, `ILM`, `RILS`, `RLM`, `RILM` |
| **RES_SHAL** (Shallow Resistivity) | `RS`, `MSFL`, `RXOZ`, `RLL1`, `SFL`, `LLS`, `RSFL` |
| **DENS** (Bulk Density) | `RHOB`, `RHOZ`, `DEN`, `ZDEN`, `BDEL` |
| **NEUT** (Neutron Porosity) | `NPHI`, `TNPH`, `NPHZ`, `CNPOR`, `NPOR`, `TNPHI` |
| **DEPTH** | `DEPT`, `DEPTH`, `DPTH`, `MD`, `TVD` |

#### 4.3 Well Header Display
Extracts and displays metadata:
- Well Name, Field, Location, Operator
- Depth Range (Start/Stop/Step)
- Logging Date, UWI/API numbers

#### 4.4 Plot Settings (Sidebar Controls)

**Depth Scale Options:**
- Preset scales: `1:200`, `1:500`, `1:1000`
- Custom scale input (100-5000)

**Data Processing:**
- **Smoothing Window**: Moving average filter (0-20 samples, center-weighted)
- Raw data display toggle

**Visualization Options:**
- **GR Sand Shading**: Fills from curve to left edge indicating sand vs shale
- **Density-Neutron Crossover**: Yellow fill for gas effect, grey for shale

**Track Limits (Configurable):**

| Track | Default Min | Default Max | Unit |
|-------|-------------|-------------|------|
| Gamma Ray | 0 | 150 | API |
| Resistivity | 0.2 | 2000 | ohm.m |
| Density | 1.95 | 2.95 | g/cc |
| Neutron | -0.15 | 0.45 | v/v |

#### 4.5 Depth Navigation
- Slider for selecting depth interval
- Default view: First 500 meters from start
- Protection against oversized plots (max 65,000 pixels)

#### 4.6 Three-Track Visualization

**Track 1: Gamma Ray**
- Linear scale
- Green color (`#00AA00`)
- Optional sand shading fill

**Track 2: Resistivity**
- Logarithmic scale (0.2 - 2000 ohm.m)
- Three curves: Deep (Blue), Medium (Red), Shallow (Orange)
- Log-spaced grid lines

**Track 3: Density-Neutron**
- Dual-axis overlay
- Density: Red solid line, normal scale
- Neutron: Blue dashed line, reversed scale
- Crossover fill for lithology/gas indication

#### 4.7 Export Options
| Format | Resolution | Use Case |
|--------|------------|----------|
| PNG | 150 DPI | Quick sharing |
| PNG (High-Res) | 300 DPI | Print quality |
| PDF | 150 DPI | Documents |
| LAS | N/A | Data exchange |

---

## 5. AI Splicing Page

**File:** `pages/02_AI_Splicing.py`

### Purpose
An educational "glass box" interface that merges two overlapping logging runs (shallow and deep) while explaining each algorithmic step.

### 5.1 User Interface Sections

#### Section 1: Data Intake
- Two file uploaders: Reference (Shallow) and Target (Deep)
- File metadata display: Well name, depth range, curve count
- Common curve detection and correlation curve selection
- Recommended curve priority: `GR > RHOB > NPHI`

#### Section 2: Algorithm Execution (Glass Box)
Real-time display of each processing step with explanatory boxes:

**Step 1: Preprocessing & Grid Alignment**
- Creates common depth grid (default: 0.1524m / 0.5ft step)
- Resamples both signals via linear interpolation
- Applies Z-Score normalization: `(x - μ) / σ`
- NaN filling with 0 for correlation math only

**Step 2: Global Shift Detection**
- Cross-correlation within ±20m window (configurable)
- Displays bulk shift in meters with direction
- Explains why: Corrects depth encoder drift

**Step 3: Elastic Correction (Constrained DTW)**
- Sakoe-Chiba band constraint (default ±5m)
- Shows DTW cost metric
- Displays overlap length and max elastic correction
- Explains why: Cable stretch isn't constant

**Step 4: Final Splice & Merge**
- Splice point at overlap midpoint
- Takes shallow log above, deep log below
- Shows merged sample count

#### Section 3: QC Visualization
Three-track professional QC plot:
1. **BEFORE**: Raw Shallow (Green) vs Raw Deep (Red)
2. **AFTER**: Raw Shallow (Green) vs Corrected Deep (Blue)
3. **CORRECTIONS**: Depth delta curve showing applied shifts

### 5.2 Sidebar Parameters

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| Max Search Window | 20.0m | 5-50m | Cross-correlation search range |
| Max Elastic Stretch | 5.0m | 1-20m | DTW Sakoe-Chiba band width |
| Grid Resolution | 0.1524m | 0.0762/0.1524/0.3048m | Resampling step |

### 5.3 Export Options
- QC Plot: PNG, PDF
- Merged LAS: Single-curve output

---

## 6. Auto Splicer Page

**File:** `pages/02_Auto_Splicer.py`

### Purpose
A "black box" batch processing interface for multiple wells with automatic unit conversion and intelligent file grouping.

### 6.1 Multi-File Upload
- Accepts unlimited LAS files simultaneously
- Files from different wells are automatically grouped

### 6.2 Automatic Processing Pipeline

**Stage 1: Well Detection & Grouping**
- Reads `WELL` header field from each file
- Sanitizes well names (handles case, whitespace)
- Groups files by well name
- Detects and skips duplicate files using fingerprint (filename + size + STRT + STOP)

**Stage 2: Unit Detection & Conversion**
- Detects units from `STRT.FT`, `STRT.M`, or curve headers
- Automatically converts feet to meters: `depth_m = depth_ft × 0.3048`
- Logs conversion actions

**Stage 3: Data Cleaning**
- Strips null padding from top/bottom of logs
- Replaces null values (`-999.25`, `-999`, `-9999`, etc.) with NaN

**Stage 4: Chain Splicing**
For each pair of consecutive files (sorted by start depth):
- **Gap > 5m**: Append with NaN fill (no correlation needed)
- **Overlap**: Apply correlation + DTW splicing algorithm

### 6.3 Well Selection Interface
When multiple wells detected:
- Dropdown selector with file count per well
- Metadata display: File count, depth range, location
- Expandable file table with unit badges

### 6.4 Results Display

**Success Banner:**
- Well name, files merged, depth range

**Metrics Dashboard:**
- Files Merged
- Total Depth (m)
- Data Points
- Number of Curves

**Processing Summary Table:**
- Per-file details: filename, original unit, depth range, action taken

**Composite Log Plot:**
- Up to 4 tracks based on available curves
- Priority: GR > RHOB > NPHI > RT > DT > CALI > SP > PEF
- Dark theme styling matching page aesthetic

### 6.5 Export Options
| Format | Filename Pattern |
|--------|------------------|
| LAS | `{well_name}_composite.las` |
| PNG | `{well_name}_composite.png` |
| CSV | `{well_name}_composite.csv` |

---

## 7. Shared Modules

### 7.1 LAS Parser (`shared/las_parser.py`)

**`load_las(file_obj)`**
- Handles file paths, bytes, and file-like objects
- UTF-8 decoding with error tolerance

**`extract_header_info(las)`**
Returns dictionary with keys:
```python
{
    'WELL', 'FIELD', 'LOC', 'COMP', 'STRT', 'STOP', 'STEP',
    'CTRY', 'SRVC', 'DATE', 'UWI', 'API'
}
```

**`detect_depth_units(las)`**
- Checks `STRT.unit`, then `STEP.unit`, then curve definitions
- Returns `'m'` or `'ft'`

**`get_available_curves(las)`**
Returns list of dictionaries:
```python
[{
    'mnemonic': str,
    'unit': str,
    'description': str,
    'data_range': {'min': float, 'max': float}
}]
```

### 7.2 Curve Mapping (`shared/curve_mapping.py`)

**`CURVE_MNEMONICS`** dictionary maps standard types to possible mnemonics.

**`get_curve_mapping(las)`**
Auto-detects curves present in file, returns:
```python
{
    'DEPTH': 'DEPT',
    'GR': 'GR',
    'RES_DEEP': 'RT',
    'RES_MED': None,
    'RES_SHAL': 'MSFL',
    'DENS': 'RHOB',
    'NEUT': 'NPHI',
    # ... etc
}
```

**`TRACK_CONFIG`** defines display properties:
- Track number, name, unit
- Color, scale (linear/log)
- Default min/max values

### 7.3 Data Processing (`shared/data_processing.py`)

**`handle_null_values(df)`**
Replaces standard null values with NaN:
```python
NULL_VALUES = [-999.25, -999, -9999, -9999.25, -999.2500, -999.00]
```

**`apply_smoothing(df, window)`**
- Center-weighted moving average
- Preserves depth column

**`process_data(las, mapping, smooth_window)`**
Complete pipeline:
1. Convert LAS to DataFrame
2. Standardize depth column name
3. Handle null values
4. Apply optional smoothing

**`get_auto_scale(df, column, default_min, default_max)`**
Expands limits if data exceeds defaults (10% margin).

**`export_to_las(df, header_info, depth_unit)`**
Creates LAS 2.0 format string output.

**`df_to_json(df)`**
Converts DataFrame to API-friendly format with NaN→None conversion.

### 7.4 Splicing Module (`shared/splicing.py`)

**Constants:**
```python
DEFAULT_GRID_STEP = 0.1524     # meters (0.5 feet)
DEFAULT_SEARCH_WINDOW = 20.0   # meters
DEFAULT_DTW_WINDOW = 5.0       # meters
FT_TO_M = 0.3048               # conversion factor
GAP_THRESHOLD = 5.0            # meters
```

**Data Classes:**
- `SplicingResult`: Merged data, metrics, correction curves
- `PreprocessedSignal`: Normalized signal data
- `PreprocessedLAS`: Unit-converted LAS file
- `WellGroupResult`: Grouped files by well
- `BatchSpliceResult`: Composite log with metadata

---

## 8. Plotting Engine

**File:** `streamlit-app/plotting.py`

### Industry-Standard Color Palette

```python
COLORS = {
    'GR': '#00AA00',           # Green
    'GR_FILL': '#90EE90',      # Light green for sand
    'RES_DEEP': '#0066CC',     # Blue
    'RES_MED': '#CC0000',      # Red
    'RES_SHAL': '#FF8C00',     # Orange
    'DENS': '#CC0000',         # Red
    'NEUT': '#0066CC',         # Blue
    'CROSS_GAS': '#FFFF00',    # Yellow
    'CROSS_SHALE': '#808080',  # Grey
}
```

### `create_log_plot(df, mapping, settings, show_gr_fill, show_dn_fill)`

Creates 3-track Matplotlib figure:

**Figure Sizing:**
```python
height_cm = (depth_range * 100) / scale_ratio
height_in = max(5, height_cm / 2.54)
figsize = (12, height_in)
```

**Track 1: Gamma Ray (`_plot_gamma_ray`)**
- Linear X-axis (API units)
- X-label at top with green coloring
- Optional fill from curve to left edge

**Track 2: Resistivity (`_plot_resistivity`)**
- Logarithmic X-axis
- `LogLocator` for proper tick marks
- Legend for multiple curves

**Track 3: Density-Neutron (`_plot_density_neutron`)**
- Primary axis: Density (left-to-right)
- Secondary axis: Neutron (reversed, right-to-left)
- Offset secondary spine for clarity
- Neutron transformed to density scale for overlay

### Crossover Fill Logic

```python
def _add_crossover_fill(ax, dens_data, neut_transformed, depth):
    # Gas effect: neutron > density (transformed)
    ax.fill_betweenx(..., color='#FFFF00', alpha=0.4)  # Yellow
    
    # Shale: density > neutron (transformed)
    ax.fill_betweenx(..., color='#808080', alpha=0.3)  # Grey
```

### `export_plot_to_bytes(fig, format, dpi)`

Exports figure to bytes buffer for download:
- PNG: Default, RGBA output
- JPEG: RGB with quality=95
- PDF: Vector format

---

## 9. Technical Algorithms

### 9.1 Z-Score Normalization

Removes tool calibration bias between runs:

```python
normalized = (signal - mean) / std
```

Where `mean` and `std` are computed ignoring NaN values.

### 9.2 Global Cross-Correlation

Finds optimal bulk shift between two signals:

```
correlation(τ) = Σ reference(t) × target(t + τ)
```

**Implementation:**
```python
correlation = scipy.signal.correlate(reference, target, mode='full')
lags = np.arange(-(n-1), n)  # Lag array in samples
```

- Search limited to ±`max_search_meters` window
- Peak location indicates optimal shift
- Positive shift = target needs to move shallower

### 9.3 Constrained DTW (Sakoe-Chiba Band)

Dynamic programming algorithm for elastic alignment:

**Recurrence relation:**
```python
D[i,j] = (x[i] - y[j])² + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
```

**Sakoe-Chiba constraint:**
```python
j_min = max(1, i - window_size)
j_max = min(c, i + window_size)
# Only fill cells within band
```

**Why constrained?**
- Standard DTW can warp signals unrealistically
- Band forces path to stay near diagonal
- Preserves geological features

**Backtracking:**
Follows minimum-cost path from `D[r,c]` back to `D[0,0]` to extract warp path.

### 9.4 Chain Splicing Algorithm

For multiple files sorted by start depth:

```
For each pair of consecutive files:
    gap = next_start - composite_end
    
    if gap > 5m:
        → Append with NaN-filled gap
    elif overlap:
        → Run correlation + DTW
        → Apply bulk shift to deep log
        → Splice at overlap midpoint
    else (edge touch):
        → Direct concatenation
```

### 9.5 Neutron-to-Density Transform

For overlay on same axis:

```python
def _transform_neutron_to_density(neut_data, neut_min, neut_max, dens_min, dens_max):
    normalized = (neut_data - neut_min) / (neut_max - neut_min)
    reversed_norm = 1 - normalized  # Reverse for overlay
    return dens_min + reversed_norm * (dens_max - dens_min)
```

---

## 10. Dependencies

**File:** `requirements.txt`

```
streamlit        # Web application framework
lasio            # LAS file parsing
pandas           # Data manipulation
numpy            # Numerical operations
matplotlib       # Plotting
scipy            # Signal processing (cross-correlation)
```

### Key Library Versions (Recommended)
- Python 3.8+
- Streamlit 1.28+
- lasio 0.31+
- matplotlib 3.7+
- scipy 1.10+

---

## Appendix: Quick Reference

### Default Track Limits

| Track | Min | Max | Scale |
|-------|-----|-----|-------|
| GR | 0 API | 150 API | Linear |
| Resistivity | 0.2 ohm.m | 2000 ohm.m | Log |
| Density | 1.95 g/cc | 2.95 g/cc | Linear |
| Neutron | -0.15 v/v | 0.45 v/v | Linear (reversed) |

### Color Codes

| Element | Hex Code | Description |
|---------|----------|-------------|
| GR Curve | `#00AA00` | Green |
| GR Sand Fill | `#90EE90` | Light Green |
| Deep Resistivity | `#0066CC` | Blue |
| Medium Resistivity | `#CC0000` | Red |
| Shallow Resistivity | `#FF8C00` | Orange |
| Density | `#CC0000` | Red |
| Neutron | `#0066CC` | Blue (dashed) |
| Gas Crossover | `#FFFF00` | Yellow |
| Shale Crossover | `#808080` | Grey |

### Splicing Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| Grid Step | 0.1524m (0.5ft) | Resampling resolution |
| Search Window | ±20m | Cross-correlation range |
| DTW Window | ±5m | Elastic stretch limit |
| Gap Threshold | 5m | Append vs splice decision |

---

