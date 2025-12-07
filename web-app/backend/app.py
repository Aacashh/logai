"""
Flask Backend Application
REST API for Well Log Viewer
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import os
import sys
import uuid
import io

# Add shared module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from shared.las_parser import load_las, extract_header_info, detect_depth_units, get_available_curves
from shared.curve_mapping import get_curve_mapping, create_track_layout
from shared.data_processing import process_data, filter_by_depth, df_to_json, export_to_las, get_depth_range

app = Flask(__name__)
CORS(app)

# In-memory storage for uploaded wells (in production, use a database)
wells_storage = {}

# Upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'version': '1.0.0'})


@app.route('/api/upload', methods=['POST'])
def upload_las():
    """
    Upload a LAS file.
    
    Returns:
        JSON with well_id, metadata, AND initial curve data (for faster first render)
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith('.las'):
        return jsonify({'error': 'File must be a .las file'}), 400
    
    try:
        # Generate unique ID
        well_id = str(uuid.uuid4())[:8]
        
        # Save file
        filepath = os.path.join(UPLOAD_FOLDER, f"{well_id}.las")
        file.save(filepath)
        
        # Parse LAS file
        las = load_las(filepath)
        header_info = extract_header_info(las)
        depth_unit = detect_depth_units(las)
        mapping = get_curve_mapping(las)
        available_curves = get_available_curves(las)
        track_layout = create_track_layout(mapping)
        
        # Process data
        df = process_data(las, mapping)
        min_depth, max_depth = get_depth_range(df)
        
        # Store in memory
        wells_storage[well_id] = {
            'filepath': filepath,
            'filename': file.filename,
            'header': header_info,
            'depth_unit': depth_unit,
            'mapping': mapping,
            'curves': available_curves,
            'track_layout': track_layout,
            'depth_range': {'min': min_depth, 'max': max_depth},
            'df': df
        }
        
        # Get initial curve data for the first view (first 500 units or full range)
        initial_end = min(min_depth + 500, max_depth)
        initial_df = filter_by_depth(df, min_depth, initial_end)
        initial_curves = df_to_json(initial_df)
        initial_curves['depth_unit'] = depth_unit
        initial_curves['mapping'] = mapping
        
        return jsonify({
            'well_id': well_id,
            'filename': file.filename,
            'header': header_info,
            'depth_unit': depth_unit,
            'depth_range': {'min': min_depth, 'max': max_depth},
            'curves': available_curves,
            'mapping': mapping,
            'track_layout': track_layout,
            # Include initial curve data to avoid second API call
            'curve_data': initial_curves
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/wells', methods=['GET'])
def list_wells():
    """List all uploaded wells."""
    wells = []
    for well_id, data in wells_storage.items():
        wells.append({
            'well_id': well_id,
            'filename': data['filename'],
            'well_name': data['header'].get('WELL', 'Unknown'),
            'depth_range': data['depth_range']
        })
    return jsonify({'wells': wells})


@app.route('/api/wells/<well_id>', methods=['GET'])
def get_well(well_id):
    """Get well metadata."""
    if well_id not in wells_storage:
        return jsonify({'error': 'Well not found'}), 404
    
    data = wells_storage[well_id]
    return jsonify({
        'well_id': well_id,
        'filename': data['filename'],
        'header': data['header'],
        'depth_unit': data['depth_unit'],
        'depth_range': data['depth_range'],
        'curves': data['curves'],
        'mapping': data['mapping'],
        'track_layout': data['track_layout']
    })


@app.route('/api/wells/<well_id>', methods=['DELETE'])
def delete_well(well_id):
    """Delete a well."""
    if well_id not in wells_storage:
        return jsonify({'error': 'Well not found'}), 404
    
    # Delete file
    filepath = wells_storage[well_id]['filepath']
    if os.path.exists(filepath):
        os.remove(filepath)
    
    del wells_storage[well_id]
    return jsonify({'message': 'Well deleted'})


@app.route('/api/wells/<well_id>/curves', methods=['GET'])
def get_curves(well_id):
    """
    Get curve data for plotting.
    
    Query params:
        start_depth: Start depth (optional)
        end_depth: End depth (optional)
        smooth: Smoothing window (optional)
        max_points: Maximum data points to return (optional, default 5000)
    """
    if well_id not in wells_storage:
        return jsonify({'error': 'Well not found'}), 404
    
    data = wells_storage[well_id]
    df = data['df'].copy()
    
    # Filter by depth if specified
    start_depth = request.args.get('start_depth', type=float)
    end_depth = request.args.get('end_depth', type=float)
    
    if start_depth is not None and end_depth is not None:
        df = filter_by_depth(df, start_depth, end_depth)
    
    # Downsample if too many points (improves transfer and rendering performance)
    max_points = request.args.get('max_points', default=5000, type=int)
    if len(df) > max_points:
        # Use every nth sample to reduce data size while preserving shape
        step = len(df) // max_points
        df = df.iloc[::step].copy()
    
    # Convert to JSON format
    result = df_to_json(df)
    result['depth_unit'] = data['depth_unit']
    result['mapping'] = data['mapping']
    result['total_points'] = len(df)
    result['downsampled'] = len(df) < len(data['df'])
    
    return jsonify(result)


@app.route('/api/wells/<well_id>/export/png', methods=['POST'])
def export_png(well_id):
    """Export plot as PNG."""
    if well_id not in wells_storage:
        return jsonify({'error': 'Well not found'}), 404
    
    # TODO: Implement server-side plot generation
    # For now, return a placeholder response
    return jsonify({'message': 'PNG export - implement client-side or add matplotlib'}), 501


@app.route('/api/wells/<well_id>/export/pdf', methods=['POST'])
def export_pdf(well_id):
    """Export plot as PDF."""
    if well_id not in wells_storage:
        return jsonify({'error': 'Well not found'}), 404
    
    # TODO: Implement server-side PDF generation
    return jsonify({'message': 'PDF export - implement with reportlab'}), 501


@app.route('/api/wells/<well_id>/export/las', methods=['GET'])
def export_las(well_id):
    """Export curves as LAS file."""
    if well_id not in wells_storage:
        return jsonify({'error': 'Well not found'}), 404
    
    data = wells_storage[well_id]
    
    # Get depth range from query params
    start_depth = request.args.get('start_depth', type=float)
    end_depth = request.args.get('end_depth', type=float)
    
    df = data['df'].copy()
    if start_depth is not None and end_depth is not None:
        df = filter_by_depth(df, start_depth, end_depth)
    
    # Export to LAS
    las_content = export_to_las(df, data['header'], data['depth_unit'])
    
    # Return as file download
    buffer = io.BytesIO(las_content.encode('utf-8'))
    buffer.seek(0)
    
    filename = f"{data['header'].get('WELL', 'export')}_processed.las"
    return send_file(
        buffer,
        mimetype='text/plain',
        as_attachment=True,
        download_name=filename
    )


# ============================================
# FUTURE PLACEHOLDERS
# ============================================

@app.route('/api/wells/compare', methods=['POST'])
def compare_wells():
    """
    FUTURE: Compare multiple wells side-by-side.
    """
    return jsonify({
        'message': 'Multi-well comparison - coming in future version',
        'status': 'not_implemented'
    }), 501


@app.route('/api/wells/<well_id>/zones', methods=['GET', 'POST'])
def well_zones(well_id):
    """
    FUTURE: Zonal shading and markers.
    """
    return jsonify({
        'message': 'Zonal shading - coming in future version',
        'status': 'not_implemented'
    }), 501


@app.route('/api/wells/<well_id>/lithology', methods=['GET', 'POST'])
def well_lithology(well_id):
    """
    FUTURE: Lithology track data.
    """
    return jsonify({
        'message': 'Lithology track - coming in future version',
        'status': 'not_implemented'
    }), 501


if __name__ == '__main__':
    app.run(debug=True, port=5000)
