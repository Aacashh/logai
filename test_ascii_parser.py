"""Test script for ASCII parser functionality."""
import sys
sys.path.insert(0, '.')

from shared.ascii_parser import load_ascii_file, validate_ascii_data

# Create test CSV data
test_csv = '''DEPTH,GR,RHOB,NPHI
100.0,45.2,2.35,0.15
100.5,48.1,2.38,0.14
101.0,52.3,2.40,0.13
101.5,55.7,2.42,0.12
102.0,50.1,2.37,0.14
'''

# Test parsing
df, meta, depth = load_ascii_file(test_csv.encode('utf-8'))

print('=== Format Detection ===')
print(f'Delimiter: {meta.detected_delimiter}')
print(f'Columns: {meta.column_names}')
print(f'Rows: {meta.num_data_rows}')
print(f'Confidence: {meta.confidence_score:.2f}')

print('\n=== Depth Detection ===')
print(f'Depth Column: {depth.column_name}')
print(f'Method: {depth.detection_method}')
print(f'Range: {depth.depth_range}')
print(f'Unit: {depth.detected_unit}')

print('\n=== Validation ===')
valid = validate_ascii_data(df, depth)
print(f'Valid: {valid["is_valid"]}')
if valid['issues']:
    print(f'Issues: {valid["issues"]}')

print('\n=== DataFrame ===')
print(df)

print('\n✅ SUCCESS!')
