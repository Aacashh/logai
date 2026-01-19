"""Comprehensive test script for curve identification."""
import lasio
import traceback
import sys
sys.path.insert(0, '.')

from shared.curve_identifier import identify_curves_layered, SCORE_WEIGHTS

# Load sample LAS file
las = lasio.read('deliverables/1055929125.las')
print(f'Loaded LAS with {len(las.curves)} curves')
print(f'Score weights: {SCORE_WEIGHTS}')
print()

try:
    report = identify_curves_layered(las)
    print('=' * 60)
    print('CURVE IDENTIFICATION REPORT')
    print('=' * 60)
    print(f'Overall confidence: {report.overall_confidence:.1%}')
    print(f'Tool context: {report.tool_context.value}')
    print()
    print('CURVE MAPPING:')
    print('-' * 40)
    for curve_type, mnemonic in report.mapping.items():
        result = report.curve_results.get(mnemonic)
        if result:
            print(f'  {curve_type:12} -> {mnemonic:12}')
            print(f'      Score: {result.raw_score:.0f}/100, Confidence: {result.confidence_score:.0%}')
            breakdown = ', '.join(f'{k}:{v:.0f}' for k,v in result.score_breakdown.items() if v > 0)
            print(f'      Breakdown: {breakdown}')
    
    if report.duplicate_warnings:
        print()
        print('DUPLICATE WARNINGS:')
        for w in report.duplicate_warnings[:3]:
            print(f'  - {w}')
    
    if report.cross_curve_insights:
        print()
        print('CROSS-CURVE INSIGHTS:')
        for i in report.cross_curve_insights:
            print(f'  - {i}')
    
    print()
    print('=' * 60)
    print('SUCCESS! All tests passed.')
    
except Exception as e:
    print(f'ERROR: {type(e).__name__}: {e}')
    traceback.print_exc()
    sys.exit(1)
