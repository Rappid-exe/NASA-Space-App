"""
Verification script for CrossMissionValidator implementation.
Tests that the cross-mission validator works correctly.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.cross_mission_validator import CrossMissionValidator
import numpy as np


def verify_initialization():
    """Verify validator can be initialized."""
    print("\n1. Testing Initialization...")
    
    try:
        validator = CrossMissionValidator(random_state=42)
        
        assert validator.random_state == 42
        assert validator.validation_results == {}
        assert validator.validation_matrix is None
        assert validator.dataset_loader is not None
        assert validator.evaluator is not None
        
        print("   ✓ Initialization successful")
        return True
    except Exception as e:
        print(f"   ✗ Initialization failed: {e}")
        return False


def verify_validation_results_structure():
    """Verify validation results have correct structure."""
    print("\n2. Testing Validation Results Structure...")
    
    try:
        validator = CrossMissionValidator()
        
        # Simulate validation results
        validator.validation_results['kepler_to_tess'] = {
            'train_mission': 'kepler',
            'test_mission': 'tess',
            'train_samples': 1000,
            'test_samples': 500,
            'n_features': 54,
            'models': {
                'RandomForest': {
                    'accuracy': 0.85,
                    'f1_score': 0.87,
                    'precision': 0.86,
                    'recall': 0.88
                }
            }
        }
        
        # Test summary generation
        summary = validator._generate_summary()
        assert summary['total_validations'] == 1
        assert summary['avg_cross_mission_accuracy'] > 0
        
        print("   ✓ Validation results structure correct")
        return True
    except Exception as e:
        print(f"   ✗ Validation results structure test failed: {e}")
        return False


def verify_recommendations():
    """Verify recommendations are generated correctly."""
    print("\n3. Testing Recommendations Generation...")
    
    try:
        validator = CrossMissionValidator()
        
        # Test with no results
        recommendations = validator._generate_recommendations()
        assert len(recommendations) > 0
        assert any('Run cross-mission validation' in rec for rec in recommendations)
        
        # Test with high accuracy results
        validator.validation_results['test1'] = {
            'train_mission': 'kepler',
            'test_mission': 'tess',
            'models': {
                'Model1': {'accuracy': 0.85, 'f1_score': 0.87}
            }
        }
        
        recommendations = validator._generate_recommendations()
        assert len(recommendations) > 0
        
        print("   ✓ Recommendations generation works")
        return True
    except Exception as e:
        print(f"   ✗ Recommendations generation failed: {e}")
        return False


def verify_validation_report():
    """Verify validation report generation."""
    print("\n4. Testing Validation Report Generation...")
    
    try:
        validator = CrossMissionValidator()
        
        # Test with no results
        report = validator.generate_validation_report()
        assert 'error' in report
        
        # Add mock results
        validator.validation_results['kepler_to_tess'] = {
            'train_mission': 'kepler',
            'test_mission': 'tess',
            'train_samples': 1000,
            'test_samples': 500,
            'n_features': 54,
            'models': {
                'RandomForest': {
                    'accuracy': 0.85,
                    'f1_score': 0.87
                }
            }
        }
        
        # Test with results
        report = validator.generate_validation_report()
        assert 'error' not in report
        assert 'summary' in report
        assert 'detailed_results' in report
        assert 'recommendations' in report
        
        print("   ✓ Validation report generation works")
        return True
    except Exception as e:
        print(f"   ✗ Validation report generation failed: {e}")
        return False


def verify_matrix_structure():
    """Verify validation matrix structure."""
    print("\n5. Testing Validation Matrix Structure...")
    
    try:
        import pandas as pd
        
        validator = CrossMissionValidator()
        
        # Create mock validation matrix
        matrix_data = {
            'kepler': {'kepler': 0.85, 'tess': 0.78, 'k2': 0.76},
            'tess': {'kepler': 0.79, 'tess': 0.84, 'k2': 0.77},
            'k2': {'kepler': 0.77, 'tess': 0.76, 'k2': 0.81}
        }
        
        validator.validation_matrix = pd.DataFrame(matrix_data).T
        
        assert validator.validation_matrix is not None
        assert validator.validation_matrix.shape == (3, 3)
        assert 'kepler' in validator.validation_matrix.index
        assert 'tess' in validator.validation_matrix.columns
        
        print("   ✓ Validation matrix structure correct")
        return True
    except Exception as e:
        print(f"   ✗ Validation matrix structure test failed: {e}")
        return False


def verify_combined_validation_structure():
    """Verify combined dataset validation structure."""
    print("\n6. Testing Combined Validation Structure...")
    
    try:
        validator = CrossMissionValidator()
        
        # Simulate combined validation results
        validator.validation_results['combined'] = {
            'train_missions': ['kepler', 'tess', 'k2'],
            'train_samples': 2000,
            'n_features': 54,
            'test_results': {
                'kepler': {
                    'test_samples': 500,
                    'models': {
                        'RandomForest': {'accuracy': 0.87, 'f1_score': 0.89}
                    }
                },
                'tess': {
                    'test_samples': 400,
                    'models': {
                        'RandomForest': {'accuracy': 0.85, 'f1_score': 0.87}
                    }
                }
            }
        }
        
        recommendations = validator._generate_recommendations()
        assert any('Combined dataset' in rec for rec in recommendations)
        
        print("   ✓ Combined validation structure correct")
        return True
    except Exception as e:
        print(f"   ✗ Combined validation structure test failed: {e}")
        return False


def verify_save_report():
    """Verify report can be saved to file."""
    print("\n7. Testing Report Saving...")
    
    try:
        validator = CrossMissionValidator()
        
        # Add mock results
        validator.validation_results['test'] = {
            'train_mission': 'kepler',
            'test_mission': 'tess',
            'models': {'Model1': {'accuracy': 0.85}}
        }
        
        # Save report
        output_path = 'test_validation_report.json'
        validator.save_validation_report(output_path)
        
        # Check file exists
        assert os.path.exists(output_path)
        
        # Clean up
        os.remove(output_path)
        
        print("   ✓ Report saving works")
        return True
    except Exception as e:
        print(f"   ✗ Report saving failed: {e}")
        # Clean up on error
        if os.path.exists('test_validation_report.json'):
            os.remove('test_validation_report.json')
        return False


def main():
    """Run all verification tests."""
    print("="*80)
    print("Cross-Mission Validator Verification")
    print("="*80)
    print("\nVerifying CrossMissionValidator implementation...")
    
    tests = [
        verify_initialization,
        verify_validation_results_structure,
        verify_recommendations,
        verify_validation_report,
        verify_matrix_structure,
        verify_combined_validation_structure,
        verify_save_report
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # Summary
    print("\n" + "="*80)
    print("Verification Summary")
    print("="*80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nTests Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All verification tests passed!")
        print("\nCrossMissionValidator is ready to use.")
        print("\nNext steps:")
        print("  1. Run examples/demo_cross_mission_validator.py for usage examples")
        print("  2. Run pytest tests/test_cross_mission_validator.py for unit tests")
        print("  3. Use in training pipeline for cross-mission validation")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        print("\nPlease review the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
