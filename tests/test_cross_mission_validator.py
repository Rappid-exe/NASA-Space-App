"""
Unit tests for CrossMissionValidator.
"""

import pytest
import numpy as np
import pandas as pd
from models.cross_mission_validator import CrossMissionValidator


class TestCrossMissionValidator:
    """Test suite for CrossMissionValidator."""
    
    def test_initialization(self):
        """Test validator initialization."""
        validator = CrossMissionValidator(random_state=42)
        
        assert validator.random_state == 42
        assert validator.validation_results == {}
        assert validator.validation_matrix is None
        assert validator.dataset_loader is not None
        assert validator.evaluator is not None
    
    def test_generate_summary_empty(self):
        """Test summary generation with no results."""
        validator = CrossMissionValidator()
        summary = validator._generate_summary()
        
        assert summary['total_validations'] == 0
        assert summary['avg_cross_mission_accuracy'] == 0.0
        assert summary['best_pair'] is None
        assert summary['worst_pair'] is None
    
    def test_generate_recommendations_empty(self):
        """Test recommendations with no results."""
        validator = CrossMissionValidator()
        recommendations = validator._generate_recommendations()
        
        assert len(recommendations) > 0
        assert any('Run cross-mission validation' in rec for rec in recommendations)
    
    def test_generate_validation_report_empty(self):
        """Test report generation with no results."""
        validator = CrossMissionValidator()
        report = validator.generate_validation_report()
        
        assert 'error' in report
        assert report['error'] == 'No validation results available'
    
    def test_validation_results_structure(self):
        """Test validation results structure."""
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
        
        summary = validator._generate_summary()
        
        assert summary['total_validations'] == 1
        assert summary['avg_cross_mission_accuracy'] > 0
    
    def test_validation_matrix_structure(self):
        """Test validation matrix structure."""
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
    
    def test_recommendations_based_on_accuracy(self):
        """Test recommendations based on different accuracy levels."""
        validator = CrossMissionValidator()
        
        # High accuracy scenario
        validator.validation_results['test1'] = {
            'train_mission': 'kepler',
            'test_mission': 'tess',
            'models': {
                'Model1': {'accuracy': 0.85, 'f1_score': 0.87}
            }
        }
        
        recommendations = validator._generate_recommendations()
        assert any('Excellent' in rec or 'Good' in rec for rec in recommendations)
        
        # Low accuracy scenario
        validator.validation_results = {}
        validator.validation_results['test2'] = {
            'train_mission': 'kepler',
            'test_mission': 'tess',
            'models': {
                'Model1': {'accuracy': 0.65, 'f1_score': 0.67}
            }
        }
        
        recommendations = validator._generate_recommendations()
        assert any('Low' in rec or 'Moderate' in rec for rec in recommendations)
    
    def test_generate_validation_report_with_results(self):
        """Test report generation with results."""
        validator = CrossMissionValidator()
        
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
        
        report = validator.generate_validation_report()
        
        assert 'error' not in report
        assert 'summary' in report
        assert 'detailed_results' in report
        assert 'recommendations' in report
        assert report['summary']['total_validations'] == 1
    
    def test_combined_validation_results(self):
        """Test combined dataset validation results structure."""
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
