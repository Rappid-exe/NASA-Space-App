"""
Tests for MultiDatasetLoader class.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.multi_dataset_loader import MultiDatasetLoader


class TestMultiDatasetLoader:
    """Test suite for MultiDatasetLoader."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = MultiDatasetLoader()
    
    def test_initialization(self):
        """Test loader initialization."""
        assert self.loader is not None
        assert isinstance(self.loader.loaded_datasets, dict)
        assert isinstance(self.loader.dataset_statistics, dict)
        assert len(self.loader.loaded_datasets) == 0
    
    def test_column_mappings_exist(self):
        """Test that column mappings are defined for all missions."""
        assert 'kepler' in MultiDatasetLoader.COLUMN_MAPPINGS
        assert 'tess' in MultiDatasetLoader.COLUMN_MAPPINGS
        assert 'k2' in MultiDatasetLoader.COLUMN_MAPPINGS
        
        # Check that all mappings have required fields
        required_fields = ['disposition', 'period', 'radius']
        for mission, mapping in MultiDatasetLoader.COLUMN_MAPPINGS.items():
            for field in required_fields:
                assert field in mapping, f"Missing {field} in {mission} mapping"
    
    def test_disposition_mappings_exist(self):
        """Test that disposition mappings are defined."""
        assert 'kepler' in MultiDatasetLoader.DISPOSITION_MAPPINGS
        assert 'tess' in MultiDatasetLoader.DISPOSITION_MAPPINGS
        assert 'k2' in MultiDatasetLoader.DISPOSITION_MAPPINGS
    
    def test_load_single_dataset(self):
        """Test loading a single dataset."""
        try:
            df = self.loader.load_dataset('kepler')
            assert df is not None
            assert len(df) > 0
            assert 'kepler' in self.loader.loaded_datasets
            assert 'kepler' in self.loader.dataset_statistics
        except FileNotFoundError:
            pytest.skip("Kepler dataset not available")
    
    def test_load_all_datasets(self):
        """Test loading all available datasets."""
        datasets = self.loader.load_all_datasets()
        
        # Should return a dictionary
        assert isinstance(datasets, dict)
        
        # Should have at least one dataset
        assert len(datasets) > 0
        
        # All values should be DataFrames
        for mission, df in datasets.items():
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
    
    def test_harmonize_single_dataset(self):
        """Test harmonizing a single dataset."""
        try:
            # Load Kepler dataset
            df = self.loader.load_dataset('kepler')
            
            # Harmonize it
            harmonized = self.loader.harmonize_dataset(df, 'kepler')
            
            # Check that harmonized dataset has expected columns
            assert 'disposition' in harmonized.columns
            assert 'period' in harmonized.columns
            assert 'radius' in harmonized.columns
            assert 'mission' in harmonized.columns
            
            # Check mission identifier
            assert all(harmonized['mission'] == 'KEPLER')
            
            # Check that disposition values are harmonized
            valid_dispositions = ['CONFIRMED', 'CANDIDATE', 'FALSE_POSITIVE', 'NOT_DISPOSITIONED']
            assert all(harmonized['disposition'].isin(valid_dispositions))
            
        except FileNotFoundError:
            pytest.skip("Kepler dataset not available")
    
    def test_harmonize_all_datasets(self):
        """Test harmonizing all datasets."""
        # Load all datasets
        raw_datasets = self.loader.load_all_datasets()
        
        if len(raw_datasets) == 0:
            pytest.skip("No datasets available")
        
        # Harmonize all
        harmonized = self.loader.harmonize_all_datasets(raw_datasets)
        
        # Check that all datasets were harmonized
        assert len(harmonized) == len(raw_datasets)
        
        # Check that all have the same column structure
        column_sets = [set(df.columns) for df in harmonized.values()]
        assert all(cols == column_sets[0] for cols in column_sets), "All datasets should have same columns"
    
    def test_combine_datasets(self):
        """Test combining multiple datasets."""
        try:
            # Load and combine
            combined = self.loader.combine_datasets()
            
            # Check that combined dataset exists
            assert combined is not None
            assert len(combined) > 0
            
            # Check that mission column exists
            assert 'mission' in combined.columns
            
            # Check that we have data from multiple missions
            missions = combined['mission'].unique()
            assert len(missions) > 0
            
            # Check that all required columns exist
            required_cols = ['disposition', 'period', 'radius', 'mission']
            for col in required_cols:
                assert col in combined.columns
            
        except FileNotFoundError:
            pytest.skip("Datasets not available")
    
    def test_disposition_harmonization(self):
        """Test that disposition values are properly harmonized."""
        # Create test data
        test_data = pd.DataFrame({
            'disposition': ['CONFIRMED', 'FALSE POSITIVE', 'CANDIDATE', None, 'UNKNOWN']
        })
        
        # Test Kepler harmonization
        harmonized = self.loader._harmonize_disposition(test_data['disposition'], 'kepler')
        
        assert harmonized[0] == 'CONFIRMED'
        assert harmonized[1] == 'FALSE_POSITIVE'
        assert harmonized[2] == 'CANDIDATE'
        assert harmonized[3] == 'NOT_DISPOSITIONED'  # NaN should map to NOT_DISPOSITIONED
    
    def test_get_dataset_statistics(self):
        """Test getting dataset statistics."""
        try:
            # Load a dataset
            self.loader.load_dataset('kepler')
            
            # Get statistics
            stats = self.loader.get_dataset_statistics()
            
            assert 'kepler' in stats
            assert 'total_rows' in stats['kepler']
            assert 'total_columns' in stats['kepler']
            assert stats['kepler']['total_rows'] > 0
            
        except FileNotFoundError:
            pytest.skip("Kepler dataset not available")
    
    def test_feature_coverage(self):
        """Test feature coverage analysis."""
        try:
            # Load and harmonize datasets
            raw_datasets = self.loader.load_all_datasets()
            harmonized = self.loader.harmonize_all_datasets(raw_datasets)
            
            if len(harmonized) == 0:
                pytest.skip("No datasets available")
            
            # Get feature coverage
            coverage = self.loader.get_feature_coverage(harmonized)
            
            assert coverage is not None
            assert isinstance(coverage, pd.DataFrame)
            assert len(coverage) > 0
            
        except FileNotFoundError:
            pytest.skip("Datasets not available")
    
    def test_filter_by_disposition(self):
        """Test filtering by disposition."""
        # Create test data
        test_df = pd.DataFrame({
            'disposition': ['CONFIRMED', 'FALSE_POSITIVE', 'CONFIRMED', 'CANDIDATE'],
            'period': [1.0, 2.0, 3.0, 4.0]
        })
        
        # Filter for confirmed only
        filtered = self.loader.filter_by_disposition(test_df, ['CONFIRMED'])
        
        assert len(filtered) == 2
        assert all(filtered['disposition'] == 'CONFIRMED')
    
    def test_create_binary_labels(self):
        """Test creating binary labels."""
        # Create test data
        test_df = pd.DataFrame({
            'disposition': ['CONFIRMED', 'FALSE_POSITIVE', 'CONFIRMED', 'CANDIDATE'],
            'period': [1.0, 2.0, 3.0, 4.0]
        })
        
        # Create binary labels
        labeled = self.loader.create_binary_labels(test_df)
        
        assert 'label' in labeled.columns
        assert labeled['label'][0] == 1  # CONFIRMED
        assert labeled['label'][1] == 0  # FALSE_POSITIVE
        assert labeled['label'][2] == 1  # CONFIRMED
        assert labeled['label'][3] == 0  # CANDIDATE
    
    def test_invalid_mission(self):
        """Test that invalid mission raises error."""
        with pytest.raises(ValueError):
            self.loader.load_dataset('invalid_mission')
    
    def test_missing_file(self):
        """Test that missing file raises error."""
        with pytest.raises(FileNotFoundError):
            self.loader.load_dataset('kepler', file_path='nonexistent.csv')


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
