"""
Unit tests for FeatureSelector class.
"""

import pytest
import numpy as np
import pandas as pd
from models.feature_selector import FeatureSelector


class TestFeatureSelector:
    """Test suite for FeatureSelector."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        # Create features with varying importance
        X = np.random.randn(n_samples, n_features)
        
        # Create target that depends on some features more than others
        # Features 0, 1, 2 are highly important
        # Features 3, 4 are moderately important
        # Rest are noise
        y = (
            2 * X[:, 0] +
            1.5 * X[:, 1] +
            1 * X[:, 2] +
            0.5 * X[:, 3] +
            0.3 * X[:, 4] +
            np.random.randn(n_samples) * 0.1
        )
        y = (y > y.mean()).astype(int)
        
        feature_names = [f"feature_{i}" for i in range(n_features)]
        
        return X, y, feature_names
    
    def test_initialization(self):
        """Test FeatureSelector initialization."""
        selector = FeatureSelector(random_state=42)
        
        assert selector.random_state == 42
        assert selector.feature_names is None
        assert selector.importance_scores is None
        assert selector.selected_features is None
    
    def test_analyze_importance(self, sample_data):
        """Test feature importance analysis."""
        X, y, feature_names = sample_data
        selector = FeatureSelector(random_state=42)
        
        importance_df = selector.analyze_importance(X, y, feature_names)
        
        # Check that importance DataFrame has correct structure
        assert isinstance(importance_df, pd.DataFrame)
        assert 'feature' in importance_df.columns
        assert 'avg_importance' in importance_df.columns
        assert 'rank' in importance_df.columns
        assert len(importance_df) == len(feature_names)
        
        # Check that importance scores are computed
        assert selector.importance_scores is not None
        assert selector.feature_names == feature_names
        
        # Check that top features are the ones we designed to be important
        top_features = importance_df.head(5)['feature'].tolist()
        # At least 3 of the top 5 should be from our important features
        important_features = ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4']
        overlap = len(set(top_features) & set(important_features))
        assert overlap >= 3, f"Expected at least 3 important features in top 5, got {overlap}"
    
    def test_select_features_by_threshold(self, sample_data):
        """Test feature selection by threshold."""
        X, y, feature_names = sample_data
        selector = FeatureSelector(random_state=42)
        
        selector.analyze_importance(X, y, feature_names)
        selected = selector.select_features(importance_threshold=0.05, min_features=5)
        
        # Check that features are selected
        assert isinstance(selected, list)
        assert len(selected) >= 5
        assert len(selected) <= len(feature_names)
        assert selector.selected_features == selected
    
    def test_select_features_top_k(self, sample_data):
        """Test feature selection by top-k."""
        X, y, feature_names = sample_data
        selector = FeatureSelector(random_state=42)
        
        selector.analyze_importance(X, y, feature_names)
        selected = selector.select_features(top_k=10)
        
        # Check that exactly 10 features are selected
        assert len(selected) == 10
        assert selector.selected_features == selected
    
    def test_get_feature_report(self, sample_data):
        """Test feature report generation."""
        X, y, feature_names = sample_data
        selector = FeatureSelector(random_state=42)
        
        selector.analyze_importance(X, y, feature_names)
        selector.select_features(top_k=10)
        
        report = selector.get_feature_report()
        
        # Check report structure
        assert isinstance(report, dict)
        assert 'total_features' in report
        assert 'selected_features' in report
        assert 'feature_rankings' in report
        assert 'top_10_features' in report
        assert 'importance_methods' in report
        
        # Check report values
        assert report['total_features'] == len(feature_names)
        assert report['selected_features'] == 10
        assert len(report['top_10_features']) == 10
    
    def test_transform(self, sample_data):
        """Test feature transformation."""
        X, y, feature_names = sample_data
        selector = FeatureSelector(random_state=42)
        
        selector.analyze_importance(X, y, feature_names)
        selector.select_features(top_k=10)
        
        X_transformed = selector.transform(X, feature_names)
        
        # Check transformed shape
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == 10
    
    def test_transform_with_dataframe(self, sample_data):
        """Test feature transformation with DataFrame input."""
        X, y, feature_names = sample_data
        X_df = pd.DataFrame(X, columns=feature_names)
        
        selector = FeatureSelector(random_state=42)
        selector.analyze_importance(X_df, y)
        selector.select_features(top_k=10)
        
        X_transformed = selector.transform(X_df)
        
        # Check transformed shape
        assert X_transformed.shape[0] == X_df.shape[0]
        assert X_transformed.shape[1] == 10
    
    def test_error_on_select_before_analyze(self):
        """Test that selecting features before analysis raises error."""
        selector = FeatureSelector()
        
        with pytest.raises(ValueError, match="Must run analyze_importance"):
            selector.select_features()
    
    def test_error_on_report_before_analyze(self):
        """Test that generating report before analysis raises error."""
        selector = FeatureSelector()
        
        with pytest.raises(ValueError, match="Must run analyze_importance"):
            selector.get_feature_report()
    
    def test_error_on_transform_before_select(self, sample_data):
        """Test that transforming before selection raises error."""
        X, y, feature_names = sample_data
        selector = FeatureSelector()
        
        with pytest.raises(ValueError, match="Must run select_features"):
            selector.transform(X, feature_names)
    
    def test_get_selected_feature_indices(self, sample_data):
        """Test getting indices of selected features."""
        X, y, feature_names = sample_data
        selector = FeatureSelector(random_state=42)
        
        selector.analyze_importance(X, y, feature_names)
        selector.select_features(top_k=5)
        
        indices = selector.get_selected_feature_indices(feature_names)
        
        # Check that indices are correct
        assert len(indices) == 5
        assert all(0 <= idx < len(feature_names) for idx in indices)
        
        # Verify that indices correspond to selected features
        for idx, feature in zip(indices, selector.selected_features):
            assert feature_names[idx] == feature


if __name__ == "__main__":
    # Run a simple test
    print("Running basic FeatureSelector test...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    y = (2 * X[:, 0] + 1.5 * X[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # Test feature selector
    selector = FeatureSelector(random_state=42)
    importance_df = selector.analyze_importance(X, y, feature_names)
    
    print("\nTop 10 features:")
    print(importance_df[['feature', 'avg_importance']].head(10))
    
    selected = selector.select_features(top_k=10)
    print(f"\nSelected {len(selected)} features")
    
    report = selector.get_feature_report()
    print(f"\nReport generated with {report['total_features']} total features")
    
    print("\n✓ Basic test passed!")
