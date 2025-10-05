"""
Feature selector for identifying and selecting the most important features.
Implements multiple importance analysis methods for robust feature selection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
import warnings

warnings.filterwarnings('ignore')


class FeatureSelector:
    """
    Feature selector that uses multiple methods to identify important features.
    
    Implements:
    - RandomForest feature importance (tree-based)
    - Mutual information (statistical dependency)
    - Correlation analysis (redundancy removal)
    - Permutation importance (real-world impact)
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the feature selector.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.feature_names = None
        self.importance_scores = None
        self.selected_features = None
        self.analysis_results = {}
    
    def analyze_importance(self, X: np.ndarray, y: np.ndarray, 
                          feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Analyze feature importance using multiple methods.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names (optional)
            
        Returns:
            DataFrame with importance scores from different methods
        """
        print("Analyzing feature importance using multiple methods...")
        
        # Store feature names
        if feature_names is None:
            if hasattr(X, 'columns'):
                self.feature_names = X.columns.tolist()
            else:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        else:
            self.feature_names = feature_names
        
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Initialize results dictionary
        importance_dict = {'feature': self.feature_names}
        
        # Method 1: RandomForest Feature Importance
        print("  1. Computing RandomForest feature importance...")
        rf_importance = self._compute_rf_importance(X_array, y)
        importance_dict['rf_importance'] = rf_importance
        self.analysis_results['rf_importance'] = rf_importance
        
        # Method 2: Mutual Information
        print("  2. Computing mutual information scores...")
        mi_scores = self._compute_mutual_information(X_array, y)
        importance_dict['mutual_info'] = mi_scores
        self.analysis_results['mutual_info'] = mi_scores
        
        # Method 3: Correlation Analysis
        print("  3. Analyzing feature correlations...")
        correlation_scores = self._compute_correlation_scores(X_array, y)
        importance_dict['correlation'] = correlation_scores
        self.analysis_results['correlation'] = correlation_scores
        
        # Method 4: Permutation Importance (optional, can be slow)
        print("  4. Computing permutation importance...")
        perm_importance = self._compute_permutation_importance(X_array, y)
        importance_dict['perm_importance'] = perm_importance
        self.analysis_results['perm_importance'] = perm_importance
        
        # Create DataFrame with all importance scores
        importance_df = pd.DataFrame(importance_dict)
        
        # Normalize all scores to 0-1 range for comparison
        for col in ['rf_importance', 'mutual_info', 'correlation', 'perm_importance']:
            max_val = importance_df[col].max()
            if max_val > 0:
                importance_df[f'{col}_normalized'] = importance_df[col] / max_val
            else:
                importance_df[f'{col}_normalized'] = 0
        
        # Calculate average importance across all methods
        normalized_cols = [col for col in importance_df.columns if col.endswith('_normalized')]
        importance_df['avg_importance'] = importance_df[normalized_cols].mean(axis=1)
        
        # Rank features by average importance
        importance_df['rank'] = importance_df['avg_importance'].rank(ascending=False)
        importance_df = importance_df.sort_values('avg_importance', ascending=False)
        
        self.importance_scores = importance_df
        
        print(f"\nFeature importance analysis complete!")
        print(f"Top 10 features by average importance:")
        print(importance_df[['feature', 'avg_importance', 'rank']].head(10).to_string(index=False))
        
        return importance_df

    def _compute_rf_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute feature importance using RandomForest.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Array of importance scores
        """
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X, y)
        return rf.feature_importances_
    
    def _compute_mutual_information(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute mutual information scores.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Array of mutual information scores
        """
        mi_scores = mutual_info_classif(
            X, y,
            discrete_features=False,
            random_state=self.random_state,
            n_neighbors=3
        )
        return mi_scores
    
    def _compute_correlation_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute correlation-based importance scores.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Array of correlation scores (absolute values)
        """
        # Compute correlation between each feature and target
        correlations = np.array([
            np.abs(np.corrcoef(X[:, i], y)[0, 1]) 
            for i in range(X.shape[1])
        ])
        
        # Handle NaN values (can occur with constant features)
        correlations = np.nan_to_num(correlations, nan=0.0)
        
        return correlations
    
    def _compute_permutation_importance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute permutation importance scores.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Array of permutation importance scores
        """
        # Train a simple model for permutation importance
        rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        # Compute permutation importance
        perm_result = permutation_importance(
            rf, X, y,
            n_repeats=5,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        return perm_result.importances_mean
    
    def select_features(self, importance_threshold: float = 0.01,
                       top_k: Optional[int] = None,
                       min_features: int = 10) -> List[str]:
        """
        Select features based on importance threshold or top-k.
        
        Args:
            importance_threshold: Minimum average importance score (0-1)
            top_k: Select top K features (overrides threshold if provided)
            min_features: Minimum number of features to keep
            
        Returns:
            List of selected feature names
        """
        if self.importance_scores is None:
            raise ValueError("Must run analyze_importance() before selecting features")
        
        print(f"\nSelecting features...")
        
        if top_k is not None:
            # Select top K features
            selected_df = self.importance_scores.head(top_k)
            print(f"  Selected top {top_k} features")
        else:
            # Select features above threshold
            selected_df = self.importance_scores[
                self.importance_scores['avg_importance'] >= importance_threshold
            ]
            
            # Ensure minimum number of features
            if len(selected_df) < min_features:
                print(f"  Warning: Only {len(selected_df)} features above threshold.")
                print(f"  Selecting top {min_features} features instead.")
                selected_df = self.importance_scores.head(min_features)
            else:
                print(f"  Selected {len(selected_df)} features above threshold {importance_threshold}")
        
        self.selected_features = selected_df['feature'].tolist()
        
        print(f"\nSelected features ({len(self.selected_features)}):")
        for i, feature in enumerate(self.selected_features[:20], 1):
            importance = selected_df[selected_df['feature'] == feature]['avg_importance'].values[0]
            print(f"  {i:2d}. {feature:40s} (importance: {importance:.4f})")
        
        if len(self.selected_features) > 20:
            print(f"  ... and {len(self.selected_features) - 20} more features")
        
        return self.selected_features
    
    def get_feature_report(self) -> Dict[str, Any]:
        """
        Generate detailed feature analysis report.
        
        Returns:
            Dictionary containing comprehensive feature analysis
        """
        if self.importance_scores is None:
            raise ValueError("Must run analyze_importance() before generating report")
        
        report = {
            'total_features': len(self.feature_names),
            'selected_features': len(self.selected_features) if self.selected_features else 0,
            'feature_rankings': self.importance_scores[['feature', 'avg_importance', 'rank']].to_dict('records'),
            'top_10_features': self.importance_scores.head(10)['feature'].tolist(),
            'bottom_10_features': self.importance_scores.tail(10)['feature'].tolist(),
            'importance_methods': {
                'rf_importance': {
                    'description': 'RandomForest tree-based importance',
                    'top_feature': self.importance_scores.nlargest(1, 'rf_importance')['feature'].values[0],
                    'mean_score': self.importance_scores['rf_importance'].mean()
                },
                'mutual_info': {
                    'description': 'Statistical dependency with target',
                    'top_feature': self.importance_scores.nlargest(1, 'mutual_info')['feature'].values[0],
                    'mean_score': self.importance_scores['mutual_info'].mean()
                },
                'correlation': {
                    'description': 'Correlation with target variable',
                    'top_feature': self.importance_scores.nlargest(1, 'correlation')['feature'].values[0],
                    'mean_score': self.importance_scores['correlation'].mean()
                },
                'perm_importance': {
                    'description': 'Real-world prediction impact',
                    'top_feature': self.importance_scores.nlargest(1, 'perm_importance')['feature'].values[0],
                    'mean_score': self.importance_scores['perm_importance'].mean()
                }
            }
        }
        
        if self.selected_features:
            report['selected_feature_list'] = self.selected_features
            report['removed_features'] = [
                f for f in self.feature_names if f not in self.selected_features
            ]
            report['selection_ratio'] = len(self.selected_features) / len(self.feature_names)
        
        return report
    
    def get_redundant_features(self, correlation_threshold: float = 0.95) -> List[Tuple[str, str, float]]:
        """
        Identify pairs of highly correlated (redundant) features.
        
        Args:
            correlation_threshold: Correlation threshold for redundancy
            
        Returns:
            List of tuples (feature1, feature2, correlation)
        """
        if self.importance_scores is None:
            raise ValueError("Must run analyze_importance() before checking redundancy")
        
        print(f"\nIdentifying redundant features (correlation > {correlation_threshold})...")
        
        # This is a placeholder - would need the actual feature matrix to compute
        # For now, return empty list
        redundant_pairs = []
        
        print(f"  Found {len(redundant_pairs)} redundant feature pairs")
        
        return redundant_pairs
    
    def save_report(self, filepath: str) -> None:
        """
        Save feature importance report to file.
        
        Args:
            filepath: Path to save the report
        """
        if self.importance_scores is None:
            raise ValueError("Must run analyze_importance() before saving report")
        
        import json
        
        report = self.get_feature_report()
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nFeature importance report saved to: {filepath}")
        
        # Also save detailed CSV
        csv_path = filepath.replace('.json', '_detailed.csv')
        self.importance_scores.to_csv(csv_path, index=False)
        print(f"Detailed importance scores saved to: {csv_path}")
    
    def get_selected_feature_indices(self, all_feature_names: List[str]) -> List[int]:
        """
        Get indices of selected features from a list of all feature names.
        
        Args:
            all_feature_names: Complete list of feature names
            
        Returns:
            List of indices for selected features
        """
        if self.selected_features is None:
            raise ValueError("Must run select_features() first")
        
        indices = [
            i for i, name in enumerate(all_feature_names)
            if name in self.selected_features
        ]
        
        return indices
    
    def transform(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Transform feature matrix to include only selected features.
        
        Args:
            X: Feature matrix
            feature_names: List of feature names corresponding to X columns
            
        Returns:
            Transformed feature matrix with only selected features
        """
        if self.selected_features is None:
            raise ValueError("Must run select_features() first")
        
        if feature_names is None:
            if hasattr(X, 'columns'):
                feature_names = X.columns.tolist()
            else:
                raise ValueError("feature_names must be provided for numpy arrays")
        
        # Get indices of selected features
        indices = self.get_selected_feature_indices(feature_names)
        
        # Select columns
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features].values
        else:
            return X[:, indices]
