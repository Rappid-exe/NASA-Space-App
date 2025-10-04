"""
Random Forest classifier for ensemble-based exoplanet classification.
"""

import numpy as np
from typing import Optional
from sklearn.ensemble import RandomForestClassifier as SKRandomForest
from models.base_classifier import BaseClassifier


class RandomForestClassifier(BaseClassifier):
    """Random Forest classifier for exoplanet identification."""
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 random_state: int = 42):
        """
        Initialize Random Forest classifier.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at leaf node
            random_state: Random seed for reproducibility
        """
        super().__init__(model_name="RandomForest")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        
        self.model = SKRandomForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Train the Random Forest classifier.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional training parameters
        """
        print(f"Training {self.model_name} with {self.n_estimators} trees...")
        
        # Store feature names if provided
        if 'feature_names' in kwargs:
            self.feature_names = kwargs['feature_names']
        elif hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Train the model
        self.model.fit(X, y)
        self.is_trained = True
        
        # Store training metadata
        self.set_training_metadata({
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'n_classes': len(np.unique(y))
        })
        
        print(f"{self.model_name} training complete!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict
            
        Returns:
            Array of predicted class labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for new data.
        
        Args:
            X: Features to predict
            
        Returns:
            Array of class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores from the Random Forest.
        
        Returns:
            Array of feature importance scores
        """
        if not self.is_trained:
            return None
        
        return self.model.feature_importances_
    
    def get_feature_importance_dict(self) -> dict:
        """
        Get feature importance as a dictionary with feature names.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or self.feature_names is None:
            return {}
        
        importances = self.get_feature_importance()
        return dict(zip(self.feature_names, importances))
