"""
Support Vector Machine (SVM) classifier for exoplanet classification.
"""

import numpy as np
from typing import Optional
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from models.base_classifier import BaseClassifier


class SVMClassifier(BaseClassifier):
    """SVM classifier for exoplanet identification."""
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale',
                 probability: bool = True, random_state: int = 42):
        """
        Initialize SVM classifier.
        
        Args:
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            C: Regularization parameter
            gamma: Kernel coefficient
            probability: Whether to enable probability estimates
            random_state: Random seed for reproducibility
        """
        super().__init__(model_name="SVM")
        
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.probability = probability
        self.random_state = random_state
        
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=probability,
            random_state=random_state,
            cache_size=1000  # Increase cache for better performance
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Train the SVM classifier.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional training parameters
        """
        print(f"Training {self.model_name} with {self.kernel} kernel...")
        
        # Store feature names if provided
        if 'feature_names' in kwargs:
            self.feature_names = kwargs['feature_names']
        elif hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Convert to numpy array if needed
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        # Train the model
        self.model.fit(X, y)
        self.is_trained = True
        
        # Store training metadata
        self.set_training_metadata({
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'n_classes': len(np.unique(y)),
            'n_support_vectors': self.model.n_support_.tolist() if hasattr(self.model, 'n_support_') else None
        })
        
        print(f"{self.model_name} training complete!")
        if hasattr(self.model, 'n_support_'):
            print(f"Number of support vectors: {sum(self.model.n_support_)}")
    
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
        
        # Convert to numpy array if needed
        if hasattr(X, 'values'):
            X = X.values
        
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
        
        if not self.probability:
            raise ValueError("Probability estimates not enabled. Set probability=True when initializing.")
        
        # Convert to numpy array if needed
        if hasattr(X, 'values'):
            X = X.values
        
        return self.model.predict_proba(X)
    
    def get_support_vectors(self) -> Optional[np.ndarray]:
        """
        Get the support vectors.
        
        Returns:
            Array of support vectors or None
        """
        if not self.is_trained:
            return None
        
        return self.model.support_vectors_
    
    def get_support_vector_indices(self) -> Optional[np.ndarray]:
        """
        Get indices of support vectors.
        
        Returns:
            Array of support vector indices or None
        """
        if not self.is_trained:
            return None
        
        return self.model.support_
