"""
Base classifier interface for consistent model API.
All exoplanet classifiers inherit from this base class.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional
import pickle
import json
from datetime import datetime


class BaseClassifier(ABC):
    """Abstract base class for all exoplanet classifiers."""
    
    def __init__(self, model_name: str):
        """
        Initialize the base classifier.
        
        Args:
            model_name: Name identifier for the model
        """
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.training_metadata = {}
        self.feature_names = None
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Train the classifier on the provided data.
        
        Args:
            X: Training features
            y: Training labels
            **kwargs: Additional training parameters
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict
            
        Returns:
            Array of predicted class labels
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for new data.
        
        Args:
            X: Features to predict
            
        Returns:
            Array of class probabilities
        """
        pass
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores if available.
        
        Returns:
            Array of feature importance scores or None
        """
        return None
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'training_metadata': self.training_metadata,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.training_metadata = model_data['training_metadata']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata.
        
        Returns:
            Dictionary containing model metadata
        """
        return {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'training_metadata': self.training_metadata,
            'feature_count': len(self.feature_names) if self.feature_names else None
        }
    
    def set_training_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Set training metadata for the model.
        
        Args:
            metadata: Dictionary containing training information
        """
        self.training_metadata.update(metadata)
        self.training_metadata['last_updated'] = datetime.now().isoformat()
