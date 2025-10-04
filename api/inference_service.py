"""
Inference service for exoplanet classification.
Handles model loading, prediction, and result formatting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.model_registry import ModelRegistry, ModelMetadata


class InferenceService:
    """Service for performing inference with trained exoplanet classification models."""
    
    def __init__(self, model_id: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize the inference service.
        
        Args:
            model_id: Specific model ID to load
            model_name: Model name to load (loads best version if not specified)
        """
        self.registry = ModelRegistry()
        self.model = None
        self.metadata: Optional[ModelMetadata] = None
        self.feature_columns = None
        
        # Load model if specified
        if model_id or model_name:
            self.load_model(model_id=model_id, model_name=model_name)
        else:
            # Try to load the best available model
            try:
                self._load_best_model()
            except Exception as e:
                print(f"Warning: Could not load default model: {e}")
    
    def _load_best_model(self) -> None:
        """Load the best performing model from the registry."""
        try:
            # Try to load raw models first (no normalization - easiest for inference)
            raw_models = [
                'raw_randomforest',
                'raw_neuralnetwork',
                'raw_svm'
            ]
            
            for model_name in raw_models:
                try:
                    self.model, self.metadata = self.registry.load_model(model_name=model_name)
                    self.feature_columns = self.metadata.feature_columns
                    print(f"✅ Loaded RAW model (no normalization): {self.metadata.model_id}")
                    print(f"  Algorithm: {self.metadata.algorithm}")
                    print(f"  F1 Score: {self.metadata.f1_score:.4f}")
                    print(f"  Training Dataset: {self.metadata.training_dataset}")
                    return
                except:
                    continue
            
            # Try to load simplified models (may need normalization)
            simple_models = [
                'simple_randomforest',
                'simple_neuralnetwork',
                'simple_svm'
            ]
            
            for model_name in simple_models:
                try:
                    self.model, self.metadata = self.registry.load_model(model_name=model_name)
                    self.feature_columns = self.metadata.feature_columns
                    print(f"⚠️  Loaded simplified model (may need normalization): {self.metadata.model_id}")
                    print(f"  Algorithm: {self.metadata.algorithm}")
                    print(f"  F1 Score: {self.metadata.f1_score:.4f}")
                    return
                except:
                    continue
            
            # Try Kepler models (may have feature mismatch)
            kepler_models = [
                'kepler_randomforest',
                'kepler_neuralnetwork', 
                'kepler_svm'
            ]
            
            for model_name in kepler_models:
                try:
                    self.model, self.metadata = self.registry.load_model(model_name=model_name)
                    self.feature_columns = self.metadata.feature_columns
                    print(f"⚠️  Loaded Kepler model (may have feature mismatch): {self.metadata.model_id}")
                    print(f"  Algorithm: {self.metadata.algorithm}")
                    print(f"  F1 Score: {self.metadata.f1_score:.4f}")
                    return
                except:
                    continue
            
            # Fallback to best model if no specific models found
            self.model, self.metadata = self.registry.get_best_model(metric='f1_score')
            self.feature_columns = self.metadata.feature_columns
            print(f"Loaded best available model: {self.metadata.model_id}")
        except Exception as e:
            print(f"No models available in registry: {e}")
    
    def load_model(self, model_id: Optional[str] = None, 
                   model_name: Optional[str] = None,
                   version: Optional[int] = None) -> None:
        """
        Load a specific model from the registry.
        
        Args:
            model_id: Specific model ID to load
            model_name: Model name to load
            version: Specific version to load
        """
        self.model, self.metadata = self.registry.load_model(
            model_id=model_id,
            model_name=model_name,
            version=version
        )
        self.feature_columns = self.metadata.feature_columns
        print(f"Loaded model: {self.metadata.model_id}")
    
    def _validate_model_loaded(self) -> None:
        """Validate that a model is loaded."""
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
    
    def _prepare_features(self, features: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare input features for prediction.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            DataFrame with features in correct order
        """
        # Map frontend feature names to model feature names
        feature_mapping = {
            'orbital_period': 'koi_period',
            'transit_duration': 'koi_duration',
            'transit_depth': 'koi_depth',
            'planetary_radius': 'koi_prad',
            'equilibrium_temperature': 'koi_teq'
        }
        
        # Rename features to match training data
        mapped_features = {}
        for frontend_name, model_name in feature_mapping.items():
            if frontend_name in features and features[frontend_name] is not None:
                mapped_features[model_name] = features[frontend_name]
        
        # Create derived features using mapped names
        derived_features = self._create_derived_features(mapped_features)
        
        # Combine original and derived features
        all_features = {**mapped_features, **derived_features}
        
        # Create DataFrame with all expected features in correct order
        feature_dict = {}
        for col in self.feature_columns:
            if col in all_features:
                feature_dict[col] = [all_features[col]]
            else:
                # Use default value for missing features
                feature_dict[col] = [0.0]
        
        return pd.DataFrame(feature_dict)
    
    def _create_derived_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create derived features from input features.
        
        Args:
            features: Dictionary of input features (with koi_ prefix)
            
        Returns:
            Dictionary of derived features
        """
        derived = {}
        
        # Period-Duration Ratio
        if 'koi_period' in features and 'koi_duration' in features:
            derived['period_duration_ratio'] = (
                features['koi_period'] / (features['koi_duration'] / 24.0)
            )
        
        # Depth-Radius Correlation
        if 'koi_depth' in features and 'koi_prad' in features:
            derived['depth_radius_correlation'] = (
                features['koi_depth'] / (features['koi_prad'] ** 2)
            )
        
        # Temperature-based features
        if 'koi_teq' in features and features['koi_teq'] is not None:
            temp = features['koi_teq']
            derived['temp_habitable_zone'] = 1 if 200 <= temp <= 350 else 0
        else:
            derived['temp_habitable_zone'] = 0
        
        # Period categories (encoded as numeric)
        if 'koi_period' in features:
            period = features['koi_period']
            if period <= 10:
                derived['period_category'] = 0  # ultra_short
            elif period <= 100:
                derived['period_category'] = 1  # short
            elif period <= 1000:
                derived['period_category'] = 2  # medium
            else:
                derived['period_category'] = 3  # long
        
        # Radius categories (encoded as numeric)
        if 'koi_prad' in features:
            radius = features['koi_prad']
            if radius <= 1.5:
                derived['radius_category'] = 0  # earth_like
            elif radius <= 2.5:
                derived['radius_category'] = 1  # super_earth
            elif radius <= 6:
                derived['radius_category'] = 2  # neptune_like
            else:
                derived['radius_category'] = 3  # jupiter_like
        
        # Transit SNR
        if 'koi_depth' in features:
            derived['transit_snr'] = np.log10(features['koi_depth'] + 1)
        
        return derived
    
    def _get_probabilities(self, X):
        """
        Get prediction probabilities, handling both sklearn and Keras models.
        
        Args:
            X: Input features
            
        Returns:
            Array of probabilities
        """
        if hasattr(self.model, 'predict_proba'):
            # scikit-learn models (RandomForest, SVM)
            return self.model.predict_proba(X)
        else:
            # Keras/TensorFlow models (Neural Network)
            probs = self.model.predict(X, verbose=0)
            # Ensure it's 2D with probabilities for both classes
            if probs.shape[1] == 1:
                # Binary classification with single output
                prob_positive = probs[:, 0]
                prob_negative = 1 - prob_positive
                return np.column_stack([prob_negative, prob_positive])
            return probs
    
    def classify_observation(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a single exoplanet observation.
        
        Args:
            features: Dictionary of astronomical features
            
        Returns:
            Dictionary with prediction, confidence, and explanation
        """
        self._validate_model_loaded()
        
        # Prepare features
        X = self._prepare_features(features)
        
        # Get probabilities (works for both sklearn and Keras models)
        # Keep as DataFrame to preserve feature names for sklearn
        probabilities = self._get_probabilities(X)[0]
        
        # Make prediction based on probabilities
        prediction = int(probabilities[1] > 0.5)  # 1 if prob of CONFIRMED > 0.5
        
        # Format results
        class_labels = ['FALSE_POSITIVE', 'CONFIRMED']
        prediction_label = class_labels[prediction]
        confidence = float(probabilities[prediction])
        
        # Create probability dictionary
        prob_dict = {
            'FALSE_POSITIVE': float(probabilities[0]),
            'CONFIRMED': float(probabilities[1])
        }
        
        # Generate explanation
        explanation = self._generate_explanation(features, prediction_label, confidence)
        
        return {
            'prediction': prediction_label,
            'confidence': confidence,
            'probabilities': prob_dict,
            'explanation': explanation
        }
    
    def classify_batch(self, observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify multiple observations in batch.
        
        Args:
            observations: List of feature dictionaries
            
        Returns:
            List of classification results
        """
        self._validate_model_loaded()
        
        results = []
        for obs in observations:
            result = self.classify_observation(obs)
            results.append(result)
        
        return results
    
    def _generate_explanation(self, features: Dict[str, Any], 
                            prediction: str, confidence: float) -> str:
        """
        Generate human-readable explanation for the prediction.
        
        Args:
            features: Input features
            prediction: Predicted class
            confidence: Confidence score
            
        Returns:
            Explanation string
        """
        explanation_parts = []
        
        # Base prediction
        if prediction == 'CONFIRMED':
            explanation_parts.append(
                f"This observation is classified as a CONFIRMED exoplanet with {confidence:.1%} confidence."
            )
        else:
            explanation_parts.append(
                f"This observation is classified as a FALSE POSITIVE with {confidence:.1%} confidence."
            )
        
        # Add feature-based insights
        if 'planetary_radius' in features:
            radius = features['planetary_radius']
            if radius < 1.5:
                explanation_parts.append("The planetary radius suggests an Earth-like planet.")
            elif radius < 2.5:
                explanation_parts.append("The planetary radius suggests a Super-Earth.")
            elif radius < 6:
                explanation_parts.append("The planetary radius suggests a Neptune-like planet.")
            else:
                explanation_parts.append("The planetary radius suggests a Jupiter-like planet.")
        
        if 'orbital_period' in features:
            period = features['orbital_period']
            if period < 10:
                explanation_parts.append("The orbital period is very short, indicating a hot planet close to its star.")
            elif period > 365:
                explanation_parts.append("The orbital period is long, similar to or exceeding Earth's year.")
        
        if 'equilibrium_temperature' in features and features['equilibrium_temperature'] is not None:
            temp = features['equilibrium_temperature']
            if 200 <= temp <= 350:
                explanation_parts.append("The equilibrium temperature falls within the habitable zone range.")
        
        return " ".join(explanation_parts)
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """
        Get current model statistics.
        
        Returns:
            Dictionary with model metadata and performance metrics
        """
        self._validate_model_loaded()
        
        return {
            'model_id': self.metadata.model_id,
            'model_name': self.metadata.model_name,
            'algorithm': self.metadata.algorithm,
            'version': self.metadata.version,
            'accuracy': self.metadata.accuracy,
            'precision': self.metadata.precision,
            'recall': self.metadata.recall,
            'f1_score': self.metadata.f1_score,
            'training_date': self.metadata.training_date,
            'training_samples': self.metadata.training_samples,
            'test_samples': self.metadata.test_samples
        }
