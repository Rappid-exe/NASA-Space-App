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
            # Try to load models compatible with our 11-feature input (5 user + 6 engineered)
            # RandomForest_Optimized is BEST for our use case!
            best_models = [
                'RandomForest_Optimized',  # 89.19% accuracy - PERFECT for our 11 features!
                'RandomForest_UserFeatures',  # 88.43% accuracy - 5 features only
            ]
            
            for model_name in best_models:
                try:
                    self.model, self.metadata = self.registry.load_model(model_name=model_name)
                    self.feature_columns = self.metadata.feature_columns
                    print(f"✅ Loaded BEST model for app: {self.metadata.model_id}")
                    print(f"  Algorithm: {self.metadata.algorithm}")
                    print(f"  Accuracy: {self.metadata.accuracy:.4f} ({self.metadata.accuracy*100:.2f}%)")
                    print(f"  F1 Score: {self.metadata.f1_score:.4f}")
                    print(f"  Features: {len(self.feature_columns)} (matches our feature engineering)")
                    return
                except Exception as e:
                    print(f"  Could not load {model_name}: {e}")
                    continue
            
            # Fallback to other models (may have feature mismatch)
            fallback_models = [
                'RandomForest_Retrained',  # 90.43% accuracy but needs 14 features
                'RandomForest_Improved',   # 90.13% accuracy but needs 14 features
                'exoplanet_classifier',    # 96.92% accuracy but needs 20 features
            ]
            
            for model_name in fallback_models:
                try:
                    self.model, self.metadata = self.registry.load_model(model_name=model_name)
                    self.feature_columns = self.metadata.feature_columns
                    print(f"⚠️  Loaded fallback model: {self.metadata.model_id}")
                    print(f"  Algorithm: {self.metadata.algorithm}")
                    print(f"  Accuracy: {self.metadata.accuracy:.4f} ({self.metadata.accuracy*100:.2f}%)")
                    print(f"  WARNING: May have feature mismatch ({len(self.feature_columns)} features expected)")
                    return
                except Exception as e:
                    print(f"  Could not load {model_name}: {e}")
                    continue
            
            # Last resort - try optimized models
            optimized_models = [
                'RandomForest_Optimized',  # 89.19% accuracy
            ]
            
            for model_name in optimized_models:
                try:
                    self.model, self.metadata = self.registry.load_model(model_name=model_name)
                    self.feature_columns = self.metadata.feature_columns
                    print(f"✅ Loaded OPTIMIZED model: {self.metadata.model_id}")
                    print(f"  Algorithm: {self.metadata.algorithm}")
                    print(f"  Accuracy: {self.metadata.accuracy:.4f} ({self.metadata.accuracy*100:.2f}%)")
                    print(f"  F1 Score: {self.metadata.f1_score:.4f}")
                    print(f"  Features: {len(self.feature_columns)} (with engineering)")
                    return
                except:
                    continue
            
            # Fallback: Try to load user-features-only models
            user_feature_models = [
                'RandomForest_UserFeatures',  # 88.43% accuracy
                'SVM_UserFeatures',           # 69.40% accuracy
                'NeuralNetwork_UserFeatures'  # 50.47% accuracy
            ]
            
            for model_name in user_feature_models:
                try:
                    self.model, self.metadata = self.registry.load_model(model_name=model_name)
                    self.feature_columns = self.metadata.feature_columns
                    print(f"✅ Loaded USER-OPTIMIZED model: {self.metadata.model_id}")
                    print(f"  Algorithm: {self.metadata.algorithm}")
                    print(f"  Accuracy: {self.metadata.accuracy:.4f} ({self.metadata.accuracy*100:.2f}%)")
                    print(f"  F1 Score: {self.metadata.f1_score:.4f}")
                    return
                except:
                    continue
            
            # Fallback: Try to load retrained models (90.43% accuracy but needs all features!)
            retrained_models = [
                'RandomForest_Retrained',  # 90.43% accuracy - BEST!
                'SVM_Retrained',           # 64.17% accuracy
                'NeuralNetwork_Retrained'  # 50.47% accuracy
            ]
            
            for model_name in retrained_models:
                try:
                    self.model, self.metadata = self.registry.load_model(model_name=model_name)
                    self.feature_columns = self.metadata.feature_columns
                    print(f"✅ Loaded RETRAINED model (multi-mission): {self.metadata.model_id}")
                    print(f"  Algorithm: {self.metadata.algorithm}")
                    print(f"  Accuracy: {self.metadata.accuracy:.4f} ({self.metadata.accuracy*100:.2f}%)")
                    print(f"  F1 Score: {self.metadata.f1_score:.4f}")
                    return
                except:
                    continue
            
            # Fallback: Try to load raw models (no normalization - easiest for inference)
            raw_models = [
                'raw_randomforest',
                'raw_neuralnetwork',
                'raw_svm'
            ]
            
            for model_name in raw_models:
                try:
                    self.model, self.metadata = self.registry.load_model(model_name=model_name)
                    self.feature_columns = self.metadata.feature_columns
                    print(f"⚠️  Loaded RAW model (fallback): {self.metadata.model_id}")
                    print(f"  Algorithm: {self.metadata.algorithm}")
                    print(f"  F1 Score: {self.metadata.f1_score:.4f}")
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
        Prepare input features for prediction with feature engineering.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            DataFrame with features in correct order
        """
        # Map frontend feature names to model feature names
        feature_mapping = {
            'orbital_period': 'period',
            'transit_duration': 'duration',
            'transit_depth': 'depth',
            'planetary_radius': 'radius',
            'equilibrium_temperature': 'temperature'
        }
        
        # Rename features to match training data
        mapped_features = {}
        for frontend_name, model_name in feature_mapping.items():
            if frontend_name in features and features[frontend_name] is not None:
                mapped_features[model_name] = features[frontend_name]
        
        # Engineer features (same as training)
        if all(k in mapped_features for k in ['depth', 'radius', 'period', 'duration', 'temperature']):
            # Depth-to-Radius ratio
            mapped_features['depth_radius_ratio'] = mapped_features['depth'] / (mapped_features['radius'] ** 2 + 1)
            
            # Period-to-Duration ratio
            mapped_features['period_duration_ratio'] = mapped_features['period'] / (mapped_features['duration'] + 0.1)
            
            # Log-scaled depth
            mapped_features['log_depth'] = np.log10(mapped_features['depth'] + 1)
            
            # Temperature zone (0=cold, 1=habitable, 2=warm, 3=hot)
            temp = mapped_features['temperature']
            if temp <= 200:
                mapped_features['temp_zone'] = 0.0
            elif temp <= 350:
                mapped_features['temp_zone'] = 1.0
            elif temp <= 1000:
                mapped_features['temp_zone'] = 2.0
            else:
                mapped_features['temp_zone'] = 3.0
            
            # Size category (0=Earth-like, 1=Super-Earth, 2=Neptune-like, 3=Jupiter-like)
            radius = mapped_features['radius']
            if radius <= 1.5:
                mapped_features['size_category'] = 0.0
            elif radius <= 2.5:
                mapped_features['size_category'] = 1.0
            elif radius <= 6:
                mapped_features['size_category'] = 2.0
            else:
                mapped_features['size_category'] = 3.0
            
            # Period category (0=ultra-short, 1=short, 2=medium, 3=long)
            period = mapped_features['period']
            if period <= 10:
                mapped_features['period_category'] = 0.0
            elif period <= 100:
                mapped_features['period_category'] = 1.0
            elif period <= 1000:
                mapped_features['period_category'] = 2.0
            else:
                mapped_features['period_category'] = 3.0
        
        # Create DataFrame with all expected features in correct order
        feature_dict = {}
        for col in self.feature_columns:
            if col in mapped_features:
                feature_dict[col] = [mapped_features[col]]
            else:
                # Use default value for missing features
                feature_dict[col] = [0.0]
        
        return pd.DataFrame(feature_dict)

    def _get_probabilities(self, X):
        """
        Get prediction probabilities, handling both sklearn and Keras models.
        
        Args:
            X: Input features (DataFrame or array)
            
        Returns:
            Array of probabilities
        """
        # Keep as DataFrame for sklearn models (they need feature names)
        # Only convert to array for Keras models
        if hasattr(self.model, 'predict_proba'):
            # scikit-learn models (RandomForest, SVM) - keep DataFrame
            return self.model.predict_proba(X)
        else:
            # Keras/TensorFlow models (Neural Network) - convert to array
            if isinstance(X, pd.DataFrame):
                X_array = X.values
            else:
                X_array = X
            
            probs = self.model.predict(X_array, verbose=0)
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
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """
        Get feature importance scores for the current model.
        
        Returns:
            Dictionary with feature names and importance scores
        """
        self._validate_model_loaded()
        
        try:
            importance_scores = {}
            
            # For tree-based models (RandomForest)
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                for i, feature in enumerate(self.feature_columns):
                    importance_scores[feature] = float(importances[i])
            
            # For SVM models with linear kernel
            elif hasattr(self.model, 'coef_'):
                # Use absolute values of coefficients as importance
                importances = np.abs(self.model.coef_[0])
                for i, feature in enumerate(self.feature_columns):
                    importance_scores[feature] = float(importances[i])
            
            # For Neural Networks, use permutation importance approximation
            else:
                # Return uniform importance for models without built-in feature importance
                uniform_importance = 1.0 / len(self.feature_columns)
                for feature in self.feature_columns:
                    importance_scores[feature] = uniform_importance
            
            # Sort by importance
            sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'algorithm': self.metadata.algorithm,
                'features': [
                    {
                        'name': name,
                        'importance': score,
                        'rank': i + 1
                    }
                    for i, (name, score) in enumerate(sorted_features)
                ],
                'top_5': [
                    {'name': name, 'importance': score}
                    for name, score in sorted_features[:5]
                ]
            }
        except Exception as e:
            raise ValueError(f"Could not extract feature importance: {str(e)}")
