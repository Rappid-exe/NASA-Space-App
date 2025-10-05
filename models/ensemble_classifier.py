"""
Ensemble classifier that combines predictions from multiple models.
Uses soft voting with weighted averaging for improved accuracy.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from models.base_classifier import BaseClassifier


class EnsembleClassifier(BaseClassifier):
    """
    Ensemble classifier that combines multiple models using soft voting.
    
    Combines predictions from RandomForest, Neural Network, and SVM models
    to achieve better accuracy and robustness through weighted averaging.
    """
    
    def __init__(self, models: List[BaseClassifier], weights: List[float] = None):
        """
        Initialize the ensemble classifier.
        
        Args:
            models: List of trained BaseClassifier instances
            weights: Optional weights for each model (default: equal weights)
        """
        super().__init__(model_name="EnsembleClassifier")
        
        if not models:
            raise ValueError("At least one model is required for ensemble")
        
        # Validate all models are trained
        for model in models:
            if not model.is_trained:
                raise ValueError(f"Model {model.model_name} is not trained")
        
        self.models = models
        
        # Set weights (equal by default)
        if weights is None:
            self.weights = np.ones(len(models)) / len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            # Normalize weights to sum to 1
            self.weights = np.array(weights) / np.sum(weights)
        
        self.is_trained = True
        self.model_names = [model.model_name for model in models]
        
        # Set metadata
        self.set_training_metadata({
            'ensemble_size': len(models),
            'model_names': self.model_names,
            'weights': self.weights.tolist()
        })
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Ensemble doesn't train - it combines already trained models.
        
        Args:
            X: Training features (not used)
            y: Training labels (not used)
            **kwargs: Additional parameters (not used)
        
        Raises:
            NotImplementedError: Ensemble uses pre-trained models
        """
        raise NotImplementedError(
            "EnsembleClassifier combines pre-trained models. "
            "Train individual models before creating ensemble."
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using weighted soft voting.
        
        Args:
            X: Features to predict
            
        Returns:
            Array of predicted class labels
        """
        # Get probability predictions from all models
        probabilities = self.predict_proba(X)
        
        # Return class with highest probability
        return np.argmax(probabilities, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using weighted averaging.
        
        Args:
            X: Features to predict
            
        Returns:
            Array of class probabilities (weighted average from all models)
        """
        # Collect probability predictions from all models
        all_probas = []
        for model in self.models:
            probas = model.predict_proba(X)
            all_probas.append(probas)
        
        # Stack probabilities: shape (n_models, n_samples, n_classes)
        all_probas = np.array(all_probas)
        
        # Apply weighted average across models
        # weights shape: (n_models, 1, 1) for broadcasting
        weights_reshaped = self.weights.reshape(-1, 1, 1)
        weighted_probas = all_probas * weights_reshaped
        
        # Sum across models (axis=0)
        ensemble_probas = np.sum(weighted_probas, axis=0)
        
        return ensemble_probas
    
    def get_model_agreement(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate agreement score between models for each prediction.
        
        Agreement is measured as the proportion of models that agree with
        the ensemble prediction.
        
        Args:
            X: Features to predict
            
        Returns:
            Array of agreement scores (0-1) for each sample
        """
        # Get ensemble predictions
        ensemble_preds = self.predict(X)
        
        # Get individual model predictions
        model_preds = []
        for model in self.models:
            preds = model.predict(X)
            model_preds.append(preds)
        
        # Stack predictions: shape (n_models, n_samples)
        model_preds = np.array(model_preds)
        
        # Calculate agreement for each sample
        agreement_scores = []
        for i in range(X.shape[0]):
            # Count how many models agree with ensemble
            ensemble_pred = ensemble_preds[i]
            model_votes = model_preds[:, i]
            agreement = np.mean(model_votes == ensemble_pred)
            agreement_scores.append(agreement)
        
        return np.array(agreement_scores)
    
    def get_individual_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get predictions from each individual model.
        
        Args:
            X: Features to predict
            
        Returns:
            Dictionary mapping model names to their predictions
        """
        predictions = {}
        for model in self.models:
            predictions[model.model_name] = model.predict(X)
        
        return predictions
    
    def get_individual_confidences(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get confidence scores from each individual model.
        
        Confidence is the maximum probability for each prediction.
        
        Args:
            X: Features to predict
            
        Returns:
            Dictionary mapping model names to their confidence scores
        """
        confidences = {}
        for model in self.models:
            probas = model.predict_proba(X)
            # Confidence is max probability
            confidence = np.max(probas, axis=1)
            confidences[model.model_name] = confidence
        
        return confidences
    
    def get_detailed_prediction(self, X: np.ndarray) -> List[Dict[str, Any]]:
        """
        Get detailed prediction information for each sample.
        
        Args:
            X: Features to predict
            
        Returns:
            List of dictionaries containing detailed prediction info
        """
        ensemble_preds = self.predict(X)
        ensemble_probas = self.predict_proba(X)
        agreement_scores = self.get_model_agreement(X)
        individual_preds = self.get_individual_predictions(X)
        individual_confs = self.get_individual_confidences(X)
        
        results = []
        for i in range(X.shape[0]):
            # Get ensemble confidence (max probability)
            ensemble_conf = np.max(ensemble_probas[i])
            
            # Collect individual model info
            model_info = {}
            for model_name in self.model_names:
                model_info[model_name] = {
                    'prediction': int(individual_preds[model_name][i]),
                    'confidence': float(individual_confs[model_name][i])
                }
            
            result = {
                'ensemble_prediction': int(ensemble_preds[i]),
                'ensemble_confidence': float(ensemble_conf),
                'agreement_score': float(agreement_scores[i]),
                'model_predictions': model_info
            }
            results.append(result)
        
        return results
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get ensemble metadata including model information.
        
        Returns:
            Dictionary containing ensemble metadata
        """
        base_metadata = super().get_metadata()
        base_metadata.update({
            'ensemble_size': len(self.models),
            'model_names': self.model_names,
            'weights': self.weights.tolist()
        })
        return base_metadata
