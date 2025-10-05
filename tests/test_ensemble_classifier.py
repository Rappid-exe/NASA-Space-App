"""
Tests for the EnsembleClassifier.
"""

import pytest
import numpy as np
from models.ensemble_classifier import EnsembleClassifier
from models.base_classifier import BaseClassifier


class MockClassifier(BaseClassifier):
    """Mock classifier for testing."""
    
    def __init__(self, model_name: str, predictions: np.ndarray, probabilities: np.ndarray):
        super().__init__(model_name)
        self.mock_predictions = predictions
        self.mock_probabilities = probabilities
        self.is_trained = True
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.mock_predictions[:X.shape[0]]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.mock_probabilities[:X.shape[0]]


class TestEnsembleClassifier:
    """Test suite for EnsembleClassifier."""
    
    def test_initialization_with_models(self):
        """Test ensemble initialization with trained models."""
        # Create mock models
        model1 = MockClassifier(
            "Model1",
            np.array([0, 1, 0]),
            np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
        )
        model2 = MockClassifier(
            "Model2",
            np.array([0, 1, 0]),
            np.array([[0.7, 0.3], [0.4, 0.6], [0.85, 0.15]])
        )
        
        ensemble = EnsembleClassifier([model1, model2])
        
        assert ensemble.is_trained
        assert len(ensemble.models) == 2
        assert ensemble.model_names == ["Model1", "Model2"]
        assert np.allclose(ensemble.weights, [0.5, 0.5])
    
    def test_initialization_with_custom_weights(self):
        """Test ensemble initialization with custom weights."""
        model1 = MockClassifier(
            "Model1",
            np.array([0, 1]),
            np.array([[0.8, 0.2], [0.3, 0.7]])
        )
        model2 = MockClassifier(
            "Model2",
            np.array([0, 1]),
            np.array([[0.7, 0.3], [0.4, 0.6]])
        )
        
        # Custom weights (will be normalized)
        ensemble = EnsembleClassifier([model1, model2], weights=[2, 1])
        
        # Weights should be normalized to sum to 1
        assert np.allclose(ensemble.weights, [2/3, 1/3])
    
    def test_initialization_empty_models(self):
        """Test that initialization fails with no models."""
        with pytest.raises(ValueError, match="At least one model is required"):
            EnsembleClassifier([])
    
    def test_initialization_untrained_model(self):
        """Test that initialization fails with untrained models."""
        model = MockClassifier("Model1", np.array([0]), np.array([[0.8, 0.2]]))
        model.is_trained = False
        
        with pytest.raises(ValueError, match="is not trained"):
            EnsembleClassifier([model])
    
    def test_initialization_weight_mismatch(self):
        """Test that initialization fails when weights don't match models."""
        model1 = MockClassifier("Model1", np.array([0]), np.array([[0.8, 0.2]]))
        model2 = MockClassifier("Model2", np.array([0]), np.array([[0.7, 0.3]]))
        
        with pytest.raises(ValueError, match="Number of weights must match"):
            EnsembleClassifier([model1, model2], weights=[0.5])
    
    def test_predict_proba_equal_weights(self):
        """Test probability prediction with equal weights."""
        model1 = MockClassifier(
            "Model1",
            np.array([0, 1]),
            np.array([[0.8, 0.2], [0.3, 0.7]])
        )
        model2 = MockClassifier(
            "Model2",
            np.array([0, 1]),
            np.array([[0.6, 0.4], [0.4, 0.6]])
        )
        
        ensemble = EnsembleClassifier([model1, model2])
        X = np.array([[1, 2], [3, 4]])
        
        probas = ensemble.predict_proba(X)
        
        # Expected: average of model probabilities
        expected = np.array([
            [(0.8 + 0.6) / 2, (0.2 + 0.4) / 2],  # [0.7, 0.3]
            [(0.3 + 0.4) / 2, (0.7 + 0.6) / 2]   # [0.35, 0.65]
        ])
        
        assert np.allclose(probas, expected)
    
    def test_predict_proba_weighted(self):
        """Test probability prediction with custom weights."""
        model1 = MockClassifier(
            "Model1",
            np.array([0]),
            np.array([[0.8, 0.2]])
        )
        model2 = MockClassifier(
            "Model2",
            np.array([0]),
            np.array([[0.6, 0.4]])
        )
        
        # Model1 has 2x weight of Model2
        ensemble = EnsembleClassifier([model1, model2], weights=[2, 1])
        X = np.array([[1, 2]])
        
        probas = ensemble.predict_proba(X)
        
        # Expected: weighted average (2/3 * model1 + 1/3 * model2)
        expected = np.array([
            [(2/3 * 0.8 + 1/3 * 0.6), (2/3 * 0.2 + 1/3 * 0.4)]
        ])
        
        assert np.allclose(probas, expected)
    
    def test_predict(self):
        """Test class prediction."""
        model1 = MockClassifier(
            "Model1",
            np.array([0, 1, 0]),
            np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
        )
        model2 = MockClassifier(
            "Model2",
            np.array([0, 1, 1]),
            np.array([[0.7, 0.3], [0.4, 0.6], [0.4, 0.6]])
        )
        
        ensemble = EnsembleClassifier([model1, model2])
        X = np.array([[1, 2], [3, 4], [5, 6]])
        
        predictions = ensemble.predict(X)
        
        # Expected: class with highest average probability
        # Sample 0: [0.75, 0.25] -> class 0
        # Sample 1: [0.35, 0.65] -> class 1
        # Sample 2: [0.65, 0.35] -> class 0
        expected = np.array([0, 1, 0])
        
        assert np.array_equal(predictions, expected)
    
    def test_get_model_agreement_full_agreement(self):
        """Test agreement calculation when all models agree."""
        # All models predict the same
        model1 = MockClassifier(
            "Model1",
            np.array([0, 1, 0]),
            np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
        )
        model2 = MockClassifier(
            "Model2",
            np.array([0, 1, 0]),
            np.array([[0.7, 0.3], [0.4, 0.6], [0.85, 0.15]])
        )
        model3 = MockClassifier(
            "Model3",
            np.array([0, 1, 0]),
            np.array([[0.75, 0.25], [0.35, 0.65], [0.88, 0.12]])
        )
        
        ensemble = EnsembleClassifier([model1, model2, model3])
        X = np.array([[1, 2], [3, 4], [5, 6]])
        
        agreement = ensemble.get_model_agreement(X)
        
        # All models agree on all samples
        assert np.allclose(agreement, [1.0, 1.0, 1.0])
    
    def test_get_model_agreement_partial(self):
        """Test agreement calculation with partial disagreement."""
        # Models disagree on some predictions
        model1 = MockClassifier(
            "Model1",
            np.array([0, 1, 0]),
            np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
        )
        model2 = MockClassifier(
            "Model2",
            np.array([0, 0, 1]),  # Disagrees on samples 1 and 2
            np.array([[0.7, 0.3], [0.6, 0.4], [0.4, 0.6]])
        )
        model3 = MockClassifier(
            "Model3",
            np.array([0, 1, 0]),
            np.array([[0.75, 0.25], [0.35, 0.65], [0.88, 0.12]])
        )
        
        ensemble = EnsembleClassifier([model1, model2, model3])
        X = np.array([[1, 2], [3, 4], [5, 6]])
        
        agreement = ensemble.get_model_agreement(X)
        
        # Sample 0: all agree (3/3 = 1.0)
        # Sample 1: 2 agree (2/3 ≈ 0.67)
        # Sample 2: 2 agree (2/3 ≈ 0.67)
        assert agreement[0] == 1.0
        assert np.isclose(agreement[1], 2/3)
        assert np.isclose(agreement[2], 2/3)
    
    def test_get_individual_predictions(self):
        """Test getting individual model predictions."""
        model1 = MockClassifier(
            "RandomForest",
            np.array([0, 1]),
            np.array([[0.8, 0.2], [0.3, 0.7]])
        )
        model2 = MockClassifier(
            "NeuralNet",
            np.array([0, 0]),
            np.array([[0.7, 0.3], [0.6, 0.4]])
        )
        
        ensemble = EnsembleClassifier([model1, model2])
        X = np.array([[1, 2], [3, 4]])
        
        predictions = ensemble.get_individual_predictions(X)
        
        assert "RandomForest" in predictions
        assert "NeuralNet" in predictions
        assert np.array_equal(predictions["RandomForest"], [0, 1])
        assert np.array_equal(predictions["NeuralNet"], [0, 0])
    
    def test_get_individual_confidences(self):
        """Test getting individual model confidences."""
        model1 = MockClassifier(
            "Model1",
            np.array([0, 1]),
            np.array([[0.8, 0.2], [0.3, 0.7]])
        )
        model2 = MockClassifier(
            "Model2",
            np.array([0, 0]),
            np.array([[0.6, 0.4], [0.55, 0.45]])
        )
        
        ensemble = EnsembleClassifier([model1, model2])
        X = np.array([[1, 2], [3, 4]])
        
        confidences = ensemble.get_individual_confidences(X)
        
        # Confidence is max probability
        assert np.allclose(confidences["Model1"], [0.8, 0.7])
        assert np.allclose(confidences["Model2"], [0.6, 0.55])
    
    def test_get_detailed_prediction(self):
        """Test getting detailed prediction information."""
        model1 = MockClassifier(
            "RandomForest",
            np.array([0, 1]),
            np.array([[0.8, 0.2], [0.3, 0.7]])
        )
        model2 = MockClassifier(
            "NeuralNet",
            np.array([0, 1]),
            np.array([[0.7, 0.3], [0.4, 0.6]])
        )
        
        ensemble = EnsembleClassifier([model1, model2])
        X = np.array([[1, 2], [3, 4]])
        
        details = ensemble.get_detailed_prediction(X)
        
        assert len(details) == 2
        
        # Check first sample
        assert 'ensemble_prediction' in details[0]
        assert 'ensemble_confidence' in details[0]
        assert 'agreement_score' in details[0]
        assert 'model_predictions' in details[0]
        
        # Check model predictions structure
        assert 'RandomForest' in details[0]['model_predictions']
        assert 'NeuralNet' in details[0]['model_predictions']
        assert 'prediction' in details[0]['model_predictions']['RandomForest']
        assert 'confidence' in details[0]['model_predictions']['RandomForest']
    
    def test_fit_raises_error(self):
        """Test that fit() raises NotImplementedError."""
        model = MockClassifier("Model1", np.array([0]), np.array([[0.8, 0.2]]))
        ensemble = EnsembleClassifier([model])
        
        X = np.array([[1, 2]])
        y = np.array([0])
        
        with pytest.raises(NotImplementedError, match="combines pre-trained models"):
            ensemble.fit(X, y)
    
    def test_get_metadata(self):
        """Test getting ensemble metadata."""
        model1 = MockClassifier("Model1", np.array([0]), np.array([[0.8, 0.2]]))
        model2 = MockClassifier("Model2", np.array([0]), np.array([[0.7, 0.3]]))
        
        ensemble = EnsembleClassifier([model1, model2], weights=[0.6, 0.4])
        
        metadata = ensemble.get_metadata()
        
        assert metadata['model_name'] == 'EnsembleClassifier'
        assert metadata['is_trained'] == True
        assert metadata['ensemble_size'] == 2
        assert metadata['model_names'] == ['Model1', 'Model2']
        assert np.allclose(metadata['weights'], [0.6, 0.4])
    
    def test_three_model_ensemble(self):
        """Test ensemble with three models (RF, NN, SVM)."""
        rf = MockClassifier(
            "RandomForest",
            np.array([0, 1, 0]),
            np.array([[0.7, 0.3], [0.4, 0.6], [0.8, 0.2]])
        )
        nn = MockClassifier(
            "NeuralNetwork",
            np.array([0, 1, 0]),
            np.array([[0.85, 0.15], [0.3, 0.7], [0.75, 0.25]])
        )
        svm = MockClassifier(
            "SVM",
            np.array([0, 1, 1]),
            np.array([[0.75, 0.25], [0.35, 0.65], [0.45, 0.55]])
        )
        
        ensemble = EnsembleClassifier([rf, nn, svm])
        X = np.array([[1, 2], [3, 4], [5, 6]])
        
        predictions = ensemble.predict(X)
        agreement = ensemble.get_model_agreement(X)
        
        # Sample 0: all predict 0 (agreement = 1.0)
        # Sample 1: all predict 1 (agreement = 1.0)
        # Sample 2: 2 predict 0, 1 predicts 1 (agreement = 2/3)
        assert predictions[0] == 0
        assert predictions[1] == 1
        assert agreement[0] == 1.0
        assert agreement[1] == 1.0
        assert np.isclose(agreement[2], 2/3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
