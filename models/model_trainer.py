"""
Model trainer that orchestrates the training process for multiple classifiers.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from models.base_classifier import BaseClassifier
from models.random_forest_classifier import RandomForestClassifier
from models.neural_network_classifier import NeuralNetworkClassifier
from models.svm_classifier import SVMClassifier
from models.model_evaluator import ModelEvaluator


class ModelTrainer:
    """Orchestrates training and evaluation of multiple ML models."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the model trainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.evaluator = ModelEvaluator()
        self.best_model = None
        self.best_model_name = None
    
    def initialize_models(self, model_configs: Optional[Dict[str, Dict]] = None) -> None:
        """
        Initialize classifiers with configurations.
        
        Args:
            model_configs: Dictionary of model configurations
        """
        if model_configs is None:
            # Default configurations
            model_configs = {
                'RandomForest': {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'random_state': self.random_state
                },
                'NeuralNetwork': {
                    'hidden_layers': (128, 64, 32),
                    'dropout_rate': 0.3,
                    'learning_rate': 0.001,
                    'random_state': self.random_state
                },
                'SVM': {
                    'kernel': 'rbf',
                    'C': 1.0,
                    'gamma': 'scale',
                    'random_state': self.random_state
                }
            }

        # Initialize models
        for model_name, config in model_configs.items():
            if model_name == 'RandomForest':
                self.models[model_name] = RandomForestClassifier(**config)
            elif model_name == 'NeuralNetwork':
                self.models[model_name] = NeuralNetworkClassifier(**config)
            elif model_name == 'SVM':
                self.models[model_name] = SVMClassifier(**config)
            else:
                print(f"Warning: Unknown model type '{model_name}', skipping...")
        
        print(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                    feature_names: Optional[List[str]] = None) -> Dict[str, BaseClassifier]:
        """
        Train all initialized models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (for Neural Network)
            y_val: Validation labels (for Neural Network)
            feature_names: List of feature names
            
        Returns:
            Dictionary of trained models
        """
        if not self.models:
            raise ValueError("No models initialized. Call initialize_models() first.")
        
        print(f"\n{'='*60}")
        print(f"Training {len(self.models)} models...")
        print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
        print(f"{'='*60}\n")
        
        for model_name, model in self.models.items():
            print(f"\n--- Training {model_name} ---")
            
            try:
                if model_name == 'NeuralNetwork' and X_val is not None and y_val is not None:
                    # Neural Network uses validation data during training
                    model.fit(X_train, y_train, X_val=X_val, y_val=y_val,
                            feature_names=feature_names, epochs=50, batch_size=32, verbose=0)
                else:
                    model.fit(X_train, y_train, feature_names=feature_names)
                
                print(f"{model_name} trained successfully!")
            
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                del self.models[model_name]
        
        print(f"\n{'='*60}")
        print(f"Training complete! {len(self.models)} models trained successfully.")
        print(f"{'='*60}\n")
        
        return self.models

    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray,
                       models: Optional[Dict[str, BaseClassifier]] = None) -> Dict[str, Dict]:
        """
        Evaluate all trained models.
        
        Args:
            X_test: Test features
            y_test: Test labels
            models: Dictionary of models to evaluate (None = use self.models)
            
        Returns:
            Dictionary of evaluation results
        """
        if models is None:
            models = self.models
        
        if not models:
            raise ValueError("No models to evaluate")
        
        print(f"\n{'='*60}")
        print(f"Evaluating {len(models)} models on test set...")
        print(f"Test samples: {X_test.shape[0]}")
        print(f"{'='*60}\n")
        
        evaluation_results = {}
        
        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")
            
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)
                
                # Evaluate
                results = self.evaluator.evaluate_model(
                    y_test, y_pred, y_pred_proba, model_name
                )
                evaluation_results[model_name] = results
                
                # Print summary
                self.evaluator.print_evaluation_summary(model_name)
            
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}\n")
        
        return evaluation_results
    
    def select_best_model(self, evaluation_results: Optional[Dict] = None,
                         metric: str = 'f1_score') -> BaseClassifier:
        """
        Select the best performing model based on a metric.
        
        Args:
            evaluation_results: Dictionary of evaluation results (None = use evaluator's results)
            metric: Metric to use for selection ('accuracy', 'f1_score', 'precision', 'recall')
            
        Returns:
            Best performing model
        """
        if evaluation_results is None:
            evaluation_results = self.evaluator.evaluation_results
        
        if not evaluation_results:
            raise ValueError("No evaluation results available")
        
        best_score = -1
        best_name = None
        
        for model_name, results in evaluation_results.items():
            score = results.get(metric, 0)
            if score > best_score:
                best_score = score
                best_name = model_name
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        print(f"\nBest model: {best_name} ({metric}: {best_score:.4f})")
        
        return self.best_model

    def train_and_evaluate(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          feature_names: Optional[List[str]] = None,
                          model_configs: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Complete training and evaluation pipeline.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Test features
            y_test: Test labels
            feature_names: List of feature names
            model_configs: Model configurations
            
        Returns:
            Dictionary with models, results, and best model
        """
        # Initialize models
        self.initialize_models(model_configs)
        
        # Train models
        trained_models = self.train_models(X_train, y_train, X_val, y_val, feature_names)
        
        # Evaluate models
        evaluation_results = self.evaluate_models(X_test, y_test, trained_models)
        
        # Compare models
        self.evaluator.print_comparison()
        
        # Select best model
        best_model = self.select_best_model(evaluation_results)
        
        return {
            'models': trained_models,
            'evaluation_results': evaluation_results,
            'best_model': best_model,
            'best_model_name': self.best_model_name,
            'evaluator': self.evaluator
        }
    
    def save_model(self, model_name: str, filepath: str) -> None:
        """
        Save a specific model to disk.
        
        Args:
            model_name: Name of the model to save
            filepath: Path to save the model
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        self.models[model_name].save_model(filepath)
        print(f"Model '{model_name}' saved to {filepath}")
    
    def save_best_model(self, filepath: str) -> None:
        """
        Save the best performing model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if self.best_model is None:
            raise ValueError("No best model selected. Run select_best_model() first.")
        
        self.best_model.save_model(filepath)
        print(f"Best model '{self.best_model_name}' saved to {filepath}")
