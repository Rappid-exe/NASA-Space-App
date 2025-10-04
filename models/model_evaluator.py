"""
Model evaluation utilities with accuracy, precision, recall, and F1-score metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import json


class ModelEvaluator:
    """Evaluates model performance with comprehensive metrics."""
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.evaluation_results = {}
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                      y_pred_proba: Optional[np.ndarray] = None,
                      model_name: str = "Model") -> Dict[str, Any]:
        """
        Evaluate model performance with multiple metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            model_name: Name of the model being evaluated
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Convert to numpy arrays if needed
        if hasattr(y_true, 'values'):
            y_true = y_true.values
        if hasattr(y_pred, 'values'):
            y_pred = y_pred.values

        # Calculate basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Determine if binary or multi-class
        n_classes = len(np.unique(y_true))
        average_method = 'binary' if n_classes == 2 else 'weighted'
        
        precision = precision_score(y_true, y_pred, average=average_method, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average_method, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average_method, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # ROC AUC (for binary classification with probabilities)
        roc_auc = None
        if y_pred_proba is not None and n_classes == 2:
            try:
                # Use probabilities for positive class
                if y_pred_proba.ndim == 2:
                    y_pred_proba_positive = y_pred_proba[:, 1]
                else:
                    y_pred_proba_positive = y_pred_proba
                roc_auc = roc_auc_score(y_true, y_pred_proba_positive)
            except Exception as e:
                print(f"Warning: Could not calculate ROC AUC: {e}")
        
        # Compile results
        results = {
            'model_name': model_name,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'roc_auc': float(roc_auc) if roc_auc is not None else None,
            'n_samples': len(y_true),
            'n_classes': n_classes
        }
        
        # Store results
        self.evaluation_results[model_name] = results
        
        return results

    def print_evaluation_summary(self, model_name: str) -> None:
        """
        Print a formatted summary of evaluation results.
        
        Args:
            model_name: Name of the model to print results for
        """
        if model_name not in self.evaluation_results:
            print(f"No evaluation results found for {model_name}")
            return
        
        results = self.evaluation_results[model_name]
        
        print(f"\n{'='*60}")
        print(f"Evaluation Results for {model_name}")
        print(f"{'='*60}")
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1-Score:  {results['f1_score']:.4f}")
        
        if results['roc_auc'] is not None:
            print(f"ROC AUC:   {results['roc_auc']:.4f}")
        
        print(f"\nConfusion Matrix:")
        cm = np.array(results['confusion_matrix'])
        print(cm)
        
        print(f"\nSamples: {results['n_samples']}, Classes: {results['n_classes']}")
        print(f"{'='*60}\n")
    
    def compare_models(self, model_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare multiple models and identify the best performer.
        
        Args:
            model_names: List of model names to compare (None = all models)
            
        Returns:
            Dictionary with comparison results and best model
        """
        if model_names is None:
            model_names = list(self.evaluation_results.keys())
        
        if not model_names:
            return {'error': 'No models to compare'}
        
        comparison = {
            'models': {},
            'best_model': None,
            'best_f1_score': 0.0
        }
        
        for name in model_names:
            if name in self.evaluation_results:
                results = self.evaluation_results[name]
                comparison['models'][name] = {
                    'accuracy': results['accuracy'],
                    'precision': results['precision'],
                    'recall': results['recall'],
                    'f1_score': results['f1_score'],
                    'roc_auc': results['roc_auc']
                }
                
                # Track best model by F1 score
                if results['f1_score'] > comparison['best_f1_score']:
                    comparison['best_f1_score'] = results['f1_score']
                    comparison['best_model'] = name
        
        return comparison

    def print_comparison(self, model_names: Optional[List[str]] = None) -> None:
        """
        Print a formatted comparison of multiple models.
        
        Args:
            model_names: List of model names to compare (None = all models)
        """
        comparison = self.compare_models(model_names)
        
        if 'error' in comparison:
            print(comparison['error'])
            return
        
        print(f"\n{'='*80}")
        print(f"Model Comparison")
        print(f"{'='*80}")
        print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print(f"{'-'*80}")
        
        for name, metrics in comparison['models'].items():
            marker = " *" if name == comparison['best_model'] else ""
            print(f"{name:<20} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f}{marker}")
        
        print(f"{'-'*80}")
        print(f"Best Model: {comparison['best_model']} (F1-Score: {comparison['best_f1_score']:.4f})")
        print(f"{'='*80}\n")
    
    def save_results(self, filepath: str) -> None:
        """
        Save evaluation results to a JSON file.
        
        Args:
            filepath: Path to save the results
        """
        with open(filepath, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        print(f"Evaluation results saved to {filepath}")
    
    def load_results(self, filepath: str) -> None:
        """
        Load evaluation results from a JSON file.
        
        Args:
            filepath: Path to load the results from
        """
        with open(filepath, 'r') as f:
            self.evaluation_results = json.load(f)
        
        print(f"Evaluation results loaded from {filepath}")
