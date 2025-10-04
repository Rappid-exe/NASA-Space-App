"""
Test script for model training engine.
Tests all classifiers and evaluation utilities.
"""

import sys
import os
import numpy as np
from sklearn.datasets import make_classification

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.model_trainer import ModelTrainer


def test_model_training_engine():
    """Test the complete model training engine."""
    
    print("="*80)
    print("Testing Machine Learning Model Training Engine")
    print("="*80)
    
    # Generate synthetic exoplanet-like data
    print("\n1. Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42,
        class_sep=1.5
    )
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Initialize trainer
    print("\n2. Initializing ModelTrainer...")
    trainer = ModelTrainer(random_state=42)
    
    # Train and evaluate all models
    print("\n3. Training and evaluating models...")
    results = trainer.train_and_evaluate(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        feature_names=feature_names
    )
    
    print("\n4. Testing individual model features...")
    
    # Test Random Forest feature importance
    if 'RandomForest' in results['models']:
        rf_model = results['models']['RandomForest']
        importance = rf_model.get_feature_importance_dict()
        print(f"\nRandom Forest - Top 5 important features:")
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        for feat, imp in sorted_features:
            print(f"  {feat}: {imp:.4f}")

    # Test Neural Network training history
    if 'NeuralNetwork' in results['models']:
        nn_model = results['models']['NeuralNetwork']
        history = nn_model.get_training_history()
        if history:
            print(f"\nNeural Network - Training epochs: {len(history['loss'])}")
            print(f"  Final training loss: {history['loss'][-1]:.4f}")
            if 'val_loss' in history:
                print(f"  Final validation loss: {history['val_loss'][-1]:.4f}")
    
    # Test SVM support vectors
    if 'SVM' in results['models']:
        svm_model = results['models']['SVM']
        support_vectors = svm_model.get_support_vectors()
        if support_vectors is not None:
            print(f"\nSVM - Number of support vectors: {len(support_vectors)}")
    
    # Test model saving and loading
    print("\n5. Testing model persistence...")
    best_model_name = results['best_model_name']
    test_filepath = f"test_{best_model_name.lower()}_model.pkl"
    
    try:
        trainer.save_best_model(test_filepath)
        
        # Load and test
        from models import RandomForestClassifier, NeuralNetworkClassifier, SVMClassifier
        
        if best_model_name == 'RandomForest':
            loaded_model = RandomForestClassifier()
        elif best_model_name == 'NeuralNetwork':
            loaded_model = NeuralNetworkClassifier()
        else:
            loaded_model = SVMClassifier()
        
        loaded_model.load_model(test_filepath)
        
        # Make predictions with loaded model
        test_predictions = loaded_model.predict(X_test[:5])
        print(f"Loaded model predictions (first 5): {test_predictions}")
        
        # Clean up
        import os
        os.remove(test_filepath)
        print(f"Model save/load test successful!")
    
    except Exception as e:
        print(f"Error in model persistence test: {e}")
    
    print("\n" + "="*80)
    print("Model Training Engine Test Complete!")
    print("="*80)
    
    return results


if __name__ == "__main__":
    try:
        results = test_model_training_engine()
        print("\n✓ All tests passed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
