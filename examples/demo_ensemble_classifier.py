"""
Demo script for EnsembleClassifier.

This script demonstrates how to create and use an ensemble classifier
that combines multiple trained models for improved predictions.
"""

import numpy as np
from models.random_forest_classifier import RandomForestClassifier
from models.neural_network_classifier import NeuralNetworkClassifier
from models.svm_classifier import SVMClassifier
from models.ensemble_classifier import EnsembleClassifier


def create_sample_data():
    """Create sample training and test data."""
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 200
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    # Create binary classification target
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Split into train and test
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return X_train, X_test, y_train, y_test


def main():
    """Demonstrate ensemble classifier usage."""
    print("=" * 60)
    print("Ensemble Classifier Demo")
    print("=" * 60)
    
    # Create sample data
    print("\n1. Creating sample data...")
    X_train, X_test, y_train, y_test = create_sample_data()
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Features: {X_train.shape[1]}")
    
    # Train individual models
    print("\n2. Training individual models...")
    
    print("   - Training RandomForest...")
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    rf_acc = np.mean(rf.predict(X_test) == y_test)
    print(f"     RandomForest accuracy: {rf_acc:.2%}")
    
    print("   - Training Neural Network...")
    nn = NeuralNetworkClassifier()
    nn.fit(X_train, y_train, epochs=50, verbose=False)
    nn_acc = np.mean(nn.predict(X_test) == y_test)
    print(f"     Neural Network accuracy: {nn_acc:.2%}")
    
    print("   - Training SVM...")
    svm = SVMClassifier()
    svm.fit(X_train, y_train)
    svm_acc = np.mean(svm.predict(X_test) == y_test)
    print(f"     SVM accuracy: {svm_acc:.2%}")
    
    # Create ensemble with equal weights
    print("\n3. Creating ensemble (equal weights)...")
    ensemble_equal = EnsembleClassifier([rf, nn, svm])
    ensemble_preds = ensemble_equal.predict(X_test)
    ensemble_acc = np.mean(ensemble_preds == y_test)
    print(f"   Ensemble accuracy: {ensemble_acc:.2%}")
    
    # Create ensemble with custom weights (favor best model)
    print("\n4. Creating ensemble (weighted by accuracy)...")
    accuracies = [rf_acc, nn_acc, svm_acc]
    weights = [acc / sum(accuracies) for acc in accuracies]
    ensemble_weighted = EnsembleClassifier([rf, nn, svm], weights=weights)
    weighted_preds = ensemble_weighted.predict(X_test)
    weighted_acc = np.mean(weighted_preds == y_test)
    print(f"   Weighted ensemble accuracy: {weighted_acc:.2%}")
    print(f"   Weights: RF={weights[0]:.3f}, NN={weights[1]:.3f}, SVM={weights[2]:.3f}")
    
    # Demonstrate model agreement
    print("\n5. Analyzing model agreement...")
    agreement_scores = ensemble_equal.get_model_agreement(X_test)
    avg_agreement = np.mean(agreement_scores)
    print(f"   Average agreement: {avg_agreement:.2%}")
    print(f"   High agreement (>90%): {np.sum(agreement_scores > 0.9)} samples")
    print(f"   Low agreement (<70%): {np.sum(agreement_scores < 0.7)} samples")
    
    # Show detailed predictions for a few samples
    print("\n6. Detailed predictions for first 3 test samples:")
    details = ensemble_equal.get_detailed_prediction(X_test[:3])
    
    for i, detail in enumerate(details):
        print(f"\n   Sample {i + 1}:")
        print(f"   - Ensemble: class {detail['ensemble_prediction']} "
              f"(confidence: {detail['ensemble_confidence']:.2%})")
        print(f"   - Agreement: {detail['agreement_score']:.2%}")
        print(f"   - Individual models:")
        for model_name, info in detail['model_predictions'].items():
            print(f"     • {model_name}: class {info['prediction']} "
                  f"(confidence: {info['confidence']:.2%})")
    
    # Compare individual vs ensemble
    print("\n7. Performance Summary:")
    print(f"   {'Model':<20} {'Accuracy':<10}")
    print(f"   {'-' * 30}")
    print(f"   {'RandomForest':<20} {rf_acc:.2%}")
    print(f"   {'Neural Network':<20} {nn_acc:.2%}")
    print(f"   {'SVM':<20} {svm_acc:.2%}")
    print(f"   {'-' * 30}")
    print(f"   {'Ensemble (equal)':<20} {ensemble_acc:.2%}")
    print(f"   {'Ensemble (weighted)':<20} {weighted_acc:.2%}")
    
    improvement = ensemble_acc - max(rf_acc, nn_acc, svm_acc)
    print(f"\n   Improvement over best single model: {improvement:+.2%}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
