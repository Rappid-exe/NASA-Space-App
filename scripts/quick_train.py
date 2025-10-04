"""
Quick training script with synthetic exoplanet-like data for testing.
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.model_trainer import ModelTrainer
from models.model_registry import ModelRegistry

def main():
    print("="*80)
    print("QUICK TRAINING - Synthetic Exoplanet Data")
    print("="*80)
    
    # Generate realistic exoplanet-like data
    print("\n1. Generating synthetic exoplanet dataset...")
    X, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_classes=2,
        weights=[0.7, 0.3],  # Imbalanced like real exoplanet data
        random_state=42
    )
    
    # Create feature names
    feature_names = [
        'koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq',
        'period_duration_ratio', 'depth_radius_correlation', 'temp_habitable_zone',
        'period_category', 'radius_category', 'transit_snr',
        'koi_impact', 'koi_insol', 'koi_steff', 'koi_slogg',
        'koi_srad', 'koi_smass', 'koi_sage', 'koi_kepmag', 'koi_dist'
    ]
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.67, random_state=42, stratify=y_temp)
    
    print(f"✓ Generated {len(X)} samples")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Class distribution: {np.bincount(y_train)}")
    
    # Train models
    print("\n2. Training models...")
    trainer = ModelTrainer()
    trainer.initialize_models()
    trainer.train_models(X_train, y_train, X_val, y_val, feature_names=feature_names)
    
    # Evaluate
    print("\n3. Evaluating models...")
    results = trainer.evaluate_models(X_test, y_test)
    
    # Select best model
    print("\n4. Selecting best model...")
    best_model = trainer.select_best_model(results, metric='f1_score')
    best_model_name = trainer.best_model_name
    best_results = results[best_model_name]
    
    print(f"✓ Best model: {best_model_name} (F1: {best_results['f1_score']:.4f})")
    
    # Register best model
    print("\n5. Registering best model...")
    registry = ModelRegistry()
    
    best_model_obj = trainer.models[best_model_name]
    
    model_id = registry.register_model(
        model=best_model_obj.model,
        model_name="exoplanet_classifier",
        algorithm=best_model_name,
        evaluation_results={
            'accuracy': best_results['accuracy'],
            'precision': best_results['precision'],
            'recall': best_results['recall'],
            'f1_score': best_results['f1_score'],
            'additional_metrics': {
                'roc_auc': best_results.get('roc_auc', 0.0)
            }
        },
        hyperparameters=best_model_obj.training_metadata,
        training_info={
            'dataset': 'Synthetic Exoplanet Data',
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'class_distribution': {
                'false_positive': int((y_train == 0).sum()),
                'confirmed': int((y_train == 1).sum())
            }
        },
        feature_columns=feature_names
    )
    
    print("\n" + "="*80)
    print("✓ TRAINING COMPLETE!")
    print("="*80)
    print(f"Best Model: {best_model_name}")
    print(f"Accuracy: {best_results['accuracy']:.2%}")
    print(f"Precision: {best_results['precision']:.2%}")
    print(f"Recall: {best_results['recall']:.2%}")
    print(f"F1 Score: {best_results['f1_score']:.2%}")
    print(f"\nModel ID: {model_id}")
    print("\n✓ Model registered and ready for inference!")
    print("\nNext steps:")
    print("  1. Start API: python -m uvicorn api.main:app --reload")
    print("  2. Start Frontend: cd frontend && npm run dev")
    print("  3. Open: http://localhost:3000")
    print("="*80)

if __name__ == "__main__":
    main()
