"""
Example usage of the Model Registry and Persistence System.
Demonstrates integration with the existing model training pipeline.
"""

import numpy as np
from models.model_trainer import ModelTrainer
from models.model_registry import ModelRegistry
from models.model_persistence import ModelPersistence
from data.data_processor import DataProcessor
from data.dataset_loader import DatasetLoader


def example_train_and_register():
    """Example: Train models and register them in the registry."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Train and Register Models")
    print("="*60)
    
    # Load and process data
    print("\n1. Loading dataset...")
    loader = DatasetLoader()
    df = loader.load_dataset('data/raw/cumulative_2025.01.01_17.28.48.csv')
    
    # Process data
    print("2. Processing data...")
    processor = DataProcessor()
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, label_encoder = \
        processor.process_dataset(df, test_size=0.15, val_size=0.15)
    
    # Train models
    print("3. Training models...")
    trainer = ModelTrainer(random_state=42)
    results = trainer.train_and_evaluate(
        X_train, y_train, X_val, y_val, X_test, y_test,
        feature_names=feature_names
    )
    
    # Initialize registry
    print("\n4. Registering models in registry...")
    registry = ModelRegistry("models/registry")
    
    # Register each trained model
    for model_name, model in results['models'].items():
        evaluation = results['evaluation_results'][model_name]
        
        # Get hyperparameters from model
        if hasattr(model, 'model') and hasattr(model.model, 'get_params'):
            hyperparameters = model.model.get_params()
        else:
            hyperparameters = {}
        
        # Prepare training info
        training_info = {
            'dataset': 'cumulative_2025.01.01',
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'class_distribution': {
                label_encoder.inverse_transform([i])[0]: int(np.sum(y_train == i))
                for i in range(len(label_encoder.classes_))
            }
        }
        
        # Register model
        model_id = registry.register_model(
            model=model,
            model_name=model_name,
            algorithm=model_name,
            evaluation_results=evaluation,
            hyperparameters=hyperparameters,
            training_info=training_info,
            feature_columns=feature_names
        )
        
        print(f"  ✓ Registered {model_name}: {model_id}")
    
    # Show registry summary
    print("\n5. Registry Summary:")
    registry.print_registry_summary()
    
    # Compare all models
    print("6. Model Comparison:")
    registry.compare_models()
    
    return registry


def example_load_and_predict():
    """Example: Load a model from registry and make predictions."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Load Model and Make Predictions")
    print("="*60)
    
    # Initialize registry
    registry = ModelRegistry("models/registry")
    
    # Get best model
    print("\n1. Loading best model from registry...")
    best_model, best_metadata = registry.get_best_model()
    
    print(f"\nBest Model Details:")
    print(f"  - Model ID: {best_metadata.model_id}")
    print(f"  - Algorithm: {best_metadata.algorithm}")
    print(f"  - Version: {best_metadata.version}")
    print(f"  - F1 Score: {best_metadata.f1_score:.4f}")
    print(f"  - Accuracy: {best_metadata.accuracy:.4f}")
    
    # Load test data
    print("\n2. Loading test data...")
    loader = DatasetLoader()
    df = loader.load_dataset('data/raw/cumulative_2025.01.01_17.28.48.csv')
    
    processor = DataProcessor()
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, label_encoder = \
        processor.process_dataset(df, test_size=0.15, val_size=0.15)
    
    # Make predictions
    print("3. Making predictions...")
    predictions = best_model.predict(X_test[:10])
    probabilities = best_model.predict_proba(X_test[:10])
    
    print("\nSample Predictions:")
    for i in range(min(5, len(predictions))):
        pred_label = label_encoder.inverse_transform([predictions[i]])[0]
        true_label = label_encoder.inverse_transform([y_test[i]])[0]
        confidence = np.max(probabilities[i])
        
        print(f"  Sample {i+1}: Predicted={pred_label}, True={true_label}, "
              f"Confidence={confidence:.2%}")


def example_version_management():
    """Example: Manage multiple versions of a model."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Version Management")
    print("="*60)
    
    registry = ModelRegistry("models/registry")
    
    # List all versions of a specific model
    print("\n1. Listing all versions of RandomForest model:")
    versions = registry.list_models("RandomForest")
    
    if versions:
        for version_info in versions:
            print(f"  - Version {version_info['version']}: "
                  f"F1={version_info['f1_score']:.4f}, "
                  f"Trained on {version_info['training_date'][:10]}")
        
        # Load specific version
        print("\n2. Loading specific version...")
        model_v1, metadata_v1 = registry.load_model(model_name="RandomForest", version=1)
        print(f"  Loaded: {metadata_v1.model_id}")
        
        # Load latest version
        print("\n3. Loading latest version...")
        model_latest, metadata_latest = registry.load_model(model_name="RandomForest")
        print(f"  Loaded: {metadata_latest.model_id} (v{metadata_latest.version})")
    else:
        print("  No RandomForest models found in registry")


def example_export_for_deployment():
    """Example: Export model for deployment."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Export Model for Deployment")
    print("="*60)
    
    registry = ModelRegistry("models/registry")
    
    # Get best model
    print("\n1. Getting best model...")
    best_model, best_metadata = registry.get_best_model()
    
    # Export for deployment
    print("2. Exporting model for deployment...")
    export_path = "deployment/exoplanet_classifier"
    
    model_info = {
        'model_name': best_metadata.model_name,
        'algorithm': best_metadata.algorithm,
        'version': best_metadata.version,
        'training_date': best_metadata.training_date,
        'accuracy': best_metadata.accuracy,
        'f1_score': best_metadata.f1_score,
        'precision': best_metadata.precision,
        'recall': best_metadata.recall,
        'feature_columns': best_metadata.feature_columns,
        'hyperparameters': best_metadata.hyperparameters
    }
    
    ModelPersistence.export_model_for_deployment(
        model=best_model,
        export_path=export_path,
        model_info=model_info
    )
    
    print(f"\n✓ Model exported to: {export_path}")
    print("  Files created:")
    print("    - model.pkl (serialized model)")
    print("    - model_info.json (metadata)")
    print("    - README.md (documentation)")


def example_model_comparison():
    """Example: Compare different models."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Compare Models")
    print("="*60)
    
    registry = ModelRegistry("models/registry")
    
    # Compare all models
    print("\n1. Comparing all models in registry:")
    registry.compare_models()
    
    # Get statistics
    print("2. Registry Statistics:")
    stats = registry.get_registry_stats()
    print(f"  - Total models: {stats['total_models']}")
    print(f"  - Model types: {stats['model_types']}")
    print(f"  - Registered: {', '.join(stats['model_names'])}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("MODEL REGISTRY AND PERSISTENCE - USAGE EXAMPLES")
    print("="*60)
    
    try:
        # Check if we have data
        import os
        if not os.path.exists('data/raw/cumulative_2025.01.01_17.28.48.csv'):
            print("\n⚠ Dataset not found. Please run data ingestion first.")
            print("  Run: python test_data_ingestion.py")
            return
        
        # Example 1: Train and register models
        registry = example_train_and_register()
        
        # Example 2: Load and predict
        example_load_and_predict()
        
        # Example 3: Version management
        example_version_management()
        
        # Example 4: Export for deployment
        example_export_for_deployment()
        
        # Example 5: Model comparison
        example_model_comparison()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
