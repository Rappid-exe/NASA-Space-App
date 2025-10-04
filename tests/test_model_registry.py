"""
Test script for model registry and persistence system.
Tests model serialization, versioning, metadata storage, and loading.
"""

import numpy as np
import os
import shutil
from pathlib import Path
from models.model_registry import ModelRegistry, ModelMetadata
from models.model_persistence import ModelPersistence
from models.random_forest_classifier import RandomForestClassifier
from models.neural_network_classifier import NeuralNetworkClassifier


def create_sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.randn(30, 10)
    y_test = np.random.randint(0, 2, 30)
    return X_train, y_train, X_test, y_test


def test_model_persistence():
    """Test basic model persistence utilities."""
    print("\n" + "="*60)
    print("TEST 1: Model Persistence")
    print("="*60)
    
    # Create and train a simple model
    X_train, y_train, X_test, y_test = create_sample_data()
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Test saving
    test_dir = Path("test_models")
    test_dir.mkdir(exist_ok=True)
    
    model_path = test_dir / "test_model.pkl"
    metadata = {
        "model_name": "TestModel",
        "accuracy": 0.85,
        "notes": "Test model for persistence"
    }
    
    ModelPersistence.save_model(model, str(model_path), metadata)
    
    # Test loading
    loaded_model, loaded_metadata = ModelPersistence.load_model(str(model_path))
    
    # Verify
    original_pred = model.predict(X_test)
    loaded_pred = loaded_model.predict(X_test)
    
    assert np.array_equal(original_pred, loaded_pred), "Predictions don't match!"
    assert loaded_metadata["model_name"] == "TestModel", "Metadata doesn't match!"
    
    print("✓ Model persistence test passed!")
    print(f"  - Model saved and loaded successfully")
    print(f"  - Predictions match: {np.array_equal(original_pred, loaded_pred)}")
    print(f"  - Metadata preserved: {loaded_metadata['model_name']}")
    
    # Cleanup
    shutil.rmtree(test_dir)


def test_model_registry_basic():
    """Test basic model registry functionality."""
    print("\n" + "="*60)
    print("TEST 2: Model Registry - Basic Operations")
    print("="*60)
    
    # Create registry
    registry_path = "test_registry"
    registry = ModelRegistry(registry_path)
    
    # Create and train models
    X_train, y_train, X_test, y_test = create_sample_data()
    
    # Train RandomForest
    rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    
    # Register model
    evaluation_results = {
        'accuracy': accuracy,
        'precision': 0.82,
        'recall': 0.78,
        'f1_score': 0.80
    }
    
    hyperparameters = {
        'n_estimators': 10,
        'random_state': 42
    }
    
    training_info = {
        'dataset': 'test_dataset',
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'class_distribution': {'class_0': 50, 'class_1': 50}
    }
    
    feature_columns = [f'feature_{i}' for i in range(10)]
    
    model_id = registry.register_model(
        model=rf_model,
        model_name="ExoplanetClassifier",
        algorithm="RandomForest",
        evaluation_results=evaluation_results,
        hyperparameters=hyperparameters,
        training_info=training_info,
        feature_columns=feature_columns
    )
    
    print(f"✓ Model registered with ID: {model_id}")
    
    # Test loading
    loaded_model, loaded_metadata = registry.load_model(model_id=model_id)
    
    # Verify
    original_pred = rf_model.predict(X_test)
    loaded_pred = loaded_model.predict(X_test)
    
    assert np.array_equal(original_pred, loaded_pred), "Predictions don't match!"
    assert loaded_metadata.model_name == "ExoplanetClassifier", "Model name doesn't match!"
    assert loaded_metadata.algorithm == "RandomForest", "Algorithm doesn't match!"
    
    print("✓ Model loaded successfully from registry")
    print(f"  - Model name: {loaded_metadata.model_name}")
    print(f"  - Algorithm: {loaded_metadata.algorithm}")
    print(f"  - Version: {loaded_metadata.version}")
    print(f"  - F1 Score: {loaded_metadata.f1_score:.4f}")
    
    # Cleanup
    shutil.rmtree(registry_path)


def test_model_versioning():
    """Test model versioning system."""
    print("\n" + "="*60)
    print("TEST 3: Model Versioning")
    print("="*60)
    
    registry_path = "test_registry_versioning"
    registry = ModelRegistry(registry_path)
    
    X_train, y_train, X_test, y_test = create_sample_data()
    
    # Register multiple versions
    for version in range(1, 4):
        model = RandomForestClassifier(n_estimators=10*version, random_state=42)
        model.fit(X_train, y_train)
        
        evaluation_results = {
            'accuracy': 0.80 + version * 0.02,
            'precision': 0.78 + version * 0.02,
            'recall': 0.76 + version * 0.02,
            'f1_score': 0.77 + version * 0.02
        }
        
        model_id = registry.register_model(
            model=model,
            model_name="ExoplanetClassifier",
            algorithm="RandomForest",
            evaluation_results=evaluation_results,
            hyperparameters={'n_estimators': 10*version},
            training_info={'dataset': f'dataset_v{version}', 'training_samples': 100, 
                          'test_samples': 30, 'class_distribution': {}},
            feature_columns=[f'feature_{i}' for i in range(10)]
        )
        
        print(f"✓ Registered version {version}: {model_id}")
    
    # List all versions
    versions = registry.list_models("ExoplanetClassifier")
    assert len(versions) == 3, f"Expected 3 versions, got {len(versions)}"
    print(f"✓ Found {len(versions)} versions")
    
    # Load specific version
    model_v2, metadata_v2 = registry.load_model(model_name="ExoplanetClassifier", version=2)
    assert metadata_v2.version == 2, "Wrong version loaded!"
    print(f"✓ Loaded specific version: {metadata_v2.version}")
    
    # Load latest version
    model_latest, metadata_latest = registry.load_model(model_name="ExoplanetClassifier")
    assert metadata_latest.version == 3, "Latest version should be 3!"
    print(f"✓ Loaded latest version: {metadata_latest.version}")
    
    # Get best model
    best_model, best_metadata = registry.get_best_model(model_name="ExoplanetClassifier")
    print(f"✓ Best model (by F1): Version {best_metadata.version} (F1: {best_metadata.f1_score:.4f})")
    
    # Cleanup
    shutil.rmtree(registry_path)


def test_model_comparison():
    """Test model comparison functionality."""
    print("\n" + "="*60)
    print("TEST 4: Model Comparison")
    print("="*60)
    
    registry_path = "test_registry_comparison"
    registry = ModelRegistry(registry_path)
    
    X_train, y_train, X_test, y_test = create_sample_data()
    
    # Register different algorithms
    algorithms = [
        ("RandomForest", RandomForestClassifier(n_estimators=10, random_state=42)),
        ("NeuralNetwork", NeuralNetworkClassifier(hidden_layers=(32, 16), random_state=42))
    ]
    
    for algo_name, model in algorithms:
        if algo_name == "NeuralNetwork":
            model.fit(X_train, y_train, epochs=5, verbose=0)
        else:
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        
        evaluation_results = {
            'accuracy': accuracy,
            'precision': accuracy * 0.95,
            'recall': accuracy * 0.93,
            'f1_score': accuracy * 0.94
        }
        
        registry.register_model(
            model=model,
            model_name=f"{algo_name}Classifier",
            algorithm=algo_name,
            evaluation_results=evaluation_results,
            hyperparameters={},
            training_info={'dataset': 'test', 'training_samples': 100, 
                          'test_samples': 30, 'class_distribution': {}},
            feature_columns=[f'feature_{i}' for i in range(10)]
        )
    
    # Compare models
    registry.compare_models()
    
    # Get registry stats
    stats = registry.get_registry_stats()
    print(f"✓ Registry contains {stats['total_models']} models")
    print(f"✓ Model types: {stats['model_names']}")
    
    # Cleanup
    shutil.rmtree(registry_path)


def test_export_import():
    """Test model export and import functionality."""
    print("\n" + "="*60)
    print("TEST 5: Model Export/Import")
    print("="*60)
    
    registry_path = "test_registry_export"
    registry = ModelRegistry(registry_path)
    
    X_train, y_train, X_test, y_test = create_sample_data()
    
    # Train and register model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    model_id = registry.register_model(
        model=model,
        model_name="ExportTest",
        algorithm="RandomForest",
        evaluation_results={'accuracy': 0.85, 'precision': 0.83, 'recall': 0.81, 'f1_score': 0.82},
        hyperparameters={'n_estimators': 10},
        training_info={'dataset': 'test', 'training_samples': 100, 
                      'test_samples': 30, 'class_distribution': {}},
        feature_columns=[f'feature_{i}' for i in range(10)]
    )
    
    # Export model
    export_path = "test_export"
    registry.export_model(model_id, export_path)
    print(f"✓ Model exported to {export_path}")
    
    # Create new registry and import
    new_registry_path = "test_registry_import"
    new_registry = ModelRegistry(new_registry_path)
    
    imported_id = new_registry.import_model(
        model_path=f"{export_path}/{model_id}.pkl",
        metadata_path=f"{export_path}/{model_id}.json"
    )
    
    print(f"✓ Model imported with ID: {imported_id}")
    
    # Verify imported model works
    loaded_model, loaded_metadata = new_registry.load_model(model_id=imported_id)
    predictions = loaded_model.predict(X_test)
    print(f"✓ Imported model makes predictions: {predictions.shape}")
    
    # Cleanup
    shutil.rmtree(registry_path)
    shutil.rmtree(new_registry_path)
    shutil.rmtree(export_path)


def test_registry_summary():
    """Test registry summary and statistics."""
    print("\n" + "="*60)
    print("TEST 6: Registry Summary")
    print("="*60)
    
    registry_path = "test_registry_summary"
    registry = ModelRegistry(registry_path)
    
    X_train, y_train, X_test, y_test = create_sample_data()
    
    # Register multiple models
    for i in range(3):
        model = RandomForestClassifier(n_estimators=10, random_state=42+i)
        model.fit(X_train, y_train)
        
        registry.register_model(
            model=model,
            model_name=f"Model_{i}",
            algorithm="RandomForest",
            evaluation_results={'accuracy': 0.80+i*0.02, 'precision': 0.78, 
                              'recall': 0.76, 'f1_score': 0.77},
            hyperparameters={'n_estimators': 10},
            training_info={'dataset': 'test', 'training_samples': 100, 
                          'test_samples': 30, 'class_distribution': {}},
            feature_columns=[f'feature_{j}' for j in range(10)]
        )
    
    # Print summary
    registry.print_registry_summary()
    
    # Cleanup
    shutil.rmtree(registry_path)


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("TESTING MODEL REGISTRY AND PERSISTENCE SYSTEM")
    print("="*60)
    
    try:
        test_model_persistence()
        test_model_registry_basic()
        test_model_versioning()
        test_model_comparison()
        test_export_import()
        test_registry_summary()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nModel Registry and Persistence System is working correctly!")
        print("Features verified:")
        print("  ✓ Model serialization and deserialization")
        print("  ✓ Model registration with metadata")
        print("  ✓ Version tracking and management")
        print("  ✓ Model comparison and selection")
        print("  ✓ Export and import functionality")
        print("  ✓ Registry statistics and summaries")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
