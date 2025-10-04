# Model Registry and Persistence System Guide

## Overview

The Model Registry and Persistence System provides a comprehensive solution for managing trained machine learning models in the Exoplanet Identifier project. It handles model serialization, versioning, metadata storage, and loading for inference pipelines.

## Features

### 1. Model Serialization
- Save and load trained models with pickle serialization
- Preserve model state and metadata
- Support for all classifier types (RandomForest, NeuralNetwork, SVM)

### 2. Model Versioning
- Automatic version tracking for each model
- Load specific versions or latest version
- Version history with performance metrics

### 3. Metadata Storage
- Comprehensive metadata for each model:
  - Performance metrics (accuracy, precision, recall, F1-score)
  - Hyperparameters
  - Training information (dataset, samples, class distribution)
  - Feature columns
  - Training date and timestamps

### 4. Model Registry
- Centralized repository for all trained models
- Index-based lookup system
- Model comparison and selection
- Export/import functionality

## Architecture

```
models/registry/
├── models/              # Serialized model files (.pkl)
├── metadata/            # Model metadata files (.json)
└── index.json          # Registry index
```

## Components

### ModelMetadata
Dataclass that stores comprehensive information about a trained model.

**Attributes:**
- `model_id`: Unique identifier
- `model_name`: Human-readable name
- `algorithm`: Algorithm type (RandomForest, NeuralNetwork, SVM)
- `version`: Version number
- `training_dataset`: Dataset used for training
- `accuracy`, `precision`, `recall`, `f1_score`: Performance metrics
- `training_date`: ISO format timestamp
- `hyperparameters`: Model configuration
- `feature_columns`: List of feature names
- `training_samples`, `test_samples`: Dataset sizes
- `class_distribution`: Distribution of classes in training data

### ModelRegistry
Main class for managing the model registry.

**Key Methods:**

#### `register_model()`
Register a trained model with metadata.

```python
model_id = registry.register_model(
    model=trained_model,
    model_name="ExoplanetClassifier",
    algorithm="RandomForest",
    evaluation_results={
        'accuracy': 0.95,
        'precision': 0.93,
        'recall': 0.94,
        'f1_score': 0.935
    },
    hyperparameters={'n_estimators': 100, 'max_depth': 20},
    training_info={
        'dataset': 'cumulative_2025',
        'training_samples': 5000,
        'test_samples': 1000,
        'class_distribution': {'CONFIRMED': 2000, 'FALSE POSITIVE': 3000}
    },
    feature_columns=['koi_period', 'koi_duration', ...]
)
```

#### `load_model()`
Load a model from the registry.

```python
# Load by model ID
model, metadata = registry.load_model(model_id="ExoplanetClassifier_v1_20250104_123456")

# Load latest version by name
model, metadata = registry.load_model(model_name="ExoplanetClassifier")

# Load specific version
model, metadata = registry.load_model(model_name="ExoplanetClassifier", version=2)
```

#### `get_best_model()`
Get the best performing model based on a metric.

```python
# Get best model overall
best_model, metadata = registry.get_best_model()

# Get best model of specific type
best_rf, metadata = registry.get_best_model(model_name="RandomForest")

# Use different metric
best_model, metadata = registry.get_best_model(metric='accuracy')
```

#### `list_models()`
List all registered models or filter by name.

```python
# List all models
all_models = registry.list_models()

# List specific model versions
rf_versions = registry.list_models("RandomForest")
```

#### `compare_models()`
Compare multiple models with a formatted table.

```python
# Compare all models
registry.compare_models()

# Compare specific models
registry.compare_models(model_ids=["model_1", "model_2"])

# Compare all versions of a model
registry.compare_models(model_name="RandomForest")
```

#### `export_model()` / `import_model()`
Export and import models for sharing or backup.

```python
# Export
registry.export_model(model_id="model_123", export_path="exports/")

# Import
imported_id = registry.import_model(
    model_path="exports/model_123.pkl",
    metadata_path="exports/model_123.json"
)
```

### ModelPersistence
Utility class for model serialization and persistence.

**Key Methods:**

#### `save_model()` / `load_model()`
Basic save and load operations.

```python
# Save
ModelPersistence.save_model(
    model=trained_model,
    filepath="models/my_model.pkl",
    metadata={'notes': 'Best performing model'}
)

# Load
model, metadata = ModelPersistence.load_model("models/my_model.pkl")
```

#### `save_model_checkpoint()`
Save model checkpoints with timestamps.

```python
checkpoint_path = ModelPersistence.save_model_checkpoint(
    model=model,
    checkpoint_dir="checkpoints/",
    checkpoint_name="exoplanet_rf",
    metadata={'epoch': 10, 'loss': 0.15}
)
```

#### `export_model_for_deployment()`
Export model in deployment-ready format with documentation.

```python
ModelPersistence.export_model_for_deployment(
    model=best_model,
    export_path="deployment/production",
    model_info={
        'model_name': 'ExoplanetClassifier',
        'version': 1,
        'accuracy': 0.95,
        'feature_columns': feature_names
    }
)
```

## Usage Examples

### Example 1: Train and Register Models

```python
from models.model_trainer import ModelTrainer
from models.model_registry import ModelRegistry

# Train models
trainer = ModelTrainer()
results = trainer.train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test)

# Initialize registry
registry = ModelRegistry("models/registry")

# Register each model
for model_name, model in results['models'].items():
    evaluation = results['evaluation_results'][model_name]
    
    model_id = registry.register_model(
        model=model,
        model_name=model_name,
        algorithm=model_name,
        evaluation_results=evaluation,
        hyperparameters={},
        training_info={'dataset': 'kepler', 'training_samples': len(X_train)},
        feature_columns=feature_names
    )
```

### Example 2: Load Best Model for Inference

```python
from models.model_registry import ModelRegistry

# Initialize registry
registry = ModelRegistry("models/registry")

# Load best model
best_model, metadata = registry.get_best_model()

# Make predictions
predictions = best_model.predict(X_new)
probabilities = best_model.predict_proba(X_new)

print(f"Using {metadata.algorithm} v{metadata.version}")
print(f"Model F1 Score: {metadata.f1_score:.4f}")
```

### Example 3: Version Management

```python
# List all versions
versions = registry.list_models("ExoplanetClassifier")
for v in versions:
    print(f"Version {v['version']}: F1={v['f1_score']:.4f}")

# Load specific version
model_v2, metadata = registry.load_model(
    model_name="ExoplanetClassifier",
    version=2
)

# Compare versions
registry.compare_models(model_name="ExoplanetClassifier")
```

### Example 4: Model Deployment

```python
from models.model_registry import ModelRegistry
from models.model_persistence import ModelPersistence

# Get best model
registry = ModelRegistry("models/registry")
best_model, metadata = registry.get_best_model()

# Export for deployment
ModelPersistence.export_model_for_deployment(
    model=best_model,
    export_path="deployment/production",
    model_info={
        'model_name': metadata.model_name,
        'algorithm': metadata.algorithm,
        'version': metadata.version,
        'accuracy': metadata.accuracy,
        'f1_score': metadata.f1_score,
        'feature_columns': metadata.feature_columns
    }
)
```

## Integration with Existing Code

The model registry integrates seamlessly with the existing training pipeline:

```python
from models.model_trainer import ModelTrainer
from models.model_registry import ModelRegistry
from data.data_processor import DataProcessor

# Process data
processor = DataProcessor()
X_train, X_val, X_test, y_train, y_val, y_test, features, encoder = \
    processor.process_dataset(df)

# Train models
trainer = ModelTrainer()
results = trainer.train_and_evaluate(
    X_train, y_train, X_val, y_val, X_test, y_test,
    feature_names=features
)

# Register models
registry = ModelRegistry()
for name, model in results['models'].items():
    registry.register_model(
        model=model,
        model_name=name,
        algorithm=name,
        evaluation_results=results['evaluation_results'][name],
        hyperparameters={},
        training_info={'dataset': 'kepler', 'training_samples': len(X_train)},
        feature_columns=features
    )

# Use best model
best_model, metadata = registry.get_best_model()
```

## Testing

Run the comprehensive test suite:

```bash
python test_model_registry.py
```

This tests:
- Model serialization and deserialization
- Model registration with metadata
- Version tracking and management
- Model comparison and selection
- Export and import functionality
- Registry statistics and summaries

## Best Practices

1. **Always register models after training** to maintain a history of experiments
2. **Use meaningful model names** that describe the model type or purpose
3. **Include comprehensive metadata** to track training conditions
4. **Version models incrementally** when making improvements
5. **Compare models** before selecting for deployment
6. **Export models** for production with full documentation
7. **Backup the registry** regularly to prevent data loss

## File Structure

```
models/
├── __init__.py
├── base_classifier.py
├── model_registry.py          # NEW: Registry implementation
├── model_persistence.py       # NEW: Persistence utilities
├── model_trainer.py
├── model_evaluator.py
├── random_forest_classifier.py
├── neural_network_classifier.py
├── svm_classifier.py
└── registry/                  # NEW: Registry storage
    ├── models/
    ├── metadata/
    └── index.json

test_model_registry.py         # NEW: Comprehensive tests
example_model_registry_usage.py # NEW: Usage examples
```

## Requirements Satisfied

This implementation satisfies the following requirements from the spec:

- **Requirement 3.4**: Model persistence and loading for inference
- **Requirement 5.4**: Model versioning and tracking for retraining

## Next Steps

The model registry is now ready for use in:
- Task 5: Inference API (load models for serving)
- Task 7: Advanced features (model retraining and hyperparameter tuning)
- Task 9: Deployment preparation (export models for production)
