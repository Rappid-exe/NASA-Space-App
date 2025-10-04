"""
Model registry and persistence system for managing trained exoplanet classification models.
Handles model serialization, versioning, metadata storage, and loading for inference.
"""

import os
import json
import pickle
import shutil
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path


@dataclass
class ModelMetadata:
    """Metadata for a trained model."""
    model_id: str
    model_name: str
    algorithm: str
    version: int
    training_dataset: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_date: str
    hyperparameters: Dict[str, Any]
    feature_columns: List[str]
    training_samples: int
    test_samples: int
    class_distribution: Dict[str, int]
    additional_metrics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create metadata from dictionary."""
        return cls(**data)


class ModelRegistry:
    """
    Registry for managing trained models with versioning and metadata.
    """
    
    def __init__(self, registry_path: str = "models/registry"):
        """
        Initialize the model registry.
        
        Args:
            registry_path: Base path for storing models and metadata
        """
        self.registry_path = Path(registry_path)
        self.models_path = self.registry_path / "models"
        self.metadata_path = self.registry_path / "metadata"
        self.index_file = self.registry_path / "index.json"
        
        # Create directory structure
        self._initialize_registry()
        
        # Load or create index
        self.index = self._load_index()
    
    def _initialize_registry(self) -> None:
        """Create registry directory structure if it doesn't exist."""
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(exist_ok=True)
        self.metadata_path.mkdir(exist_ok=True)
    
    def _load_index(self) -> Dict[str, Any]:
        """Load the registry index."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {"models": {}, "latest_versions": {}}
    
    def _save_index(self) -> None:
        """Save the registry index."""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def _generate_model_id(self, model_name: str, version: int) -> str:
        """Generate unique model ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model_name}_v{version}_{timestamp}"
    
    def _get_next_version(self, model_name: str) -> int:
        """Get the next version number for a model."""
        if model_name in self.index["latest_versions"]:
            return self.index["latest_versions"][model_name] + 1
        return 1

    def register_model(self, model: Any, model_name: str, algorithm: str,
                      evaluation_results: Dict[str, float],
                      hyperparameters: Dict[str, Any],
                      training_info: Dict[str, Any],
                      feature_columns: List[str],
                      version: Optional[int] = None) -> str:
        """
        Register a trained model with metadata.
        
        Args:
            model: Trained model object
            model_name: Name of the model
            algorithm: Algorithm type (e.g., 'RandomForest', 'NeuralNetwork')
            evaluation_results: Dictionary with accuracy, precision, recall, f1_score
            hyperparameters: Model hyperparameters
            training_info: Additional training information (dataset, samples, etc.)
            feature_columns: List of feature column names
            version: Specific version number (auto-increments if None)
            
        Returns:
            model_id: Unique identifier for the registered model
        """
        # Determine version
        if version is None:
            version = self._get_next_version(model_name)
        
        # Generate model ID
        model_id = self._generate_model_id(model_name, version)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            algorithm=algorithm,
            version=version,
            training_dataset=training_info.get('dataset', 'unknown'),
            accuracy=evaluation_results.get('accuracy', 0.0),
            precision=evaluation_results.get('precision', 0.0),
            recall=evaluation_results.get('recall', 0.0),
            f1_score=evaluation_results.get('f1_score', 0.0),
            training_date=datetime.now().isoformat(),
            hyperparameters=hyperparameters,
            feature_columns=feature_columns,
            training_samples=training_info.get('training_samples', 0),
            test_samples=training_info.get('test_samples', 0),
            class_distribution=training_info.get('class_distribution', {}),
            additional_metrics=evaluation_results.get('additional_metrics', {})
        )
        
        # Save model
        model_file = self.models_path / f"{model_id}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata_file = self.metadata_path / f"{model_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        # Update index
        if model_name not in self.index["models"]:
            self.index["models"][model_name] = []
        
        self.index["models"][model_name].append({
            "model_id": model_id,
            "version": version,
            "algorithm": algorithm,
            "f1_score": metadata.f1_score,
            "accuracy": metadata.accuracy,
            "training_date": metadata.training_date
        })
        
        self.index["latest_versions"][model_name] = version
        self._save_index()
        
        print(f"Model registered: {model_id}")
        print(f"  Algorithm: {algorithm}")
        print(f"  Version: {version}")
        print(f"  F1 Score: {metadata.f1_score:.4f}")
        print(f"  Accuracy: {metadata.accuracy:.4f}")
        
        return model_id
    
    def load_model(self, model_id: Optional[str] = None,
                  model_name: Optional[str] = None,
                  version: Optional[int] = None) -> tuple:
        """
        Load a model from the registry.
        
        Args:
            model_id: Specific model ID to load
            model_name: Model name (loads latest version if version not specified)
            version: Specific version to load
            
        Returns:
            Tuple of (model, metadata)
        """
        # Determine which model to load
        if model_id is None:
            if model_name is None:
                raise ValueError("Must provide either model_id or model_name")
            
            model_id = self._find_model_id(model_name, version)
        
        # Load model
        model_file = self.models_path / f"{model_id}.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        
        # Load metadata
        metadata = self.load_metadata(model_id)
        
        print(f"Loaded model: {model_id}")
        print(f"  Algorithm: {metadata.algorithm}")
        print(f"  Version: {metadata.version}")
        print(f"  F1 Score: {metadata.f1_score:.4f}")
        
        return model, metadata
    
    def load_metadata(self, model_id: str) -> ModelMetadata:
        """
        Load metadata for a specific model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            ModelMetadata object
        """
        metadata_file = self.metadata_path / f"{model_id}.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            metadata_dict = json.load(f)
        
        return ModelMetadata.from_dict(metadata_dict)
    
    def _find_model_id(self, model_name: str, version: Optional[int] = None) -> str:
        """Find model ID by name and version."""
        if model_name not in self.index["models"]:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        models = self.index["models"][model_name]
        
        if version is None:
            # Get latest version
            version = self.index["latest_versions"][model_name]
        
        # Find model with matching version
        for model_info in models:
            if model_info["version"] == version:
                return model_info["model_id"]
        
        raise ValueError(f"Model '{model_name}' version {version} not found")

    def list_models(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all registered models or models with a specific name.
        
        Args:
            model_name: Optional model name to filter by
            
        Returns:
            List of model information dictionaries
        """
        if model_name:
            if model_name not in self.index["models"]:
                return []
            return self.index["models"][model_name]
        
        # Return all models
        all_models = []
        for name, models in self.index["models"].items():
            all_models.extend(models)
        return all_models
    
    def get_best_model(self, model_name: Optional[str] = None,
                      metric: str = 'f1_score') -> tuple:
        """
        Get the best performing model based on a metric.
        
        Args:
            model_name: Optional model name to filter by
            metric: Metric to use for comparison ('f1_score', 'accuracy', etc.)
            
        Returns:
            Tuple of (model, metadata) for best model
        """
        models = self.list_models(model_name)
        
        if not models:
            raise ValueError("No models found in registry")
        
        # Find best model by metric
        best_model_info = max(models, key=lambda x: x.get(metric, 0))
        
        return self.load_model(model_id=best_model_info["model_id"])
    
    def delete_model(self, model_id: str) -> None:
        """
        Delete a model from the registry.
        
        Args:
            model_id: Model identifier to delete
        """
        # Delete model file
        model_file = self.models_path / f"{model_id}.pkl"
        if model_file.exists():
            model_file.unlink()
        
        # Delete metadata file
        metadata_file = self.metadata_path / f"{model_id}.json"
        if metadata_file.exists():
            metadata_file.unlink()
        
        # Update index
        for model_name, models in self.index["models"].items():
            self.index["models"][model_name] = [
                m for m in models if m["model_id"] != model_id
            ]
        
        self._save_index()
        print(f"Deleted model: {model_id}")
    
    def compare_models(self, model_ids: Optional[List[str]] = None,
                      model_name: Optional[str] = None) -> None:
        """
        Compare multiple models and print comparison table.
        
        Args:
            model_ids: List of specific model IDs to compare
            model_name: Compare all versions of a specific model name
        """
        if model_ids:
            models_to_compare = [self.load_metadata(mid) for mid in model_ids]
        elif model_name:
            model_infos = self.list_models(model_name)
            models_to_compare = [self.load_metadata(m["model_id"]) for m in model_infos]
        else:
            # Compare all models
            all_model_infos = self.list_models()
            models_to_compare = [self.load_metadata(m["model_id"]) for m in all_model_infos]
        
        if not models_to_compare:
            print("No models to compare")
            return
        
        # Print comparison table
        print("\n" + "="*100)
        print("MODEL COMPARISON")
        print("="*100)
        print(f"{'Model ID':<40} {'Algorithm':<15} {'Ver':<5} {'Accuracy':<10} {'F1':<10} {'Precision':<10} {'Recall':<10}")
        print("-"*100)
        
        for metadata in sorted(models_to_compare, key=lambda x: x.f1_score, reverse=True):
            print(f"{metadata.model_id:<40} {metadata.algorithm:<15} {metadata.version:<5} "
                  f"{metadata.accuracy:<10.4f} {metadata.f1_score:<10.4f} "
                  f"{metadata.precision:<10.4f} {metadata.recall:<10.4f}")
        
        print("="*100 + "\n")
    
    def export_model(self, model_id: str, export_path: str) -> None:
        """
        Export a model and its metadata to a specified path.
        
        Args:
            model_id: Model identifier to export
            export_path: Directory path to export to
        """
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model file
        model_file = self.models_path / f"{model_id}.pkl"
        if model_file.exists():
            shutil.copy(model_file, export_dir / f"{model_id}.pkl")
        
        # Copy metadata file
        metadata_file = self.metadata_path / f"{model_id}.json"
        if metadata_file.exists():
            shutil.copy(metadata_file, export_dir / f"{model_id}.json")
        
        print(f"Model exported to: {export_path}")
    
    def import_model(self, model_path: str, metadata_path: str) -> str:
        """
        Import a model and its metadata into the registry.
        
        Args:
            model_path: Path to model pickle file
            metadata_path: Path to metadata JSON file
            
        Returns:
            model_id: Identifier of imported model
        """
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        metadata = ModelMetadata.from_dict(metadata_dict)
        model_id = metadata.model_id
        
        # Copy model file
        shutil.copy(model_path, self.models_path / f"{model_id}.pkl")
        
        # Copy metadata file
        shutil.copy(metadata_path, self.metadata_path / f"{model_id}.json")
        
        # Update index
        model_name = metadata.model_name
        if model_name not in self.index["models"]:
            self.index["models"][model_name] = []
        
        self.index["models"][model_name].append({
            "model_id": model_id,
            "version": metadata.version,
            "algorithm": metadata.algorithm,
            "f1_score": metadata.f1_score,
            "accuracy": metadata.accuracy,
            "training_date": metadata.training_date
        })
        
        # Update latest version if necessary
        if model_name not in self.index["latest_versions"] or \
           metadata.version > self.index["latest_versions"][model_name]:
            self.index["latest_versions"][model_name] = metadata.version
        
        self._save_index()
        print(f"Model imported: {model_id}")
        
        return model_id
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the model registry.
        
        Returns:
            Dictionary with registry statistics
        """
        total_models = sum(len(models) for models in self.index["models"].values())
        
        stats = {
            "total_models": total_models,
            "model_types": len(self.index["models"]),
            "model_names": list(self.index["models"].keys()),
            "latest_versions": self.index["latest_versions"]
        }
        
        return stats
    
    def print_registry_summary(self) -> None:
        """Print a summary of the model registry."""
        stats = self.get_registry_stats()
        
        print("\n" + "="*60)
        print("MODEL REGISTRY SUMMARY")
        print("="*60)
        print(f"Total Models: {stats['total_models']}")
        print(f"Model Types: {stats['model_types']}")
        print(f"\nRegistered Models:")
        for model_name in stats['model_names']:
            latest_version = stats['latest_versions'][model_name]
            model_count = len(self.index["models"][model_name])
            print(f"  - {model_name}: {model_count} version(s) (latest: v{latest_version})")
        print("="*60 + "\n")
