"""
Model persistence utilities for saving and loading trained models.
Provides high-level interface for model serialization and deserialization.
"""

import os
import pickle
import json
from typing import Any, Dict, Optional
from pathlib import Path
from datetime import datetime


class ModelPersistence:
    """
    Utilities for model serialization and persistence.
    """
    
    @staticmethod
    def save_model(model: Any, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save a model to disk with optional metadata.
        
        Args:
            model: Model object to save
            filepath: Path to save the model
            metadata: Optional metadata dictionary
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare model data
        model_data = {
            'model': model,
            'saved_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        # Save model
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Model saved to: {filepath}")
        
        # Save metadata separately as JSON if provided
        if metadata:
            metadata_path = filepath.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Metadata saved to: {metadata_path}")
    
    @staticmethod
    def load_model(filepath: str) -> tuple:
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Tuple of (model, metadata)
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load model
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data.get('model')
        metadata = model_data.get('metadata', {})
        
        print(f"Model loaded from: {filepath}")
        
        return model, metadata
    
    @staticmethod
    def save_model_checkpoint(model: Any, checkpoint_dir: str, 
                             checkpoint_name: str, metadata: Optional[Dict] = None) -> str:
        """
        Save a model checkpoint with timestamp.
        
        Args:
            model: Model object to save
            checkpoint_dir: Directory for checkpoints
            checkpoint_name: Base name for checkpoint
            metadata: Optional metadata
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = checkpoint_dir / f"{checkpoint_name}_{timestamp}.pkl"
        
        ModelPersistence.save_model(model, str(checkpoint_file), metadata)
        
        return str(checkpoint_file)
    
    @staticmethod
    def load_latest_checkpoint(checkpoint_dir: str, checkpoint_name: str) -> tuple:
        """
        Load the most recent checkpoint for a given name.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            checkpoint_name: Base name of checkpoint
            
        Returns:
            Tuple of (model, metadata)
        """
        checkpoint_dir = Path(checkpoint_dir)
        
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        # Find all matching checkpoints
        checkpoints = list(checkpoint_dir.glob(f"{checkpoint_name}_*.pkl"))
        
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found for: {checkpoint_name}")
        
        # Get most recent checkpoint
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        return ModelPersistence.load_model(str(latest_checkpoint))
    
    @staticmethod
    def export_model_for_deployment(model: Any, export_path: str,
                                   model_info: Dict[str, Any]) -> None:
        """
        Export a model in a deployment-ready format.
        
        Args:
            model: Model object to export
            export_path: Path to export the model
            model_info: Model information and metadata
        """
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = export_path / "model.pkl"
        ModelPersistence.save_model(model, str(model_file))
        
        # Save model info
        info_file = export_path / "model_info.json"
        with open(info_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Create README
        readme_file = export_path / "README.md"
        readme_content = f"""# Exoplanet Classification Model

## Model Information
- **Model Name**: {model_info.get('model_name', 'Unknown')}
- **Algorithm**: {model_info.get('algorithm', 'Unknown')}
- **Version**: {model_info.get('version', 'Unknown')}
- **Training Date**: {model_info.get('training_date', 'Unknown')}

## Performance Metrics
- **Accuracy**: {model_info.get('accuracy', 'N/A')}
- **F1 Score**: {model_info.get('f1_score', 'N/A')}
- **Precision**: {model_info.get('precision', 'N/A')}
- **Recall**: {model_info.get('recall', 'N/A')}

## Usage
```python
from models.model_persistence import ModelPersistence

# Load model
model, metadata = ModelPersistence.load_model('model.pkl')

# Make predictions
predictions = model.predict(X)
```

## Features
{json.dumps(model_info.get('feature_columns', []), indent=2)}
"""
        
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        print(f"Model exported for deployment to: {export_path}")
    
    @staticmethod
    def validate_model_file(filepath: str) -> bool:
        """
        Validate that a model file can be loaded.
        
        Args:
            filepath: Path to model file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            model, metadata = ModelPersistence.load_model(filepath)
            return model is not None
        except Exception as e:
            print(f"Model validation failed: {e}")
            return False
