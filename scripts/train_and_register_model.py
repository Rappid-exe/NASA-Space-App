"""
Train models with real NASA data and register the best one.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data.dataset_downloader import DatasetDownloader
from data.dataset_loader import DatasetLoader
from data.data_processor import DataProcessor
from models.model_trainer import ModelTrainer
from models.model_registry import ModelRegistry

def main():
    print("="*80)
    print("EXOPLANET CLASSIFIER - TRAINING WITH NASA DATA")
    print("="*80)
    
    # Step 1: Download NASA data
    print("\n1. Downloading NASA KOI dataset...")
    downloader = DatasetDownloader()
    try:
        koi_path = downloader.download_koi_dataset()
        print(f"✓ Downloaded: {koi_path}")
    except Exception as e:
        print(f"✗ Download failed: {e}")
        print("Using cached data if available...")
        koi_path = "data/raw/koi_dataset.csv"
    
    # Step 2: Load dataset
    print("\n2. Loading dataset...")
    loader = DatasetLoader()
    df = loader.load_dataset(koi_path)
    print(f"✓ Loaded {len(df)} observations")
    
    # Step 3: Process data
    print("\n3. Processing data...")
    processor = DataProcessor()
    processed = processor.process_dataset(df, dataset_type='koi', target_col='koi_disposition')
    
    # Step 4: Prepare for training
    print("\n4. Preparing train/val/test splits...")
    splits = processor.prepare_for_training(processed, test_size=0.2, val_size=0.1)
    
    X_train = splits['X_train'].values
    y_train = splits['y_train'].values
    X_val = splits['X_val'].values
    y_val = splits['y_val'].values
    X_test = splits['X_test'].values
    y_test = splits['y_test'].values
    
    print(f"✓ Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Step 5: Train models
    print("\n5. Training models...")
    trainer = ModelTrainer()
    trainer.initialize_models()
    trainer.train_models(X_train, y_train, X_val, y_val, feature_names=splits['feature_columns'])
    
    # Step 6: Evaluate models
    print("\n6. Evaluating models...")
    results = trainer.evaluate_models(X_test, y_test)
    
    # Step 7: Register best model
    print("\n7. Registering best model...")
    registry = ModelRegistry()
    
    best_model_name = trainer.best_model_name
    best_model = trainer.models[best_model_name]
    best_results = results[best_model_name]
    
    model_id = registry.register_model(
        model=best_model.model,
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
        hyperparameters=best_model.training_metadata,
        training_info={
            'dataset': 'NASA KOI',
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'class_distribution': {
                'false_positive': int((y_train == 0).sum()),
                'confirmed': int((y_train == 1).sum())
            }
        },
        feature_columns=splits['feature_columns']
    )
    
    print(f"\n✓ Model registered with ID: {model_id}")
    
    # Step 8: Summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Best Model: {best_model_name}")
    print(f"Accuracy: {best_results['accuracy']:.4f}")
    print(f"F1 Score: {best_results['f1_score']:.4f}")
    print(f"Model ID: {model_id}")
    print("\nYou can now start the API server:")
    print("  python -m uvicorn api.main:app --reload")
    print("="*80)

if __name__ == "__main__":
    main()
