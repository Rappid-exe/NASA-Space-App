"""
Train exoplanet classification model with real NASA Kepler data.
Uses the cumulative KOI dataset downloaded from NASA Exoplanet Archive.
"""

import pandas as pd
import numpy as np
from data.data_processor import DataProcessor
from models.model_trainer import ModelTrainer
from models.model_registry import ModelRegistry

def main():
    print("="*80)
    print("TRAINING EXOPLANET CLASSIFIER WITH REAL NASA KEPLER DATA")
    print("="*80)
    
    # Load real NASA data
    print("\n1. Loading NASA Kepler KOI dataset...")
    df = pd.read_csv('cumulative_2025.10.04_10.12.10.csv', comment='#')
    print(f"   Loaded {len(df)} observations")
    print(f"   Columns: {len(df.columns)}")
    
    # Show disposition distribution
    print("\n2. Dataset composition:")
    disposition_counts = df['koi_disposition'].value_counts()
    for disp, count in disposition_counts.items():
        print(f"   {disp}: {count} ({count/len(df)*100:.1f}%)")
    
    # Select relevant columns for training
    print("\n3. Selecting features for training...")
    feature_columns = [
        'koi_period',      # Orbital period (days)
        'koi_duration',    # Transit duration (hours)
        'koi_depth',       # Transit depth (ppm)
        'koi_prad',        # Planetary radius (Earth radii)
        'koi_teq',         # Equilibrium temperature (K)
        'koi_impact',      # Impact parameter
        'koi_insol',       # Insolation flux (Earth flux)
        'koi_steff',       # Stellar effective temperature
        'koi_slogg',       # Stellar surface gravity
        'koi_srad'         # Stellar radius (solar radii)
    ]
    
    # Keep only rows with required columns
    required_cols = ['koi_disposition'] + feature_columns
    df_clean = df[required_cols].copy()
    
    # Remove rows with missing values
    initial_count = len(df_clean)
    df_clean = df_clean.dropna()
    print(f"   Removed {initial_count - len(df_clean)} rows with missing values")
    print(f"   Training dataset: {len(df_clean)} observations")
    
    # Process the data
    print("\n4. Processing data...")
    processor = DataProcessor()
    
    processed_data = processor.process_dataset(
        df_clean,
        dataset_type='koi',
        target_col='koi_disposition',
        remove_outliers=True,
        create_features=True
    )
    
    print(f"   Features after processing: {len(processed_data['feature_columns'])}")
    
    # Get target values
    target_col = processed_data['target_column']
    y_values = processed_data['data'][target_col].values
    
    print(f"   Class distribution:")
    for label, count in zip(*np.unique(y_values, return_counts=True)):
        label_name = "CONFIRMED" if label == 1 else "FALSE POSITIVE"
        print(f"     {label_name}: {count} ({count/len(y_values)*100:.1f}%)")
    
    # Prepare training data
    print("\n5. Splitting data into train/validation/test sets...")
    training_data = processor.prepare_for_training(
        processed_data,
        test_size=0.2,
        val_size=0.1
    )
    
    X_train = training_data['X_train']
    y_train = training_data['y_train']
    X_val = training_data['X_val']
    y_val = training_data['y_val']
    X_test = training_data['X_test']
    y_test = training_data['y_test']
    
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Validation set: {len(X_val)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Train models
    print("\n6. Training machine learning models...")
    print("   This may take a few minutes...\n")
    
    trainer = ModelTrainer(random_state=42)
    results = trainer.train_and_evaluate(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        feature_names=processed_data['feature_columns']
    )
    
    # Display results
    print("\n" + "="*80)
    print("TRAINING RESULTS")
    print("="*80)
    
    for model_name, eval_result in results['evaluation_results'].items():
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {eval_result['accuracy']:.4f}")
        print(f"  Precision: {eval_result['precision']:.4f}")
        print(f"  Recall:    {eval_result['recall']:.4f}")
        print(f"  F1 Score:  {eval_result['f1_score']:.4f}")
    
    print(f"\n{'='*80}")
    print(f"BEST MODEL: {results['best_model_name']}")
    print(f"F1 Score: {results['evaluation_results'][results['best_model_name']]['f1_score']:.4f}")
    print(f"{'='*80}")
    
    # Register models in registry
    print("\n7. Registering models in registry...")
    registry = ModelRegistry("models/registry")
    
    for model_name, model in results['models'].items():
        model_id = registry.register_model(
            model=model,
            model_name=f"kepler_{model_name.lower().replace(' ', '_')}",
            algorithm=model_name,
            evaluation_results=results['evaluation_results'][model_name],
            hyperparameters={},
            training_info={
                'dataset': 'NASA Kepler KOI (Real Data)',
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'class_distribution': {
                    'CONFIRMED': int(np.sum(y_train == 1)),
                    'FALSE_POSITIVE': int(np.sum(y_train == 0))
                },
                'source_file': 'cumulative_2025.10.04_10.12.10.csv',
                'total_observations': len(df_clean)
            },
            feature_columns=processed_data['feature_columns']
        )
    
    # Show registry summary
    print("\n8. Model Registry Summary:")
    registry.print_registry_summary()
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print("\nYour model is ready to use!")
    print("\nNext steps:")
    print("1. Start the API server:")
    print("   python -m uvicorn api.main:app --reload")
    print("\n2. Start the frontend:")
    print("   cd frontend && npm run dev")
    print("\n3. Open http://localhost:3000 in your browser")
    print("="*80)

if __name__ == "__main__":
    main()
