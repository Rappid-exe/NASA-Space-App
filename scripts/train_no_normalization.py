"""
Train exoplanet classification model WITHOUT normalization.
This makes inference simpler as we don't need to apply scaling.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from models.model_trainer import ModelTrainer
from models.model_registry import ModelRegistry
from data.data_processor import CategoryEncoder

def create_derived_features(df):
    """Create derived features manually."""
    df = df.copy()
    
    # Period-Duration Ratio
    df['period_duration_ratio'] = df['koi_period'] / (df['koi_duration'] / 24.0)
    
    # Depth-Radius Correlation
    df['depth_radius_correlation'] = df['koi_depth'] / (df['koi_prad'] ** 2)
    
    # Temperature habitable zone
    df['temp_habitable_zone'] = ((df['koi_teq'] >= 200) & (df['koi_teq'] <= 350)).astype(int)
    
    # Period categories
    df['period_category'] = pd.cut(df['koi_period'], 
                                    bins=[0, 10, 100, 1000, float('inf')],
                                    labels=[0, 1, 2, 3]).astype(int)
    
    # Radius categories
    df['radius_category'] = pd.cut(df['koi_prad'],
                                    bins=[0, 1.5, 2.5, 6, float('inf')],
                                    labels=[0, 1, 2, 3]).astype(int)
    
    # Transit SNR
    df['transit_snr'] = np.log10(df['koi_depth'] + 1)
    
    return df

def main():
    print("="*80)
    print("TRAINING MODEL WITHOUT NORMALIZATION - For Easy Inference")
    print("="*80)
    
    # Load real NASA data
    print("\n1. Loading NASA Kepler KOI dataset...")
    df = pd.read_csv('cumulative_2025.10.04_10.12.10.csv', comment='#')
    print(f"   Loaded {len(df)} observations")
    
    # Select features
    print("\n2. Selecting features...")
    feature_columns = [
        'koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq'
    ]
    
    required_cols = ['koi_disposition'] + feature_columns
    df_clean = df[required_cols].copy()
    
    # Remove missing values
    df_clean = df_clean.dropna()
    print(f"   Clean dataset: {len(df_clean)} observations")
    
    # Encode target
    print("\n3. Encoding target variable...")
    encoder = CategoryEncoder()
    df_clean = encoder.encode_disposition(df_clean, 'koi_disposition')
    
    # Create derived features
    print("\n4. Creating derived features...")
    df_processed = create_derived_features(df_clean)
    
    # Remove outliers using IQR
    print("\n5. Removing outliers...")
    numeric_cols = ['koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq',
                    'period_duration_ratio', 'depth_radius_correlation', 'transit_snr']
    
    for col in numeric_cols:
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 3.0 * IQR
        upper = Q3 + 3.0 * IQR
        df_processed = df_processed[(df_processed[col] >= lower) & (df_processed[col] <= upper)]
    
    print(f"   After outlier removal: {len(df_processed)} observations")
    
    # Prepare X and y
    feature_cols = [
        'koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq',
        'period_duration_ratio', 'depth_radius_correlation', 'temp_habitable_zone',
        'period_category', 'radius_category', 'transit_snr'
    ]
    
    X = df_processed[feature_cols].values
    y = df_processed['koi_disposition'].values
    
    print(f"\n   Features: {len(feature_cols)}")
    print(f"   Feature list: {feature_cols}")
    print(f"   Class distribution:")
    for label, count in zip(*np.unique(y, return_counts=True)):
        label_name = "CONFIRMED" if label == 1 else "FALSE POSITIVE"
        print(f"     {label_name}: {count} ({count/len(y)*100:.1f}%)")
    
    # Split data
    print("\n6. Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
    )
    
    print(f"   Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    # Train models
    print("\n7. Training models (NO NORMALIZATION)...")
    trainer = ModelTrainer(random_state=42)
    results = trainer.train_and_evaluate(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        feature_names=feature_cols
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
    
    # Register models
    print("\n8. Registering models...")
    registry = ModelRegistry("models/registry")
    
    for model_name, model in results['models'].items():
        model_id = registry.register_model(
            model=model,
            model_name=f"raw_{model_name.lower().replace(' ', '_')}",
            algorithm=model_name,
            evaluation_results=results['evaluation_results'][model_name],
            hyperparameters={'normalization': 'none'},
            training_info={
                'dataset': 'NASA Kepler KOI (Real Data - NO NORMALIZATION)',
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'class_distribution': {
                    'CONFIRMED': int(np.sum(y_train == 1)),
                    'FALSE_POSITIVE': int(np.sum(y_train == 0))
                },
                'source_file': 'cumulative_2025.10.04_10.12.10.csv',
                'total_observations': len(df_processed),
                'normalization': 'NONE - Raw features for easy inference'
            },
            feature_columns=feature_cols
        )
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print("\nModel trained WITHOUT normalization for easy inference!")
    print("\nNext steps:")
    print("1. Restart API: python -m uvicorn api.main:app --reload")
    print("2. Test at: http://localhost:3000")
    print("="*80)

if __name__ == "__main__":
    main()
