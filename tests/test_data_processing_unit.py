"""
Unit tests for data processing components without requiring dataset download.
"""

import sys
import os
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from data.data_processor import (
    DataCleaner,
    FeatureNormalizer,
    CategoryEncoder,
    FeatureEngineer,
    DataSplitter,
    DataProcessor
)


def create_sample_koi_data():
    """Create a sample KOI dataset for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'koi_disposition': np.random.choice(['CONFIRMED', 'FALSE POSITIVE', 'CANDIDATE'], n_samples),
        'koi_period': np.random.uniform(0.5, 500, n_samples),
        'koi_duration': np.random.uniform(1, 10, n_samples),
        'koi_depth': np.random.uniform(10, 5000, n_samples),
        'koi_prad': np.random.uniform(0.5, 20, n_samples),
        'koi_teq': np.random.uniform(200, 2000, n_samples),
    }
    
    # Add some missing values
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 10, replace=False), 'koi_teq'] = np.nan
    df.loc[np.random.choice(df.index, 5, replace=False), 'koi_prad'] = np.nan
    
    return df


def test_data_cleaner():
    """Test DataCleaner component."""
    print("\n" + "="*80)
    print("TEST 1: DataCleaner")
    print("="*80)
    
    df = create_sample_koi_data()
    print(f"Original data shape: {df.shape}")
    print(f"Missing values before cleaning:\n{df.isnull().sum()}")
    
    cleaner = DataCleaner()
    df_clean = cleaner.handle_missing_values(df, strategy='auto')
    
    print(f"\nCleaned data shape: {df_clean.shape}")
    print(f"Missing values after cleaning:\n{df_clean.isnull().sum()}")
    
    assert df_clean.isnull().sum().sum() == 0, "Should have no missing values"
    print("\n✓ DataCleaner test passed!")
    
    return df_clean


def test_feature_normalizer():
    """Test FeatureNormalizer component."""
    print("\n" + "="*80)
    print("TEST 2: FeatureNormalizer")
    print("="*80)
    
    df = create_sample_koi_data()
    cleaner = DataCleaner()
    df_clean = cleaner.handle_missing_values(df, strategy='auto')
    
    print(f"Before normalization - koi_period range: [{df_clean['koi_period'].min():.2f}, {df_clean['koi_period'].max():.2f}]")
    
    normalizer = FeatureNormalizer()
    numeric_cols = ['koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq']
    df_normalized = normalizer.normalize_features(df_clean, numeric_cols, method='standard')
    
    print(f"After normalization - koi_period range: [{df_normalized['koi_period'].min():.2f}, {df_normalized['koi_period'].max():.2f}]")
    print(f"After normalization - koi_period mean: {df_normalized['koi_period'].mean():.6f}")
    print(f"After normalization - koi_period std: {df_normalized['koi_period'].std():.6f}")
    
    assert abs(df_normalized['koi_period'].mean()) < 1e-10, "Mean should be close to 0"
    assert abs(df_normalized['koi_period'].std() - 1.0) < 0.1, "Std should be close to 1"
    print("\n✓ FeatureNormalizer test passed!")
    
    return df_normalized, normalizer


def test_category_encoder():
    """Test CategoryEncoder component."""
    print("\n" + "="*80)
    print("TEST 3: CategoryEncoder")
    print("="*80)
    
    df = create_sample_koi_data()
    
    print(f"Original disposition values: {df['koi_disposition'].unique()}")
    print(f"Value counts:\n{df['koi_disposition'].value_counts()}")
    
    encoder = CategoryEncoder()
    df_encoded = encoder.encode_disposition(df, 'koi_disposition')
    
    print(f"\nEncoded disposition values: {df_encoded['koi_disposition'].unique()}")
    print(f"Value counts:\n{df_encoded['koi_disposition'].value_counts()}")
    print(f"Encoding mapping: {encoder.encoding_mappings['koi_disposition']}")
    
    assert df_encoded['koi_disposition'].dtype in [np.int64, np.int32], "Should be encoded as integers"
    print("\n✓ CategoryEncoder test passed!")
    
    return df_encoded


def test_feature_engineer():
    """Test FeatureEngineer component."""
    print("\n" + "="*80)
    print("TEST 4: FeatureEngineer")
    print("="*80)
    
    df = create_sample_koi_data()
    cleaner = DataCleaner()
    df_clean = cleaner.handle_missing_values(df, strategy='auto')
    
    print(f"Original columns: {list(df_clean.columns)}")
    
    engineer = FeatureEngineer()
    df_engineered = engineer.create_derived_features(df_clean, dataset_type='koi')
    
    print(f"\nColumns after feature engineering: {list(df_engineered.columns)}")
    
    derived_features = [col for col in df_engineered.columns if col not in df_clean.columns]
    print(f"\nDerived features created: {derived_features}")
    
    if 'period_duration_ratio' in df_engineered.columns:
        print(f"period_duration_ratio sample values: {df_engineered['period_duration_ratio'].head()}")
    
    assert len(derived_features) > 0, "Should create at least one derived feature"
    print("\n✓ FeatureEngineer test passed!")
    
    return df_engineered


def test_data_splitter():
    """Test DataSplitter component."""
    print("\n" + "="*80)
    print("TEST 5: DataSplitter")
    print("="*80)
    
    df = create_sample_koi_data()
    encoder = CategoryEncoder()
    df_encoded = encoder.encode_disposition(df, 'koi_disposition')
    
    print(f"Total samples: {len(df_encoded)}")
    
    splitter = DataSplitter(random_state=42)
    train_df, val_df, test_df = splitter.split_data(
        df_encoded, 
        target_col='koi_disposition',
        test_size=0.2,
        val_size=0.1,
        stratify=True
    )
    
    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")
    
    total = len(train_df) + len(val_df) + len(test_df)
    assert total == len(df_encoded), "Split sizes should sum to original size"
    print("\n✓ DataSplitter test passed!")
    
    return train_df, val_df, test_df


def test_data_processor_pipeline():
    """Test complete DataProcessor pipeline."""
    print("\n" + "="*80)
    print("TEST 6: Complete DataProcessor Pipeline")
    print("="*80)
    
    df = create_sample_koi_data()
    print(f"Input data shape: {df.shape}")
    print(f"Input missing values: {df.isnull().sum().sum()}")
    
    processor = DataProcessor()
    
    # Process dataset
    processed_data = processor.process_dataset(
        df,
        dataset_type='koi',
        target_col='koi_disposition',
        remove_outliers=False,  # Skip outlier removal for small test dataset
        create_features=True
    )
    
    print(f"\nProcessed data shape: {processed_data['data'].shape}")
    print(f"Target column: {processed_data['target_column']}")
    print(f"Number of features: {len(processed_data['feature_columns'])}")
    
    # Prepare for training
    training_data = processor.prepare_for_training(
        processed_data,
        test_size=0.2,
        val_size=0.1
    )
    
    print(f"\nTraining set shape: {training_data['X_train'].shape}")
    print(f"Validation set shape: {training_data['X_val'].shape}")
    print(f"Test set shape: {training_data['X_test'].shape}")
    
    print(f"\nTarget distribution:")
    print(f"  Train: {dict(training_data['y_train'].value_counts())}")
    print(f"  Val: {dict(training_data['y_val'].value_counts())}")
    print(f"  Test: {dict(training_data['y_test'].value_counts())}")
    
    assert training_data['X_train'].shape[1] == training_data['X_val'].shape[1] == training_data['X_test'].shape[1], \
        "All splits should have same number of features"
    
    print("\n✓ Complete pipeline test passed!")
    
    return training_data


def run_all_tests():
    """Run all unit tests."""
    print("="*80)
    print("DATA PROCESSING AND FEATURE ENGINEERING UNIT TESTS")
    print("="*80)
    
    try:
        # Run individual component tests
        test_data_cleaner()
        test_feature_normalizer()
        test_category_encoder()
        test_feature_engineer()
        test_data_splitter()
        
        # Run complete pipeline test
        test_data_processor_pipeline()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED!")
        print("="*80)
        print("\nData processing pipeline is working correctly.")
        print("Ready to process real NASA exoplanet datasets.")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
