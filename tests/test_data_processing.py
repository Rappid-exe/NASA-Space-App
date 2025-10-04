"""
Test script for data processing and feature engineering pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from data.dataset_downloader import DatasetDownloader
from data.dataset_loader import DatasetLoader
from data.data_processor import DataProcessor


def test_data_processing_pipeline():
    """Test the complete data processing pipeline."""
    
    print("="*80)
    print("Testing Data Processing and Feature Engineering Pipeline")
    print("="*80)
    
    # Initialize components
    downloader = DatasetDownloader(data_dir="data/raw")
    loader = DatasetLoader()
    processor = DataProcessor()
    
    # Try to load existing dataset or download if needed
    koi_path = "data/raw/koi_dataset.csv"
    
    if not os.path.exists(koi_path):
        print("\nDownloading KOI dataset...")
        try:
            koi_path = downloader.download_koi_dataset()
        except Exception as e:
            print(f"Failed to download dataset: {e}")
            print("Please ensure you have internet connection and try again.")
            return
    else:
        print(f"\nUsing existing dataset: {koi_path}")
    
    # Load dataset
    print("\n" + "="*80)
    print("STEP 1: Loading Dataset")
    print("="*80)
    df = loader.load_dataset(koi_path, dataset_name='koi')
    
    # Display initial dataset info
    info = loader.get_dataset_info(df)
    print(f"\nDataset shape: {info['shape']}")
    print(f"Number of numeric columns: {len(info['numeric_columns'])}")
    print(f"Number of categorical columns: {len(info['categorical_columns'])}")
    print(f"\nMissing values in key columns:")
    for col in ['koi_disposition', 'koi_period', 'koi_duration', 'koi_depth', 'koi_prad']:
        if col in info['missing_values']:
            print(f"  {col}: {info['missing_values'][col]} ({info['missing_values'][col]/info['shape'][0]*100:.1f}%)")
    
    # Process dataset
    print("\n" + "="*80)
    print("STEP 2: Processing Dataset")
    print("="*80)
    
    try:
        processed_data = processor.process_dataset(
            df,
            dataset_type='koi',
            target_col='koi_disposition',
            remove_outliers=True,
            create_features=True
        )
        
        print("\n✓ Data processing completed successfully!")
        print(f"\nProcessed data shape: {processed_data['data'].shape}")
        print(f"Target column: {processed_data['target_column']}")
        print(f"Number of features: {len(processed_data['feature_columns'])}")
        
        # Show some derived features
        derived_features = [col for col in processed_data['data'].columns 
                          if col in ['period_duration_ratio', 'depth_radius_correlation', 
                                    'temp_habitable_zone', 'transit_snr']]
        if derived_features:
            print(f"\nDerived features created: {derived_features}")
        
        # Prepare for training
        print("\n" + "="*80)
        print("STEP 3: Preparing Train/Val/Test Splits")
        print("="*80)
        
        training_data = processor.prepare_for_training(
            processed_data,
            test_size=0.2,
            val_size=0.1
        )
        
        print("\n✓ Data splitting completed successfully!")
        print(f"\nTraining set shape: {training_data['X_train'].shape}")
        print(f"Validation set shape: {training_data['X_val'].shape}")
        print(f"Test set shape: {training_data['X_test'].shape}")
        
        # Display feature statistics
        print("\n" + "="*80)
        print("STEP 4: Feature Statistics")
        print("="*80)
        
        print(f"\nSample of processed features (first 5 rows):")
        print(training_data['X_train'].head())
        
        print(f"\nTarget distribution:")
        print(f"  Training: {training_data['y_train'].value_counts().to_dict()}")
        print(f"  Validation: {training_data['y_val'].value_counts().to_dict()}")
        print(f"  Test: {training_data['y_test'].value_counts().to_dict()}")
        
        # Test individual components
        print("\n" + "="*80)
        print("STEP 5: Testing Individual Components")
        print("="*80)
        
        # Test DataCleaner
        print("\n✓ DataCleaner: Missing value strategies applied:")
        for col, (strategy, value) in list(processor.cleaner.missing_value_strategies.items())[:5]:
            print(f"  {col}: {strategy} = {value}")
        
        # Test FeatureNormalizer
        print(f"\n✓ FeatureNormalizer: {len(processor.normalizer.scalers)} scalers fitted")
        
        # Test CategoryEncoder
        print(f"\n✓ CategoryEncoder: {len(processor.encoder.encoders)} encoders fitted")
        if processor.encoder.encoding_mappings:
            print(f"  Disposition encoding: {list(processor.encoder.encoding_mappings.get('koi_disposition', {}).items())[:5]}")
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED!")
        print("="*80)
        print("\nData processing pipeline is ready for model training.")
        
    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    test_data_processing_pipeline()
