"""
Example showing how to use MultiDatasetLoader with existing DataProcessor
for model training with combined datasets.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.multi_dataset_loader import MultiDatasetLoader
from data.data_processor import DataProcessor
import pandas as pd


def prepare_multi_dataset_for_training():
    """
    Complete example of preparing multi-mission data for model training.
    """
    
    print("="*80)
    print("Multi-Dataset Training Preparation Example")
    print("="*80)
    
    # Step 1: Load and combine datasets
    print("\n[1/4] Loading and combining datasets...")
    loader = MultiDatasetLoader()
    combined_df = loader.combine_datasets()
    
    print(f"  ✓ Combined dataset: {len(combined_df):,} observations")
    print(f"  ✓ Features: {len(combined_df.columns)} columns")
    
    # Step 2: Create binary labels
    print("\n[2/4] Creating binary labels...")
    labeled_df = loader.create_binary_labels(combined_df)
    
    confirmed = (labeled_df['label'] == 1).sum()
    not_confirmed = (labeled_df['label'] == 0).sum()
    print(f"  ✓ Confirmed: {confirmed:,} ({confirmed/len(labeled_df)*100:.1f}%)")
    print(f"  ✓ Not Confirmed: {not_confirmed:,} ({not_confirmed/len(labeled_df)*100:.1f}%)")
    
    # Step 3: Process data for training
    print("\n[3/4] Processing data with DataProcessor...")
    processor = DataProcessor()
    
    # Note: We'll use 'label' as target instead of 'disposition'
    # First, let's prepare the dataframe for processing
    df_for_processing = labeled_df.copy()
    
    # The DataProcessor expects certain column names, so we'll work with what we have
    # For now, we'll just demonstrate the concept
    
    print("  ✓ Data ready for processing")
    print(f"  ✓ Shape: {df_for_processing.shape}")
    
    # Step 4: Show data split by mission
    print("\n[4/4] Data distribution by mission...")
    for mission in labeled_df['mission'].unique():
        mission_data = labeled_df[labeled_df['mission'] == mission]
        confirmed_pct = (mission_data['label'] == 1).sum() / len(mission_data) * 100
        
        print(f"\n  {mission}:")
        print(f"    Observations: {len(mission_data):,}")
        print(f"    Confirmed: {(mission_data['label'] == 1).sum():,} ({confirmed_pct:.1f}%)")
        print(f"    Features with data:")
        
        # Show which features have good coverage
        non_null_counts = mission_data.notna().sum()
        good_features = non_null_counts[non_null_counts > len(mission_data) * 0.8]
        print(f"      {len(good_features)} features with >80% coverage")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n✓ Combined {len(labeled_df['mission'].unique())} NASA missions")
    print(f"✓ Total training samples: {len(labeled_df):,}")
    print(f"✓ Unified feature schema: {len(labeled_df.columns)} features")
    print(f"✓ Ready for ensemble model training")
    
    print("\nNext steps:")
    print("  1. Run feature selection to identify top features")
    print("  2. Train RandomForest, Neural Network, and SVM models")
    print("  3. Create ensemble classifier")
    print("  4. Validate with cross-mission testing")
    
    return labeled_df


def demonstrate_cross_mission_split():
    """
    Demonstrate how to split data for cross-mission validation.
    """
    
    print("\n" + "="*80)
    print("Cross-Mission Validation Split Example")
    print("="*80)
    
    # Load combined data
    loader = MultiDatasetLoader()
    combined_df = loader.combine_datasets()
    labeled_df = loader.create_binary_labels(combined_df)
    
    # Example: Train on Kepler, test on TESS
    print("\nExample: Train on KEPLER, test on TESS")
    
    train_data = labeled_df[labeled_df['mission'] == 'KEPLER']
    test_data = labeled_df[labeled_df['mission'] == 'TESS']
    
    print(f"  Training set (KEPLER): {len(train_data):,} samples")
    print(f"  Test set (TESS): {len(test_data):,} samples")
    
    # Show feature overlap
    train_features = train_data.columns[train_data.notna().any()]
    test_features = test_data.columns[test_data.notna().any()]
    common_features = set(train_features) & set(test_features)
    
    print(f"  Common features: {len(common_features)}")
    
    print("\nThis split allows testing model generalization across missions!")
    
    return train_data, test_data


if __name__ == '__main__':
    # Run the main example
    combined_data = prepare_multi_dataset_for_training()
    
    # Show cross-mission validation example
    train_data, test_data = demonstrate_cross_mission_split()
    
    print("\n" + "="*80)
    print("Example complete!")
    print("="*80)
