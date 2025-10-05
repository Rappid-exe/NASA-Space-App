"""
Demonstration of MultiDatasetLoader functionality.
Shows how to load, harmonize, and combine multiple NASA datasets.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.multi_dataset_loader import MultiDatasetLoader
import pandas as pd


def main():
    """Demonstrate MultiDatasetLoader capabilities."""
    
    print("="*80)
    print("Multi-Dataset Loader Demonstration")
    print("="*80)
    
    # Initialize loader
    loader = MultiDatasetLoader()
    
    # Step 1: Load all available datasets
    print("\n" + "="*80)
    print("STEP 1: Loading all available datasets")
    print("="*80)
    
    raw_datasets = loader.load_all_datasets()
    print(f"\nSuccessfully loaded {len(raw_datasets)} datasets")
    
    # Step 2: Show dataset statistics
    print("\n" + "="*80)
    print("STEP 2: Dataset Statistics")
    print("="*80)
    
    stats = loader.get_dataset_statistics()
    for mission, mission_stats in stats.items():
        print(f"\n{mission.upper()} Dataset:")
        print(f"  Total rows: {mission_stats['total_rows']:,}")
        print(f"  Total columns: {mission_stats['total_columns']}")
        print(f"  Missing values: {mission_stats['missing_values']:,}")
        print(f"  Memory usage: {mission_stats['memory_usage_mb']:.2f} MB")
        
        if 'disposition_counts' in mission_stats:
            print(f"  Disposition distribution:")
            for disp, count in mission_stats['disposition_counts'].items():
                print(f"    {disp}: {count:,}")
    
    # Step 3: Harmonize all datasets
    print("\n" + "="*80)
    print("STEP 3: Harmonizing datasets to unified schema")
    print("="*80)
    
    harmonized_datasets = loader.harmonize_all_datasets(raw_datasets)
    
    print(f"\nHarmonized {len(harmonized_datasets)} datasets")
    print("\nUnified columns:")
    if harmonized_datasets:
        first_dataset = list(harmonized_datasets.values())[0]
        for i, col in enumerate(first_dataset.columns, 1):
            print(f"  {i:2d}. {col}")
    
    # Step 4: Analyze feature coverage
    print("\n" + "="*80)
    print("STEP 4: Feature Coverage Analysis")
    print("="*80)
    
    coverage = loader.get_feature_coverage(harmonized_datasets)
    print("\nFeature availability by mission (% non-null):")
    print(coverage.to_string())
    
    # Step 5: Combine all datasets
    print("\n" + "="*80)
    print("STEP 5: Combining all datasets")
    print("="*80)
    
    combined_df = loader.combine_datasets(harmonized_datasets)
    
    print(f"\nCombined dataset shape: {combined_df.shape}")
    print(f"Total observations: {len(combined_df):,}")
    
    # Step 6: Create binary labels for classification
    print("\n" + "="*80)
    print("STEP 6: Creating binary labels for classification")
    print("="*80)
    
    labeled_df = loader.create_binary_labels(combined_df)
    
    print(f"\nLabel distribution:")
    print(f"  Confirmed planets (1): {(labeled_df['label'] == 1).sum():,}")
    print(f"  Not confirmed (0): {(labeled_df['label'] == 0).sum():,}")
    print(f"  Confirmation rate: {(labeled_df['label'] == 1).sum() / len(labeled_df) * 100:.2f}%")
    
    # Step 7: Show sample of combined data
    print("\n" + "="*80)
    print("STEP 7: Sample of combined dataset")
    print("="*80)
    
    print("\nFirst 5 rows:")
    display_cols = ['mission', 'disposition', 'period', 'radius', 'temperature', 'label']
    print(labeled_df[display_cols].head().to_string(index=False))
    
    # Step 8: Mission-specific statistics
    print("\n" + "="*80)
    print("STEP 8: Mission-specific statistics in combined dataset")
    print("="*80)
    
    for mission in labeled_df['mission'].unique():
        mission_data = labeled_df[labeled_df['mission'] == mission]
        confirmed = (mission_data['label'] == 1).sum()
        total = len(mission_data)
        
        print(f"\n{mission}:")
        print(f"  Total observations: {total:,}")
        print(f"  Confirmed planets: {confirmed:,}")
        print(f"  Confirmation rate: {confirmed/total*100:.2f}%")
        print(f"  Average period: {mission_data['period'].mean():.2f} days")
        print(f"  Average radius: {mission_data['radius'].mean():.2f} Earth radii")
    
    # Step 9: Filter for confirmed planets only
    print("\n" + "="*80)
    print("STEP 9: Filtering for confirmed planets only")
    print("="*80)
    
    confirmed_only = loader.filter_by_disposition(combined_df, ['CONFIRMED'])
    
    print(f"\nConfirmed planets by mission:")
    for mission, count in confirmed_only['mission'].value_counts().items():
        print(f"  {mission}: {count:,}")
    
    # Step 10: Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n✓ Successfully loaded {len(raw_datasets)} NASA datasets")
    print(f"✓ Harmonized to unified schema with {len(first_dataset.columns)} features")
    print(f"✓ Combined into single dataset with {len(combined_df):,} observations")
    print(f"✓ Ready for multi-mission model training")
    
    print("\n" + "="*80)
    print("Demonstration complete!")
    print("="*80)
    
    return labeled_df


if __name__ == '__main__':
    combined_data = main()
