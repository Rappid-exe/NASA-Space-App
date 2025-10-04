"""
Test script for data ingestion functionality.
Demonstrates downloading and loading NASA exoplanet datasets.
"""

from data.dataset_downloader import DatasetDownloader
from data.dataset_loader import DatasetLoader
from data.dataset_validator import DatasetValidator


def test_data_ingestion():
    """Test the complete data ingestion pipeline."""
    
    print("="*60)
    print("Testing NASA Exoplanet Data Ingestion")
    print("="*60)
    
    # Initialize components
    downloader = DatasetDownloader(data_dir="data/raw")
    loader = DatasetLoader()
    validator = DatasetValidator()
    
    # Download datasets
    print("\n[1] Downloading datasets...")
    try:
        datasets = downloader.download_all_datasets()
        print(f"\nSuccessfully downloaded {len(datasets)} datasets:")
        for name, path in datasets.items():
            print(f"  - {name}: {path}")
    except Exception as e:
        print(f"Error downloading datasets: {e}")
        print("Continuing with any successfully downloaded datasets...")
        datasets = {}
    
    # Load and inspect each dataset
    for dataset_name, file_path in datasets.items():
        try:
            print(f"\n[2] Loading {dataset_name.upper()} dataset...")
            df = loader.load_dataset(file_path, dataset_name)
            
            print(f"\n[3] Inspecting {dataset_name.upper()} dataset...")
            loader.inspect_dataset(df, dataset_name)
            
            print(f"\n[4] Validating {dataset_name.upper()} dataset...")
            is_valid, messages = validator.validate_dataset(df, dataset_name)
            for msg in messages:
                print(f"  {msg}")
            
            print(f"\n[5] Getting dataset info for {dataset_name.upper()}...")
            info = loader.get_dataset_info(df)
            print(f"  Total memory usage: {info['memory_usage'] / 1024 / 1024:.2f} MB")
            print(f"  Numeric columns: {len(info['numeric_columns'])}")
            print(f"  Categorical columns: {len(info['categorical_columns'])}")
            
            # Identify classification column
            class_col = loader.get_classification_column(df, dataset_name)
            if class_col:
                print(f"  Classification column: {class_col}")
            
        except Exception as e:
            print(f"Error processing {dataset_name} dataset: {e}")
            continue
    
    print("\n" + "="*60)
    print("Data ingestion test complete!")
    print("="*60)


if __name__ == "__main__":
    test_data_ingestion()
