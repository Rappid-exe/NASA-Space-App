"""
Dataset loader for NASA exoplanet datasets.
Handles loading CSV files and providing dataset information.
"""

import pandas as pd
from typing import Dict, Optional


class DatasetLoader:
    """Loads and provides information about NASA exoplanet datasets."""
    
    def __init__(self):
        """Initialize the dataset loader."""
        self.loaded_datasets = {}
    
    def load_dataset(self, file_path: str, dataset_name: Optional[str] = None) -> pd.DataFrame:
        """
        Load a dataset from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            dataset_name: Optional name to cache the dataset
            
        Returns:
            Loaded dataset as a pandas DataFrame
        """
        print(f"Loading dataset from {file_path}...")
        df = pd.read_csv(file_path, comment='#', low_memory=False)
        
        if dataset_name:
            self.loaded_datasets[dataset_name] = df
        
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def get_dataset_info(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive information about a dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary containing dataset information
        """
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_columns': list(df.select_dtypes(include=['number']).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns),
        }
        
        return info
    
    def inspect_dataset(self, df: pd.DataFrame, dataset_type: str = 'unknown') -> None:
        """
        Print detailed inspection of a dataset.
        
        Args:
            df: DataFrame to inspect
            dataset_type: Type of dataset (koi, toi, k2)
        """
        print(f"\n{'='*60}")
        print(f"Dataset Inspection: {dataset_type.upper()}")
        print(f"{'='*60}")
        
        print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
        
        print(f"\nFirst few rows:")
        print(df.head())
        
        print(f"\nColumn data types:")
        print(df.dtypes)
        
        print(f"\nMissing values:")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Percentage': missing_pct
        })
        print(missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False))
        
        print(f"\nBasic statistics:")
        print(df.describe())
        
        # Look for disposition/classification columns
        disposition_cols = [col for col in df.columns if 'disposition' in col.lower() or 'status' in col.lower()]
        if disposition_cols:
            print(f"\nClassification columns found: {disposition_cols}")
            for col in disposition_cols:
                print(f"\n{col} value counts:")
                print(df[col].value_counts())
    
    def get_classification_column(self, df: pd.DataFrame, dataset_type: str) -> Optional[str]:
        """
        Identify the classification/disposition column for a dataset.
        
        Args:
            df: DataFrame to analyze
            dataset_type: Type of dataset (koi, toi, k2)
            
        Returns:
            Name of the classification column or None
        """
        # Common classification column names by dataset type
        classification_mappings = {
            'koi': ['koi_disposition', 'koi_pdisposition'],
            'toi': ['tfopwg_disp', 'toi_disposition'],
            'k2': ['k2c_disp', 'epic_disposition']
        }
        
        possible_cols = classification_mappings.get(dataset_type.lower(), [])
        
        for col in possible_cols:
            if col in df.columns:
                return col
        
        # Fallback: search for any column with 'disposition' in name
        for col in df.columns:
            if 'disposition' in col.lower():
                return col
        
        return None
