"""
Dataset validator for NASA exoplanet datasets.
Validates data integrity and format.
"""

import pandas as pd
from typing import List, Dict, Tuple


class DatasetValidator:
    """Validates NASA exoplanet dataset integrity and format."""
    
    def __init__(self):
        """Initialize the dataset validator."""
        pass
    
    def validate_dataset(self, df: pd.DataFrame, dataset_type: str) -> Tuple[bool, List[str]]:
        """
        Validate a dataset for integrity and format issues.
        
        Args:
            df: DataFrame to validate
            dataset_type: Type of dataset (koi, toi, k2)
            
        Returns:
            Tuple of (is_valid, list of validation messages)
        """
        messages = []
        is_valid = True
        
        # Check if dataset is empty
        if df.empty:
            messages.append("ERROR: Dataset is empty")
            return False, messages
        
        # Check for minimum number of rows
        if len(df) < 10:
            messages.append(f"WARNING: Dataset has only {len(df)} rows, which may be insufficient for training")
            is_valid = False
        
        # Check for excessive missing values
        missing_pct = (df.isnull().sum() / len(df)) * 100
        high_missing = missing_pct[missing_pct > 50]
        if not high_missing.empty:
            messages.append(f"WARNING: {len(high_missing)} columns have >50% missing values")
        
        # Check for classification column
        classification_col = self._find_classification_column(df, dataset_type)
        if classification_col is None:
            messages.append(f"ERROR: No classification/disposition column found for {dataset_type} dataset")
            is_valid = False
        else:
            messages.append(f"INFO: Classification column identified: {classification_col}")
            
            # Check classification distribution
            class_counts = df[classification_col].value_counts()
            messages.append(f"INFO: Classification distribution: {class_counts.to_dict()}")
            
            # Check for class imbalance
            if len(class_counts) > 1:
                min_class = class_counts.min()
                max_class = class_counts.max()
                if max_class / min_class > 10:
                    messages.append("WARNING: Significant class imbalance detected")
        
        # Check for required numerical features
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) < 5:
            messages.append(f"WARNING: Only {len(numeric_cols)} numerical columns found, may need more features")
        
        if is_valid:
            messages.append("SUCCESS: Dataset validation passed")
        
        return is_valid, messages
    
    def _find_classification_column(self, df: pd.DataFrame, dataset_type: str) -> str:
        """Find the classification column in the dataset."""
        classification_mappings = {
            'koi': ['koi_disposition', 'koi_pdisposition'],
            'toi': ['tfopwg_disp', 'toi_disposition'],
            'k2': ['k2c_disp', 'epic_disposition']
        }
        
        possible_cols = classification_mappings.get(dataset_type.lower(), [])
        
        for col in possible_cols:
            if col in df.columns:
                return col
        
        for col in df.columns:
            if 'disposition' in col.lower():
                return col
        
        return None
