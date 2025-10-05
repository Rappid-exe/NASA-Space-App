"""
Multi-dataset loader for NASA exoplanet datasets.
Handles loading and harmonizing data from Kepler, TESS, and K2 missions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os


class MultiDatasetLoader:
    """Loads and harmonizes data from multiple NASA exoplanet missions."""
    
    # Column mappings for each mission to unified schema
    COLUMN_MAPPINGS = {
        'kepler': {
            'disposition': 'koi_disposition',
            'period': 'koi_period',
            'duration': 'koi_duration',
            'depth': 'koi_depth',
            'radius': 'koi_prad',
            'temperature': 'koi_teq',
            'impact': 'koi_impact',
            'stellar_temp': 'koi_steff',
            'stellar_radius': 'koi_srad',
            'stellar_mass': 'koi_smass',
            'semi_major_axis': 'koi_sma',
            'eccentricity': 'koi_eccen',
            'insolation_flux': 'koi_insol',
            'transit_epoch': 'koi_time0bk',
            'snr': 'koi_model_snr'
        },
        'tess': {
            'disposition': 'tfopwg_disp',
            'period': 'pl_orbper',
            'duration': 'pl_trandur',
            'depth': 'pl_trandep',
            'radius': 'pl_rade',
            'temperature': 'pl_eqt',
            'impact': 'pl_imppar',
            'stellar_temp': 'st_teff',
            'stellar_radius': 'st_rad',
            'stellar_mass': 'st_mass',
            'semi_major_axis': 'pl_orbsmax',
            'eccentricity': 'pl_orbeccen',
            'insolation_flux': 'pl_insol',
            'transit_epoch': 'pl_tranmid',
            'snr': 'pl_trandep'  # Using depth as proxy for SNR
        },
        'k2': {
            'disposition': 'disposition',
            'period': 'pl_orbper',
            'duration': 'pl_trandur',
            'depth': 'pl_trandep',
            'radius': 'pl_rade',
            'temperature': 'pl_eqt',
            'impact': 'pl_imppar',
            'stellar_temp': 'st_teff',
            'stellar_radius': 'st_rad',
            'stellar_mass': 'st_mass',
            'semi_major_axis': 'pl_orbsmax',
            'eccentricity': 'pl_orbeccen',
            'insolation_flux': 'pl_insol',
            'transit_epoch': 'pl_tranmid',
            'snr': 'pl_trandep'  # Using depth as proxy for SNR
        }
    }
    
    # Disposition value mappings to unified labels
    DISPOSITION_MAPPINGS = {
        'kepler': {
            'CONFIRMED': 'CONFIRMED',
            'CANDIDATE': 'CANDIDATE',
            'FALSE POSITIVE': 'FALSE_POSITIVE',
            'NOT DISPOSITIONED': 'NOT_DISPOSITIONED'
        },
        'tess': {
            'CP': 'CONFIRMED',  # Confirmed Planet
            'PC': 'CANDIDATE',  # Planet Candidate
            'KP': 'CONFIRMED',  # Known Planet
            'FP': 'FALSE_POSITIVE',  # False Positive
            'APC': 'CANDIDATE',  # Ambiguous Planet Candidate
            'FA': 'FALSE_POSITIVE'  # False Alarm
        },
        'k2': {
            'CONFIRMED': 'CONFIRMED',
            'CANDIDATE': 'CANDIDATE',
            'FALSE POSITIVE': 'FALSE_POSITIVE',
            'NOT DISPOSITIONED': 'NOT_DISPOSITIONED'
        }
    }
    
    # Default file paths for each dataset
    DEFAULT_PATHS = {
        'kepler': 'data/raw/cumulative.csv',
        'tess': 'data/raw/toi.csv',
        'k2': 'data/raw/k2targets.csv'
    }
    
    def __init__(self):
        """Initialize the multi-dataset loader."""
        self.loaded_datasets = {}
        self.dataset_statistics = {}
    
    def load_dataset(self, mission: str, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load a single dataset from a specific mission.
        
        Args:
            mission: Mission name ('kepler', 'tess', or 'k2')
            file_path: Optional custom file path (uses default if None)
            
        Returns:
            Loaded dataset as pandas DataFrame
            
        Raises:
            ValueError: If mission is not supported
            FileNotFoundError: If dataset file doesn't exist
        """
        mission = mission.lower()
        
        if mission not in self.COLUMN_MAPPINGS:
            raise ValueError(f"Unsupported mission: {mission}. Must be one of: {list(self.COLUMN_MAPPINGS.keys())}")
        
        if file_path is None:
            file_path = self.DEFAULT_PATHS.get(mission)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        print(f"Loading {mission.upper()} dataset from {file_path}...")
        df = pd.read_csv(file_path, comment='#', low_memory=False)
        
        print(f"  Loaded {len(df)} rows and {len(df.columns)} columns")
        
        # Store original dataset
        self.loaded_datasets[mission] = df
        
        # Calculate statistics
        self._calculate_statistics(mission, df)
        
        return df
    
    def load_all_datasets(self, missions: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Load all available NASA datasets.
        
        Args:
            missions: Optional list of missions to load (loads all if None)
            
        Returns:
            Dictionary mapping mission names to DataFrames
        """
        if missions is None:
            missions = list(self.DEFAULT_PATHS.keys())
        
        datasets = {}
        
        for mission in missions:
            try:
                df = self.load_dataset(mission)
                datasets[mission] = df
            except FileNotFoundError as e:
                print(f"  Warning: {e}")
                print(f"  Skipping {mission.upper()} dataset")
            except Exception as e:
                print(f"  Error loading {mission.upper()}: {e}")
                print(f"  Skipping {mission.upper()} dataset")
        
        return datasets
    
    def harmonize_dataset(self, df: pd.DataFrame, mission: str) -> pd.DataFrame:
        """
        Harmonize a single dataset to unified schema.
        
        Args:
            df: DataFrame to harmonize
            mission: Mission name for column mapping
            
        Returns:
            Harmonized DataFrame with unified column names
        """
        mission = mission.lower()
        
        if mission not in self.COLUMN_MAPPINGS:
            raise ValueError(f"Unsupported mission: {mission}")
        
        print(f"Harmonizing {mission.upper()} dataset...")
        
        harmonized_df = pd.DataFrame()
        column_mapping = self.COLUMN_MAPPINGS[mission]
        
        # Map columns to unified schema
        for unified_col, original_col in column_mapping.items():
            if original_col in df.columns:
                harmonized_df[unified_col] = df[original_col]
            else:
                # Column doesn't exist, fill with NaN
                harmonized_df[unified_col] = np.nan
                print(f"  Warning: Column '{original_col}' not found, using NaN")
        
        # Harmonize disposition values
        if 'disposition' in harmonized_df.columns:
            harmonized_df['disposition'] = self._harmonize_disposition(
                harmonized_df['disposition'], mission
            )
        
        # Add mission identifier
        harmonized_df['mission'] = mission.upper()
        
        # Add row count before filtering
        original_count = len(harmonized_df)
        
        # Remove rows with missing critical values
        critical_columns = ['disposition', 'period', 'radius']
        harmonized_df = harmonized_df.dropna(subset=critical_columns)
        
        removed_count = original_count - len(harmonized_df)
        if removed_count > 0:
            print(f"  Removed {removed_count} rows with missing critical values")
        
        print(f"  Harmonized to {len(harmonized_df)} rows with {len(harmonized_df.columns)} columns")
        
        return harmonized_df
    
    def harmonize_all_datasets(self, datasets: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, pd.DataFrame]:
        """
        Harmonize all loaded datasets.
        
        Args:
            datasets: Optional dictionary of datasets (uses loaded_datasets if None)
            
        Returns:
            Dictionary of harmonized DataFrames
        """
        if datasets is None:
            datasets = self.loaded_datasets
        
        harmonized = {}
        
        for mission, df in datasets.items():
            try:
                harmonized[mission] = self.harmonize_dataset(df, mission)
            except Exception as e:
                print(f"Error harmonizing {mission.upper()}: {e}")
        
        return harmonized
    
    def combine_datasets(self, harmonized_datasets: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Combine harmonized datasets into a single unified dataset.
        
        Args:
            harmonized_datasets: Optional dictionary of harmonized datasets
            
        Returns:
            Combined DataFrame with all missions
        """
        if harmonized_datasets is None:
            # Load and harmonize all datasets
            raw_datasets = self.load_all_datasets()
            harmonized_datasets = self.harmonize_all_datasets(raw_datasets)
        
        if not harmonized_datasets:
            raise ValueError("No datasets available to combine")
        
        print("\nCombining datasets...")
        
        # Concatenate all datasets
        combined_df = pd.concat(harmonized_datasets.values(), ignore_index=True)
        
        print(f"  Combined dataset: {len(combined_df)} rows")
        print(f"  Mission distribution:")
        for mission, count in combined_df['mission'].value_counts().items():
            print(f"    {mission}: {count} rows")
        
        print(f"  Disposition distribution:")
        for disp, count in combined_df['disposition'].value_counts().items():
            print(f"    {disp}: {count} rows")
        
        return combined_df
    
    def _harmonize_disposition(self, disposition_series: pd.Series, mission: str) -> pd.Series:
        """
        Harmonize disposition values to unified labels.
        
        Args:
            disposition_series: Series with disposition values
            mission: Mission name for mapping
            
        Returns:
            Series with harmonized disposition values
        """
        mapping = self.DISPOSITION_MAPPINGS.get(mission, {})
        
        def map_disposition(value):
            if pd.isna(value):
                return 'NOT_DISPOSITIONED'
            
            value_str = str(value).strip().upper()
            
            # Try exact match first
            if value_str in mapping:
                return mapping[value_str]
            
            # Try partial matches
            for key, unified_value in mapping.items():
                if key in value_str or value_str in key:
                    return unified_value
            
            # Default to NOT_DISPOSITIONED for unknown values
            return 'NOT_DISPOSITIONED'
        
        return disposition_series.apply(map_disposition)
    
    def _calculate_statistics(self, mission: str, df: pd.DataFrame) -> None:
        """
        Calculate and store statistics for a dataset.
        
        Args:
            mission: Mission name
            df: DataFrame to analyze
        """
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        # Get disposition column if it exists
        column_mapping = self.COLUMN_MAPPINGS.get(mission, {})
        disp_col = column_mapping.get('disposition')
        
        if disp_col and disp_col in df.columns:
            stats['disposition_counts'] = df[disp_col].value_counts().to_dict()
        
        self.dataset_statistics[mission] = stats
    
    def get_dataset_statistics(self) -> Dict[str, Dict]:
        """
        Get statistics for all loaded datasets.
        
        Returns:
            Dictionary of statistics for each mission
        """
        return self.dataset_statistics
    
    def get_feature_coverage(self, harmonized_datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Analyze feature coverage across missions.
        
        Args:
            harmonized_datasets: Dictionary of harmonized DataFrames
            
        Returns:
            DataFrame showing feature availability by mission
        """
        coverage = {}
        
        for mission, df in harmonized_datasets.items():
            coverage[mission] = {}
            for col in df.columns:
                if col != 'mission':
                    # Calculate percentage of non-null values
                    non_null_pct = (df[col].notna().sum() / len(df)) * 100
                    coverage[mission][col] = non_null_pct
        
        coverage_df = pd.DataFrame(coverage).T
        coverage_df = coverage_df.round(2)
        
        return coverage_df
    
    def filter_by_disposition(self, df: pd.DataFrame, 
                             include_dispositions: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Filter dataset by disposition values.
        
        Args:
            df: DataFrame to filter
            include_dispositions: List of dispositions to keep (None = keep all)
            
        Returns:
            Filtered DataFrame
        """
        if include_dispositions is None:
            return df
        
        filtered_df = df[df['disposition'].isin(include_dispositions)]
        
        print(f"Filtered from {len(df)} to {len(filtered_df)} rows")
        print(f"Disposition distribution:")
        for disp, count in filtered_df['disposition'].value_counts().items():
            print(f"  {disp}: {count} rows")
        
        return filtered_df
    
    def create_binary_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary labels for classification (CONFIRMED vs others).
        
        Args:
            df: DataFrame with disposition column
            
        Returns:
            DataFrame with added 'label' column (1=CONFIRMED, 0=other)
        """
        df_labeled = df.copy()
        
        df_labeled['label'] = (df_labeled['disposition'] == 'CONFIRMED').astype(int)
        
        print(f"Binary label distribution:")
        print(f"  Confirmed (1): {(df_labeled['label'] == 1).sum()} rows")
        print(f"  Not Confirmed (0): {(df_labeled['label'] == 0).sum()} rows")
        
        return df_labeled
