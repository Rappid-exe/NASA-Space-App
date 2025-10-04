"""
Data processor for NASA exoplanet datasets.
Handles data cleaning, normalization, encoding, and feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class DataCleaner:
    """Handles missing values and outliers in astronomical data."""
    
    def __init__(self):
        """Initialize the data cleaner."""
        self.missing_value_strategies = {}
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: DataFrame with potential missing values
            strategy: Strategy for handling missing values ('auto', 'drop', 'median', 'mean')
            
        Returns:
            DataFrame with missing values handled
        """
        df_clean = df.copy()
        
        if strategy == 'auto':
            # For numerical columns, use median imputation
            numeric_cols = df_clean.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if df_clean[col].isnull().any():
                    median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val)
                    self.missing_value_strategies[col] = ('median', median_val)
            
            # For categorical columns, use mode or 'UNKNOWN'
            categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if df_clean[col].isnull().any():
                    mode_val = df_clean[col].mode()
                    if len(mode_val) > 0:
                        df_clean[col] = df_clean[col].fillna(mode_val[0])
                        self.missing_value_strategies[col] = ('mode', mode_val[0])
                    else:
                        df_clean[col] = df_clean[col].fillna('UNKNOWN')
                        self.missing_value_strategies[col] = ('constant', 'UNKNOWN')
        
        elif strategy == 'drop':
            df_clean = df_clean.dropna()
        
        elif strategy == 'median':
            numeric_cols = df_clean.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if df_clean[col].isnull().any():
                    median_val = df_clean[col].median()
                    df_clean[col].fillna(median_val, inplace=True)
        
        elif strategy == 'mean':
            numeric_cols = df_clean.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if df_clean[col].isnull().any():
                    mean_val = df_clean[col].mean()
                    df_clean[col].fillna(mean_val, inplace=True)
        
        return df_clean
    
    def remove_outliers(self, df: pd.DataFrame, columns: List[str], 
                       method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers from specified columns.
        
        Args:
            df: DataFrame to process
            columns: List of column names to check for outliers
            method: Method for outlier detection ('iqr' or 'zscore')
            threshold: Threshold for outlier detection (IQR multiplier or z-score)
            
        Returns:
            DataFrame with outliers removed
        """
        df_clean = df.copy()
        
        for col in columns:
            if col not in df_clean.columns or df_clean[col].dtype not in ['int64', 'float64']:
                continue
            
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            
            elif method == 'zscore':
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                df_clean = df_clean[z_scores < threshold]
        
        return df_clean


class FeatureNormalizer:
    """Normalizes numerical features to consistent scales."""
    
    def __init__(self):
        """Initialize the feature normalizer."""
        self.scalers = {}
    
    def normalize_features(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                          method: str = 'standard') -> pd.DataFrame:
        """
        Normalize numerical features.
        
        Args:
            df: DataFrame to normalize
            columns: List of columns to normalize (None = all numeric columns)
            method: Normalization method ('standard', 'minmax')
            
        Returns:
            DataFrame with normalized features
        """
        df_normalized = df.copy()
        
        if columns is None:
            columns = df_normalized.select_dtypes(include=['number']).columns.tolist()
        
        if method == 'standard':
            for col in columns:
                if col in df_normalized.columns:
                    scaler = StandardScaler()
                    df_normalized[col] = scaler.fit_transform(df_normalized[[col]])
                    self.scalers[col] = scaler
        
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            for col in columns:
                if col in df_normalized.columns:
                    scaler = MinMaxScaler()
                    df_normalized[col] = scaler.fit_transform(df_normalized[[col]])
                    self.scalers[col] = scaler
        
        return df_normalized
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted scalers.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        df_transformed = df.copy()
        
        for col, scaler in self.scalers.items():
            if col in df_transformed.columns:
                df_transformed[col] = scaler.transform(df_transformed[[col]])
        
        return df_transformed


class CategoryEncoder:
    """Encodes categorical variables for ML compatibility."""
    
    def __init__(self):
        """Initialize the category encoder."""
        self.encoders = {}
        self.encoding_mappings = {}
    
    def encode_categories(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                         method: str = 'label') -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            df: DataFrame to encode
            columns: List of columns to encode (None = all object columns)
            method: Encoding method ('label', 'onehot')
            
        Returns:
            DataFrame with encoded categories
        """
        df_encoded = df.copy()
        
        if columns is None:
            columns = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if method == 'label':
            for col in columns:
                if col in df_encoded.columns:
                    encoder = LabelEncoder()
                    # Convert categorical to string first, then handle NaN values
                    if df_encoded[col].dtype.name == 'category':
                        df_encoded[col] = df_encoded[col].astype(str)
                    # Handle any remaining NaN values
                    df_encoded[col] = df_encoded[col].fillna('UNKNOWN')
                    df_encoded[col] = encoder.fit_transform(df_encoded[col])
                    self.encoders[col] = encoder
                    self.encoding_mappings[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
        
        elif method == 'onehot':
            df_encoded = pd.get_dummies(df_encoded, columns=columns, prefix=columns)
        
        return df_encoded
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted encoders.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        df_transformed = df.copy()
        
        for col, encoder in self.encoders.items():
            if col in df_transformed.columns:
                df_transformed[col] = df_transformed[col].fillna('UNKNOWN')
                # Handle unseen categories
                df_transformed[col] = df_transformed[col].apply(
                    lambda x: x if x in encoder.classes_ else 'UNKNOWN'
                )
                df_transformed[col] = encoder.transform(df_transformed[col])
        
        return df_transformed
    
    def encode_disposition(self, df: pd.DataFrame, disposition_col: str,
                          target_mapping: Optional[Dict[str, int]] = None) -> pd.DataFrame:
        """
        Encode disposition/classification column with custom mapping.
        
        Args:
            df: DataFrame to encode
            disposition_col: Name of the disposition column
            target_mapping: Custom mapping for disposition values
            
        Returns:
            DataFrame with encoded disposition
        """
        df_encoded = df.copy()
        
        if target_mapping is None:
            # Default mapping for common disposition values
            target_mapping = {
                'CONFIRMED': 1,
                'CANDIDATE': 1,
                'PC': 1,  # Planetary Candidate
                'CP': 1,  # Confirmed Planet
                'KP': 1,  # Known Planet
                'FALSE POSITIVE': 0,
                'FP': 0,
                'NOT DISPOSITIONED': 0,
                'APC': 0,  # Ambiguous Planetary Candidate
            }
        
        # Apply mapping with case-insensitive matching
        df_encoded[disposition_col] = df_encoded[disposition_col].apply(
            lambda x: target_mapping.get(str(x).upper(), 0) if pd.notna(x) else 0
        )
        
        self.encoding_mappings[disposition_col] = target_mapping
        
        return df_encoded


class FeatureEngineer:
    """Creates derived features from astronomical observations."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        pass
    
    def create_derived_features(self, df: pd.DataFrame, dataset_type: str = 'koi') -> pd.DataFrame:
        """
        Create derived astronomical features.
        
        Args:
            df: DataFrame with raw features
            dataset_type: Type of dataset (koi, toi, k2) for column mapping
            
        Returns:
            DataFrame with additional derived features
        """
        df_engineered = df.copy()
        
        # Map column names based on dataset type
        col_mapping = self._get_column_mapping(dataset_type)
        
        # Period-Duration Ratio
        if col_mapping['period'] in df.columns and col_mapping['duration'] in df.columns:
            df_engineered['period_duration_ratio'] = (
                df_engineered[col_mapping['period']] / 
                (df_engineered[col_mapping['duration']] / 24.0)  # Convert hours to days
            )
        
        # Transit Depth to Radius Relationship
        if col_mapping['depth'] in df.columns and col_mapping['radius'] in df.columns:
            df_engineered['depth_radius_correlation'] = (
                df_engineered[col_mapping['depth']] / 
                (df_engineered[col_mapping['radius']] ** 2)
            )
        
        # Equilibrium Temperature features (if available)
        if col_mapping['temperature'] in df.columns:
            df_engineered['temp_habitable_zone'] = df_engineered[col_mapping['temperature']].apply(
                lambda x: 1 if 200 <= x <= 350 else 0 if pd.notna(x) else 0
            )
        
        # Orbital Period categories
        if col_mapping['period'] in df.columns:
            df_engineered['period_category'] = pd.cut(
                df_engineered[col_mapping['period']],
                bins=[0, 10, 100, 1000, float('inf')],
                labels=['ultra_short', 'short', 'medium', 'long']
            )
        
        # Planetary Radius categories (Earth-like, Super-Earth, Neptune-like, Jupiter-like)
        if col_mapping['radius'] in df.columns:
            df_engineered['radius_category'] = pd.cut(
                df_engineered[col_mapping['radius']],
                bins=[0, 1.5, 2.5, 6, float('inf')],
                labels=['earth_like', 'super_earth', 'neptune_like', 'jupiter_like']
            )
        
        # Transit Signal-to-Noise (if depth and stellar magnitude available)
        if col_mapping['depth'] in df.columns:
            df_engineered['transit_snr'] = np.log10(df_engineered[col_mapping['depth']] + 1)
        
        return df_engineered
    
    def _get_column_mapping(self, dataset_type: str) -> Dict[str, str]:
        """
        Get column name mapping for different dataset types.
        
        Args:
            dataset_type: Type of dataset (koi, toi, k2)
            
        Returns:
            Dictionary mapping feature types to column names
        """
        mappings = {
            'koi': {
                'period': 'koi_period',
                'duration': 'koi_duration',
                'depth': 'koi_depth',
                'radius': 'koi_prad',
                'temperature': 'koi_teq',
                'disposition': 'koi_disposition'
            },
            'toi': {
                'period': 'pl_orbper',
                'duration': 'pl_trandur',
                'depth': 'pl_trandep',
                'radius': 'pl_rade',
                'temperature': 'pl_eqt',
                'disposition': 'tfopwg_disp'
            },
            'k2': {
                'period': 'k2c_period',
                'duration': 'k2c_duration',
                'depth': 'k2c_depth',
                'radius': 'k2c_prad',
                'temperature': 'k2c_teq',
                'disposition': 'k2c_disp'
            }
        }
        
        return mappings.get(dataset_type.lower(), mappings['koi'])


class DataSplitter:
    """Creates train/validation/test splits with stratification."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the data splitter.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
    
    def split_data(self, df: pd.DataFrame, target_col: str,
                   test_size: float = 0.2, val_size: float = 0.1,
                   stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: DataFrame to split
            target_col: Name of the target column
            test_size: Proportion of data for test set
            val_size: Proportion of training data for validation set
            stratify: Whether to stratify splits by target
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        stratify_col = df[target_col] if stratify else None
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_col
        )
        
        # Second split: separate validation from training
        stratify_col_train = train_val_df[target_col] if stratify else None
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size / (1 - test_size),  # Adjust val_size relative to remaining data
            random_state=self.random_state,
            stratify=stratify_col_train
        )
        
        return train_df, val_df, test_df


class DataProcessor:
    """Main data processing pipeline orchestrator."""
    
    def __init__(self):
        """Initialize the data processor with all components."""
        self.cleaner = DataCleaner()
        self.normalizer = FeatureNormalizer()
        self.encoder = CategoryEncoder()
        self.engineer = FeatureEngineer()
        self.splitter = DataSplitter()
    
    def process_dataset(self, df: pd.DataFrame, dataset_type: str = 'koi',
                       target_col: Optional[str] = None,
                       remove_outliers: bool = True,
                       create_features: bool = True) -> Dict:
        """
        Complete data processing pipeline.
        
        Args:
            df: Raw DataFrame to process
            dataset_type: Type of dataset (koi, toi, k2)
            target_col: Name of target column (auto-detected if None)
            remove_outliers: Whether to remove outliers
            create_features: Whether to create derived features
            
        Returns:
            Dictionary containing processed data and metadata
        """
        print(f"Starting data processing pipeline for {dataset_type} dataset...")
        print(f"Initial shape: {df.shape}")
        
        # Step 1: Handle missing values
        print("\n1. Handling missing values...")
        df_processed = self.cleaner.handle_missing_values(df, strategy='auto')
        print(f"Shape after handling missing values: {df_processed.shape}")
        
        # Step 2: Identify target column
        if target_col is None:
            col_mapping = self.engineer._get_column_mapping(dataset_type)
            target_col = col_mapping['disposition']
        
        if target_col not in df_processed.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        # Step 3: Encode target variable
        print(f"\n2. Encoding target variable: {target_col}...")
        df_processed = self.encoder.encode_disposition(df_processed, target_col)
        
        # Step 4: Create derived features
        if create_features:
            print("\n3. Creating derived features...")
            df_processed = self.engineer.create_derived_features(df_processed, dataset_type)
            print(f"Shape after feature engineering: {df_processed.shape}")
        
        # Step 5: Remove outliers from numerical columns
        if remove_outliers:
            print("\n4. Removing outliers...")
            numeric_cols = df_processed.select_dtypes(include=['number']).columns.tolist()
            # Don't remove outliers from target column
            numeric_cols = [col for col in numeric_cols if col != target_col]
            df_processed = self.cleaner.remove_outliers(df_processed, numeric_cols, method='iqr', threshold=3.0)
            print(f"Shape after outlier removal: {df_processed.shape}")
        
        # Step 6: Encode categorical features
        print("\n5. Encoding categorical features...")
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            df_processed = self.encoder.encode_categories(df_processed, categorical_cols, method='label')
        
        # Step 7: Normalize numerical features
        print("\n6. Normalizing numerical features...")
        numeric_cols = df_processed.select_dtypes(include=['number']).columns.tolist()
        # Don't normalize target column
        numeric_cols = [col for col in numeric_cols if col != target_col]
        df_processed = self.normalizer.normalize_features(df_processed, numeric_cols, method='standard')
        
        print(f"\nFinal processed shape: {df_processed.shape}")
        
        return {
            'data': df_processed,
            'target_column': target_col,
            'feature_columns': [col for col in df_processed.columns if col != target_col],
            'cleaner': self.cleaner,
            'normalizer': self.normalizer,
            'encoder': self.encoder,
            'engineer': self.engineer
        }
    
    def prepare_for_training(self, processed_data: Dict,
                           test_size: float = 0.2,
                           val_size: float = 0.1) -> Dict:
        """
        Prepare processed data for model training.
        
        Args:
            processed_data: Output from process_dataset
            test_size: Proportion for test set
            val_size: Proportion for validation set
            
        Returns:
            Dictionary with train/val/test splits
        """
        df = processed_data['data']
        target_col = processed_data['target_column']
        
        print("\nSplitting data into train/validation/test sets...")
        train_df, val_df, test_df = self.splitter.split_data(
            df, target_col, test_size=test_size, val_size=val_size, stratify=True
        )
        
        # Separate features and target
        feature_cols = processed_data['feature_columns']
        
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        
        X_val = val_df[feature_cols]
        y_val = val_df[target_col]
        
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]
        
        print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
        print(f"Target distribution - Train: {y_train.value_counts().to_dict()}")
        print(f"Target distribution - Val: {y_val.value_counts().to_dict()}")
        print(f"Target distribution - Test: {y_test.value_counts().to_dict()}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'feature_columns': feature_cols,
            'target_column': target_col
        }
