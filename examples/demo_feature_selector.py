"""
Demo script for FeatureSelector class.
Shows how to use feature selection for exoplanet classification.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from models.feature_selector import FeatureSelector
from data.dataset_loader import DatasetLoader
from data.data_processor import DataProcessor


def demo_feature_selector():
    """Demonstrate feature selection on real exoplanet data."""
    
    print("=" * 80)
    print("Feature Selector Demo")
    print("=" * 80)
    
    # Load real dataset
    print("\n1. Loading Kepler dataset...")
    loader = DatasetLoader()
    df = loader.load_kepler_data()
    
    if df is None or len(df) == 0:
        print("Error: Could not load dataset. Using synthetic data instead.")
        return demo_with_synthetic_data()
    
    print(f"   Loaded {len(df)} observations with {len(df.columns)} columns")
    
    # Process dataset
    print("\n2. Processing dataset...")
    processor = DataProcessor()
    processed = processor.process_dataset(df, dataset_type='koi', create_features=True)
    
    # Prepare data for feature selection
    print("\n3. Preparing data for feature selection...")
    training_data = processor.prepare_for_training(processed, test_size=0.2, val_size=0.1)
    
    X_train = training_data['X_train']
    y_train = training_data['y_train']
    feature_names = training_data['feature_columns']
    
    print(f"   Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"   Class distribution: {dict(pd.Series(y_train).value_counts())}")
    
    # Initialize feature selector
    print("\n4. Analyzing feature importance...")
    selector = FeatureSelector(random_state=42)
    
    # Analyze importance using multiple methods
    importance_df = selector.analyze_importance(
        X_train.values if isinstance(X_train, pd.DataFrame) else X_train,
        y_train.values if isinstance(y_train, pd.Series) else y_train,
        feature_names
    )
    
    # Select top features
    print("\n5. Selecting top features...")
    selected_features = selector.select_features(top_k=30)
    
    print(f"\n   Reduced features from {len(feature_names)} to {len(selected_features)}")
    print(f"   Feature reduction: {(1 - len(selected_features)/len(feature_names))*100:.1f}%")
    
    # Generate report
    print("\n6. Generating feature importance report...")
    report = selector.get_feature_report()
    
    print(f"\n   Report Summary:")
    print(f"   - Total features: {report['total_features']}")
    print(f"   - Selected features: {report['selected_features']}")
    print(f"   - Selection ratio: {report['selection_ratio']:.2%}")
    
    print(f"\n   Top 10 features by importance:")
    for i, feature in enumerate(report['top_10_features'], 1):
        importance = importance_df[importance_df['feature'] == feature]['avg_importance'].values[0]
        print(f"   {i:2d}. {feature:40s} (importance: {importance:.4f})")
    
    # Save report
    print("\n7. Saving feature importance report...")
    selector.save_report('models/registry/feature_importance_report.json')
    
    # Transform data to use only selected features
    print("\n8. Transforming data to use selected features...")
    X_train_selected = selector.transform(X_train, feature_names)
    
    print(f"   Original shape: {X_train.shape}")
    print(f"   Transformed shape: {X_train_selected.shape}")
    
    print("\n" + "=" * 80)
    print("Feature Selection Demo Complete!")
    print("=" * 80)
    
    return selector, selected_features, report


def demo_with_synthetic_data():
    """Demo with synthetic data if real data is not available."""
    
    print("\n" + "=" * 80)
    print("Feature Selector Demo (Synthetic Data)")
    print("=" * 80)
    
    # Create synthetic data
    print("\n1. Creating synthetic exoplanet data...")
    np.random.seed(42)
    n_samples = 2000
    n_features = 54  # Similar to real dataset
    
    X = np.random.randn(n_samples, n_features)
    
    # Create target with some features being more important
    y = (
        2.0 * X[:, 0] +      # Very important
        1.5 * X[:, 1] +      # Important
        1.0 * X[:, 2] +      # Important
        0.8 * X[:, 5] +      # Moderately important
        0.5 * X[:, 10] +     # Somewhat important
        np.random.randn(n_samples) * 0.5
    )
    y = (y > y.mean()).astype(int)
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    print(f"   Created {n_samples} samples with {n_features} features")
    print(f"   Class distribution: {dict(pd.Series(y).value_counts())}")
    
    # Initialize feature selector
    print("\n2. Analyzing feature importance...")
    selector = FeatureSelector(random_state=42)
    
    importance_df = selector.analyze_importance(X, y, feature_names)
    
    # Select features
    print("\n3. Selecting top 30 features...")
    selected_features = selector.select_features(top_k=30)
    
    print(f"\n   Reduced features from {n_features} to {len(selected_features)}")
    
    # Generate report
    print("\n4. Generating report...")
    report = selector.get_feature_report()
    
    print(f"\n   Top 10 features:")
    for i, feature in enumerate(report['top_10_features'], 1):
        importance = importance_df[importance_df['feature'] == feature]['avg_importance'].values[0]
        print(f"   {i:2d}. {feature:20s} (importance: {importance:.4f})")
    
    # Verify that important features are detected
    important_features = ['feature_0', 'feature_1', 'feature_2', 'feature_5', 'feature_10']
    detected = [f for f in important_features if f in report['top_10_features']]
    
    print(f"\n   Detected {len(detected)}/{len(important_features)} known important features")
    print(f"   Detection rate: {len(detected)/len(important_features)*100:.1f}%")
    
    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    
    return selector, selected_features, report


if __name__ == "__main__":
    try:
        selector, selected_features, report = demo_feature_selector()
    except Exception as e:
        print(f"\nError during demo: {e}")
        print("Falling back to synthetic data demo...")
        selector, selected_features, report = demo_with_synthetic_data()
