"""
Demo script for CrossMissionValidator.
Shows how to validate model performance across different NASA missions.
"""

from models.cross_mission_validator import CrossMissionValidator
import os


def demo_single_cross_mission_validation():
    """Demo: Validate training on one mission and testing on another."""
    print("\n" + "="*80)
    print("DEMO 1: Single Cross-Mission Validation")
    print("="*80)
    
    validator = CrossMissionValidator(random_state=42)
    
    # Validate Kepler → TESS
    print("\nValidating: Train on Kepler, Test on TESS")
    results = validator.validate_cross_mission(
        train_mission='kepler',
        test_mission='tess'
    )
    
    if 'error' not in results:
        print(f"\nResults:")
        print(f"  Training samples: {results['train_samples']}")
        print(f"  Test samples: {results['test_samples']}")
        print(f"  Features: {results['n_features']}")
        print(f"\nModel Performance:")
        for model_name, metrics in results['models'].items():
            print(f"  {model_name}:")
            print(f"    Accuracy: {metrics['accuracy']:.4f}")
            print(f"    F1-Score: {metrics['f1_score']:.4f}")


def demo_validation_matrix():
    """Demo: Generate validation matrix for all mission combinations."""
    print("\n" + "="*80)
    print("DEMO 2: Validation Matrix Generation")
    print("="*80)
    
    validator = CrossMissionValidator(random_state=42)
    
    # Note: This will take a while as it trains models for all combinations
    print("\nGenerating validation matrix for all mission combinations...")
    print("(This may take several minutes)")
    
    # For demo purposes, we'll use a subset
    missions = ['kepler', 'tess']  # Add 'k2' for full matrix
    
    matrix = validator.generate_validation_matrix(missions=missions)
    
    print("\nValidation Matrix:")
    print(matrix)
    
    # Print report
    validator.print_validation_report()


def demo_combined_dataset_validation():
    """Demo: Train on combined datasets and test on individual missions."""
    print("\n" + "="*80)
    print("DEMO 3: Combined Dataset Validation")
    print("="*80)
    
    validator = CrossMissionValidator(random_state=42)
    
    print("\nTraining on combined Kepler + TESS dataset...")
    results = validator.validate_combined_dataset(
        missions=['kepler', 'tess']  # Add 'k2' for all missions
    )
    
    if 'error' not in results:
        print(f"\nCombined Training:")
        print(f"  Total samples: {results['train_samples']}")
        print(f"  Features: {results['n_features']}")
        print(f"  Missions: {', '.join(results['train_missions'])}")
        
        print(f"\nTest Results by Mission:")
        for mission, test_results in results['test_results'].items():
            print(f"\n  {mission.upper()}:")
            print(f"    Test samples: {test_results['test_samples']}")
            for model_name, metrics in test_results['models'].items():
                print(f"    {model_name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")


def demo_with_feature_selection():
    """Demo: Cross-mission validation with selected features."""
    print("\n" + "="*80)
    print("DEMO 4: Cross-Mission Validation with Feature Selection")
    print("="*80)
    
    validator = CrossMissionValidator(random_state=42)
    
    # Example: Use only top features
    selected_features = [
        'period', 'duration', 'depth', 'radius', 'temperature',
        'stellar_temp', 'stellar_radius', 'stellar_mass',
        'period_duration_ratio', 'radius_stellar_radius_ratio'
    ]
    
    print(f"\nUsing {len(selected_features)} selected features")
    print(f"Features: {', '.join(selected_features[:5])}...")
    
    results = validator.validate_cross_mission(
        train_mission='kepler',
        test_mission='tess',
        selected_features=selected_features
    )
    
    if 'error' not in results:
        print(f"\nResults with feature selection:")
        print(f"  Features used: {results['n_features']}")
        for model_name, metrics in results['models'].items():
            print(f"  {model_name}: Accuracy={metrics['accuracy']:.4f}")


def demo_save_validation_report():
    """Demo: Save validation report to file."""
    print("\n" + "="*80)
    print("DEMO 5: Save Validation Report")
    print("="*80)
    
    validator = CrossMissionValidator(random_state=42)
    
    # Run a quick validation
    print("\nRunning validation...")
    validator.validate_cross_mission('kepler', 'tess')
    
    # Save report
    output_path = 'cross_mission_validation_report.json'
    validator.save_validation_report(output_path)
    
    print(f"\nReport saved to: {output_path}")
    
    # Clean up
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Demo file cleaned up")


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("Cross-Mission Validator Demo")
    print("="*80)
    print("\nThis demo shows how to use CrossMissionValidator to test")
    print("model generalization across different NASA missions.")
    
    # Check if datasets are available
    print("\nChecking for datasets...")
    datasets_available = []
    for mission in ['kepler', 'tess', 'k2']:
        path = f'data/raw/{"cumulative.csv" if mission == "kepler" else "toi.csv" if mission == "tess" else "k2targets.csv"}'
        if os.path.exists(path):
            datasets_available.append(mission)
            print(f"  ✓ {mission.upper()} dataset found")
        else:
            print(f"  ✗ {mission.upper()} dataset not found at {path}")
    
    if len(datasets_available) < 2:
        print("\nError: Need at least 2 datasets for cross-mission validation")
        print("Please download datasets first using data/dataset_downloader.py")
        return
    
    print(f"\nFound {len(datasets_available)} datasets: {', '.join([m.upper() for m in datasets_available])}")
    
    # Run demos
    try:
        # Demo 1: Single validation (quick)
        demo_single_cross_mission_validation()
        
        # Demo 4: With feature selection (quick)
        demo_with_feature_selection()
        
        # Demo 5: Save report (quick)
        demo_save_validation_report()
        
        # Note: Demos 2 and 3 are commented out as they take longer
        # Uncomment to run full validation matrix and combined dataset validation
        
        # demo_validation_matrix()
        # demo_combined_dataset_validation()
        
        print("\n" + "="*80)
        print("Demo Complete!")
        print("="*80)
        print("\nTo run full validation matrix or combined dataset validation,")
        print("uncomment the respective demo functions in the main() function.")
        
    except Exception as e:
        print(f"\nError running demo: {e}")
        print("Make sure datasets are downloaded and processed correctly.")


if __name__ == '__main__':
    main()
