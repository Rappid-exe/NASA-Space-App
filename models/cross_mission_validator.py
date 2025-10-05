"""
Cross-mission validator for testing model generalization across NASA missions.
Validates that models trained on one mission work on others.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from sklearn.model_selection import train_test_split
from models.base_classifier import BaseClassifier
from models.model_evaluator import ModelEvaluator
from models.model_trainer import ModelTrainer
from data.multi_dataset_loader import MultiDatasetLoader
from data.data_processor import DataProcessor


class CrossMissionValidator:
    """Validates model performance across different NASA missions."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the cross-mission validator.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.validation_results = {}
        self.validation_matrix = None
        self.dataset_loader = MultiDatasetLoader()
        self.evaluator = ModelEvaluator()
    
    def validate_cross_mission(self, train_mission: str, test_mission: str,
                               model_configs: Optional[Dict] = None,
                               selected_features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train on one mission and test on another.
        
        Args:
            train_mission: Mission to train on ('kepler', 'tess', or 'k2')
            test_mission: Mission to test on ('kepler', 'tess', or 'k2')
            model_configs: Optional model configurations
            selected_features: Optional list of features to use
            
        Returns:
            Dictionary containing validation results
        """
        print(f"\n{'='*80}")
        print(f"Cross-Mission Validation: Train on {train_mission.upper()} → Test on {test_mission.upper()}")
        print(f"{'='*80}\n")
        
        # Load datasets
        train_data = self._load_and_prepare_mission_data(train_mission, selected_features)
        test_data = self._load_and_prepare_mission_data(test_mission, selected_features)
        
        if train_data is None or test_data is None:
            return {'error': 'Failed to load mission data'}
        
        X_train, y_train = train_data
        X_test, y_test = test_data
        
        print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        
        # Train models
        trainer = ModelTrainer(random_state=self.random_state)
        trainer.initialize_models(model_configs)
        
        # Split training data for validation (needed for neural network)
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state, stratify=y_train
        )
        
        trained_models = trainer.train_models(
            X_train_split, y_train_split, X_val_split, y_val_split
        )
        
        # Evaluate on test mission
        evaluation_results = trainer.evaluate_models(X_test, y_test, trained_models)
        
        # Store results
        key = f"{train_mission}_to_{test_mission}"
        self.validation_results[key] = {
            'train_mission': train_mission,
            'test_mission': test_mission,
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'n_features': X_train.shape[1],
            'models': evaluation_results
        }
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"Cross-Mission Validation Results")
        print(f"{'='*80}")
        for model_name, results in evaluation_results.items():
            print(f"{model_name}: Accuracy={results['accuracy']:.4f}, F1={results['f1_score']:.4f}")
        print(f"{'='*80}\n")
        
        return self.validation_results[key]

    
    def generate_validation_matrix(self, missions: Optional[List[str]] = None,
                                   model_configs: Optional[Dict] = None,
                                   selected_features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate validation matrix for all mission combinations.
        
        Args:
            missions: List of missions to validate (None = all available)
            model_configs: Optional model configurations
            selected_features: Optional list of features to use
            
        Returns:
            DataFrame with validation matrix showing accuracy for each combination
        """
        if missions is None:
            missions = ['kepler', 'tess', 'k2']
        
        print(f"\n{'='*80}")
        print(f"Generating Cross-Mission Validation Matrix")
        print(f"Missions: {', '.join([m.upper() for m in missions])}")
        print(f"{'='*80}\n")
        
        # Initialize matrix
        matrix_data = {}
        
        # Validate all combinations
        for train_mission in missions:
            matrix_data[train_mission] = {}
            
            for test_mission in missions:
                print(f"\nValidating: {train_mission.upper()} → {test_mission.upper()}")
                
                try:
                    results = self.validate_cross_mission(
                        train_mission, test_mission, model_configs, selected_features
                    )
                    
                    # Get average accuracy across all models
                    if 'models' in results:
                        accuracies = [m['accuracy'] for m in results['models'].values()]
                        avg_accuracy = np.mean(accuracies) if accuracies else 0.0
                        matrix_data[train_mission][test_mission] = avg_accuracy
                    else:
                        matrix_data[train_mission][test_mission] = 0.0
                
                except Exception as e:
                    print(f"  Error: {e}")
                    matrix_data[train_mission][test_mission] = 0.0
        
        # Create DataFrame
        self.validation_matrix = pd.DataFrame(matrix_data).T
        self.validation_matrix = self.validation_matrix.round(4)
        
        print(f"\n{'='*80}")
        print(f"Validation Matrix (Average Accuracy)")
        print(f"{'='*80}")
        print(self.validation_matrix)
        print(f"{'='*80}\n")
        
        return self.validation_matrix
    
    def validate_combined_dataset(self, missions: Optional[List[str]] = None,
                                  model_configs: Optional[Dict] = None,
                                  selected_features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train on combined dataset and test on individual missions.
        
        Args:
            missions: List of missions to include (None = all available)
            model_configs: Optional model configurations
            selected_features: Optional list of features to use
            
        Returns:
            Dictionary containing validation results for combined training
        """
        if missions is None:
            missions = ['kepler', 'tess', 'k2']
        
        print(f"\n{'='*80}")
        print(f"Combined Dataset Validation")
        print(f"Training on: {', '.join([m.upper() for m in missions])}")
        print(f"{'='*80}\n")
        
        # Load and combine all missions for training
        combined_data = self._load_and_prepare_combined_data(missions, selected_features)
        
        if combined_data is None:
            return {'error': 'Failed to load combined data'}
        
        X_combined, y_combined = combined_data
        
        print(f"Combined training set: {X_combined.shape[0]} samples, {X_combined.shape[1]} features")
        
        # Split for training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_combined, y_combined, test_size=0.2, random_state=self.random_state, stratify=y_combined
        )
        
        # Train models on combined data
        trainer = ModelTrainer(random_state=self.random_state)
        trainer.initialize_models(model_configs)
        trained_models = trainer.train_models(X_train, y_train, X_val, y_val)
        
        # Test on each individual mission
        results = {
            'train_missions': missions,
            'train_samples': X_combined.shape[0],
            'n_features': X_combined.shape[1],
            'test_results': {}
        }
        
        for test_mission in missions:
            print(f"\nTesting on {test_mission.upper()}...")
            
            test_data = self._load_and_prepare_mission_data(test_mission, selected_features)
            if test_data is None:
                continue
            
            X_test, y_test = test_data
            evaluation_results = trainer.evaluate_models(X_test, y_test, trained_models)
            
            results['test_results'][test_mission] = {
                'test_samples': X_test.shape[0],
                'models': evaluation_results
            }
            
            # Print summary
            print(f"Results for {test_mission.upper()}:")
            for model_name, model_results in evaluation_results.items():
                print(f"  {model_name}: Accuracy={model_results['accuracy']:.4f}, F1={model_results['f1_score']:.4f}")
        
        # Store results
        self.validation_results['combined'] = results
        
        return results
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Returns:
            Dictionary containing complete validation report
        """
        if not self.validation_results:
            return {'error': 'No validation results available'}
        
        report = {
            'summary': self._generate_summary(),
            'detailed_results': self.validation_results,
            'validation_matrix': self.validation_matrix.to_dict() if self.validation_matrix is not None else None,
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def print_validation_report(self) -> None:
        """Print formatted validation report."""
        report = self.generate_validation_report()
        
        if 'error' in report:
            print(report['error'])
            return
        
        print(f"\n{'='*80}")
        print(f"Cross-Mission Validation Report")
        print(f"{'='*80}\n")
        
        # Print summary
        if 'summary' in report:
            summary = report['summary']
            print(f"Total Validations: {summary.get('total_validations', 0)}")
            print(f"Average Cross-Mission Accuracy: {summary.get('avg_cross_mission_accuracy', 0):.4f}")
            print(f"Best Cross-Mission Pair: {summary.get('best_pair', 'N/A')}")
            print(f"Worst Cross-Mission Pair: {summary.get('worst_pair', 'N/A')}")
        
        # Print validation matrix
        if self.validation_matrix is not None:
            print(f"\n{'='*80}")
            print(f"Validation Matrix (Average Accuracy)")
            print(f"{'='*80}")
            print(self.validation_matrix)
        
        # Print recommendations
        if 'recommendations' in report:
            print(f"\n{'='*80}")
            print(f"Recommendations")
            print(f"{'='*80}")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"{i}. {rec}")
        
        print(f"\n{'='*80}\n")
    
    def save_validation_report(self, filepath: str) -> None:
        """
        Save validation report to file.
        
        Args:
            filepath: Path to save the report
        """
        report = self.generate_validation_report()
        
        # Convert validation matrix to dict if it exists
        if self.validation_matrix is not None:
            report['validation_matrix'] = self.validation_matrix.to_dict()
        
        import json
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Validation report saved to {filepath}")

    
    def _load_and_prepare_mission_data(self, mission: str,
                                       selected_features: Optional[List[str]] = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Load and prepare data for a specific mission.
        
        Args:
            mission: Mission name
            selected_features: Optional list of features to use
            
        Returns:
            Tuple of (X, y) or None if loading fails
        """
        try:
            # Load dataset
            df = self.dataset_loader.load_dataset(mission)
            
            # Harmonize
            harmonized_df = self.dataset_loader.harmonize_dataset(df, mission)
            
            # Filter to confirmed and false positive only
            harmonized_df = self.dataset_loader.filter_by_disposition(
                harmonized_df, ['CONFIRMED', 'FALSE_POSITIVE']
            )
            
            # Create binary labels
            harmonized_df = self.dataset_loader.create_binary_labels(harmonized_df)
            
            # Process features
            processor = DataProcessor()
            
            # Drop non-feature columns
            feature_df = harmonized_df.drop(columns=['disposition', 'mission', 'label'], errors='ignore')
            
            # Engineer features
            feature_df = processor.engineer_features(feature_df)
            
            # Handle missing values
            feature_df = processor.handle_missing_values(feature_df)
            
            # Select specific features if provided
            if selected_features is not None:
                available_features = [f for f in selected_features if f in feature_df.columns]
                if available_features:
                    feature_df = feature_df[available_features]
            
            # Normalize features
            X = processor.normalize_features(feature_df)
            y = harmonized_df['label'].values
            
            return X, y
        
        except Exception as e:
            print(f"Error loading {mission} data: {e}")
            return None
    
    def _load_and_prepare_combined_data(self, missions: List[str],
                                        selected_features: Optional[List[str]] = None) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Load and prepare combined data from multiple missions.
        
        Args:
            missions: List of mission names
            selected_features: Optional list of features to use
            
        Returns:
            Tuple of (X, y) or None if loading fails
        """
        try:
            # Load all datasets
            datasets = self.dataset_loader.load_all_datasets(missions)
            
            # Harmonize all
            harmonized_datasets = self.dataset_loader.harmonize_all_datasets(datasets)
            
            # Combine
            combined_df = self.dataset_loader.combine_datasets(harmonized_datasets)
            
            # Filter to confirmed and false positive only
            combined_df = self.dataset_loader.filter_by_disposition(
                combined_df, ['CONFIRMED', 'FALSE_POSITIVE']
            )
            
            # Create binary labels
            combined_df = self.dataset_loader.create_binary_labels(combined_df)
            
            # Process features
            processor = DataProcessor()
            
            # Drop non-feature columns
            feature_df = combined_df.drop(columns=['disposition', 'mission', 'label'], errors='ignore')
            
            # Engineer features
            feature_df = processor.engineer_features(feature_df)
            
            # Handle missing values
            feature_df = processor.handle_missing_values(feature_df)
            
            # Select specific features if provided
            if selected_features is not None:
                available_features = [f for f in selected_features if f in feature_df.columns]
                if available_features:
                    feature_df = feature_df[available_features]
            
            # Normalize features
            X = processor.normalize_features(feature_df)
            y = combined_df['label'].values
            
            return X, y
        
        except Exception as e:
            print(f"Error loading combined data: {e}")
            return None
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics from validation results."""
        summary = {
            'total_validations': len(self.validation_results),
            'avg_cross_mission_accuracy': 0.0,
            'best_pair': None,
            'worst_pair': None
        }
        
        if not self.validation_results:
            return summary
        
        # Calculate average accuracy across all validations
        accuracies = []
        pair_accuracies = {}
        
        for key, results in self.validation_results.items():
            if key == 'combined':
                continue
            
            if 'models' in results:
                model_accuracies = [m['accuracy'] for m in results['models'].values()]
                if model_accuracies:
                    avg_acc = np.mean(model_accuracies)
                    accuracies.append(avg_acc)
                    pair_accuracies[key] = avg_acc
        
        if accuracies:
            summary['avg_cross_mission_accuracy'] = float(np.mean(accuracies))
        
        if pair_accuracies:
            best_key = max(pair_accuracies, key=pair_accuracies.get)
            worst_key = min(pair_accuracies, key=pair_accuracies.get)
            summary['best_pair'] = f"{best_key} ({pair_accuracies[best_key]:.4f})"
            summary['worst_pair'] = f"{worst_key} ({pair_accuracies[worst_key]:.4f})"
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if not self.validation_results:
            return ["Run cross-mission validation first to get recommendations"]
        
        summary = self._generate_summary()
        avg_accuracy = summary.get('avg_cross_mission_accuracy', 0)
        
        # Recommendation based on average accuracy
        if avg_accuracy >= 0.80:
            recommendations.append("Excellent cross-mission generalization! Models work well across different missions.")
        elif avg_accuracy >= 0.75:
            recommendations.append("Good cross-mission performance. Models show reasonable generalization.")
        elif avg_accuracy >= 0.70:
            recommendations.append("Moderate cross-mission performance. Consider mission-specific fine-tuning.")
        else:
            recommendations.append("Low cross-mission accuracy. Recommend training on combined datasets.")
        
        # Check if combined results exist
        if 'combined' in self.validation_results:
            recommendations.append("Combined dataset training shows improved generalization across missions.")
        else:
            recommendations.append("Consider training on combined datasets for better cross-mission performance.")
        
        # Check validation matrix for patterns
        if self.validation_matrix is not None:
            # Check diagonal (same mission train/test)
            diagonal_values = [self.validation_matrix.iloc[i, i] for i in range(min(len(self.validation_matrix), len(self.validation_matrix.columns)))]
            if diagonal_values:
                avg_diagonal = np.mean(diagonal_values)
                if avg_diagonal > avg_accuracy + 0.05:
                    recommendations.append("Models perform significantly better on same-mission data. Mission-specific biases detected.")
        
        return recommendations
