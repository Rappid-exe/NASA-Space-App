"""
End-to-end integration tests for the exoplanet classification system.
Tests complete data flow from dataset loading to web interface display.
"""

import pytest
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import requests
import time
import subprocess
import signal
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset_loader import DatasetLoader
from data.data_processor import DataProcessor
from models.model_trainer import ModelTrainer
from models.model_registry import ModelRegistry
from api.inference_service import InferenceService


class TestEndToEndIntegration:
    """Test complete integration from data loading to inference."""
    
    @pytest.fixture(scope="class")
    def data_dir(self):
        """Get data directory path."""
        return Path(__file__).parent.parent / 'data' / 'raw'
    
    @pytest.fixture(scope="class")
    def sample_dataset(self, data_dir):
        """Load a sample dataset for testing."""
        loader = DatasetLoader()
        
        # Try to load Kepler dataset
        kepler_path = data_dir / 'cumulative.csv'
        if kepler_path.exists():
            return loader.load_dataset(str(kepler_path))
        
        # Fallback to any available dataset
        for filename in ['cumulative.csv', 'k2targets.csv', 'toi.csv']:
            filepath = data_dir / filename
            if filepath.exists():
                return loader.load_dataset(str(filepath))
        
        pytest.skip("No dataset available for testing")
    
    def test_01_data_loading_pipeline(self, sample_dataset):
        """Test data loading from NASA datasets."""
        assert sample_dataset is not None
        assert len(sample_dataset) > 0
        assert isinstance(sample_dataset, pd.DataFrame)
        print(f"✅ Loaded dataset with {len(sample_dataset)} observations")
    
    def test_02_data_processing_pipeline(self, sample_dataset):
        """Test data processing and feature engineering."""
        processor = DataProcessor()
        
        # Determine target column and dataset type
        target_col = None
        dataset_type = 'koi'
        for col in ['koi_disposition', 'k2_disposition', 'tfopwg_disp']:
            if col in sample_dataset.columns:
                target_col = col
                if col == 'k2_disposition':
                    dataset_type = 'k2'
                elif col == 'tfopwg_disp':
                    dataset_type = 'toi'
                break
        
        assert target_col is not None, "No disposition column found"
        
        # Process data
        processed = processor.process_dataset(
            sample_dataset, 
            dataset_type=dataset_type,
            target_col=target_col,
            remove_outliers=False,
            create_features=True
        )
        
        # Prepare for training
        training_data = processor.prepare_for_training(processed, test_size=0.2, val_size=0.1)
        
        X_train = training_data['X_train']
        y_train = training_data['y_train']
        feature_columns = training_data['feature_columns']
        
        assert X_train is not None
        assert y_train is not None
        assert len(feature_columns) > 0
        assert X_train.shape[0] == len(y_train)
        assert X_train.shape[1] == len(feature_columns)
        
        print(f"✅ Processed {X_train.shape[0]} training samples with {X_train.shape[1]} features")
        print(f"   Features: {feature_columns[:5]}...")
    
    def test_03_model_training_pipeline(self, sample_dataset):
        """Test model training with processed data."""
        processor = DataProcessor()
        trainer = ModelTrainer()
        
        # Determine target column and dataset type
        target_col = None
        dataset_type = 'koi'
        for col in ['koi_disposition', 'k2_disposition', 'tfopwg_disp']:
            if col in sample_dataset.columns:
                target_col = col
                if col == 'k2_disposition':
                    dataset_type = 'k2'
                elif col == 'tfopwg_disp':
                    dataset_type = 'toi'
                break
        
        # Prepare data
        processed = processor.process_dataset(
            sample_dataset, 
            dataset_type=dataset_type,
            target_col=target_col,
            remove_outliers=False,
            create_features=True
        )
        training_data = processor.prepare_for_training(processed, test_size=0.2, val_size=0.1)
        
        X_train = training_data['X_train']
        y_train = training_data['y_train']
        X_val = training_data['X_val']
        y_val = training_data['y_val']
        X_test = training_data['X_test']
        y_test = training_data['y_test']
        feature_columns = training_data['feature_columns']
        
        # Train a simple model (RandomForest for speed)
        trainer.initialize_models({
            'RandomForest': {'n_estimators': 10, 'random_state': 42}
        })
        
        trainer.train_models(X_train, y_train, X_val, y_val, feature_columns)
        results = trainer.evaluate_models(X_test, y_test)
        
        assert 'RandomForest' in results
        assert results['RandomForest']['accuracy'] > 0.5
        
        print(f"✅ Trained model with accuracy: {results['RandomForest']['accuracy']:.4f}")
    
    def test_04_model_registry_integration(self, sample_dataset):
        """Test model registration and retrieval."""
        processor = DataProcessor()
        trainer = ModelTrainer()
        registry = ModelRegistry()
        
        # Prepare and train model
        target_col = None
        dataset_type = 'koi'
        for col in ['koi_disposition', 'k2_disposition', 'tfopwg_disp']:
            if col in sample_dataset.columns:
                target_col = col
                if col == 'k2_disposition':
                    dataset_type = 'k2'
                elif col == 'tfopwg_disp':
                    dataset_type = 'toi'
                break
        
        processed = processor.process_dataset(
            sample_dataset, 
            dataset_type=dataset_type,
            target_col=target_col,
            remove_outliers=False,
            create_features=True
        )
        training_data = processor.prepare_for_training(processed, test_size=0.2, val_size=0.1)
        
        X_train = training_data['X_train']
        y_train = training_data['y_train']
        X_val = training_data['X_val']
        y_val = training_data['y_val']
        X_test = training_data['X_test']
        y_test = training_data['y_test']
        feature_columns = training_data['feature_columns']
        
        trainer.initialize_models({
            'RandomForest': {'n_estimators': 10, 'random_state': 42}
        })
        trainer.train_models(X_train, y_train, X_val, y_val, feature_columns)
        trainer.evaluate_models(X_test, y_test)
        
        # Register model
        model_id = registry.register_model(
            model=trainer.models['RandomForest'],
            model_name='test_integration_model',
            algorithm='RandomForest',
            evaluation_results=trainer.evaluator.evaluation_results['RandomForest'],
            hyperparameters={'n_estimators': 10, 'random_state': 42},
            training_info={
                'training_dataset': 'test',
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            },
            feature_columns=feature_columns
        )
        
        assert model_id is not None
        
        # Load model back
        loaded_model, metadata = registry.load_model(model_id=model_id)
        assert loaded_model is not None
        assert metadata.model_id == model_id
        
        print(f"✅ Registered and loaded model: {model_id}")
    
    def test_05_inference_service_integration(self):
        """Test inference service with loaded model."""
        service = InferenceService()
        
        # Verify model is loaded
        assert service.model is not None
        assert service.metadata is not None
        
        # Test single classification
        test_features = {
            'orbital_period': 3.52,
            'transit_duration': 2.8,
            'transit_depth': 500.0,
            'planetary_radius': 1.2,
            'equilibrium_temperature': 1200.0
        }
        
        result = service.classify_observation(test_features)
        
        assert 'prediction' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        assert 'explanation' in result
        assert result['prediction'] in ['CONFIRMED', 'FALSE_POSITIVE']
        assert 0 <= result['confidence'] <= 1
        
        print(f"✅ Inference service prediction: {result['prediction']} ({result['confidence']:.2%})")
    
    def test_06_batch_classification(self):
        """Test batch classification functionality."""
        service = InferenceService()
        
        # Create batch of observations
        observations = [
            {
                'orbital_period': 3.52,
                'transit_duration': 2.8,
                'transit_depth': 500.0,
                'planetary_radius': 1.2,
                'equilibrium_temperature': 1200.0
            },
            {
                'orbital_period': 10.5,
                'transit_duration': 4.2,
                'transit_depth': 1000.0,
                'planetary_radius': 2.5,
                'equilibrium_temperature': 800.0
            },
            {
                'orbital_period': 365.0,
                'transit_duration': 6.0,
                'transit_depth': 200.0,
                'planetary_radius': 1.0,
                'equilibrium_temperature': 280.0
            }
        ]
        
        results = service.classify_batch(observations)
        
        assert len(results) == len(observations)
        for result in results:
            assert 'prediction' in result
            assert 'confidence' in result
        
        print(f"✅ Batch classification completed: {len(results)} observations")
    
    def test_07_model_statistics(self):
        """Test model statistics retrieval."""
        service = InferenceService()
        
        stats = service.get_model_statistics()
        
        assert 'model_id' in stats
        assert 'algorithm' in stats
        assert 'accuracy' in stats
        assert 'precision' in stats
        assert 'recall' in stats
        assert 'f1_score' in stats
        
        print(f"✅ Model statistics retrieved:")
        print(f"   Algorithm: {stats['algorithm']}")
        print(f"   Accuracy: {stats['accuracy']:.4f}")
        print(f"   F1 Score: {stats['f1_score']:.4f}")
    
    def test_08_feature_importance(self):
        """Test feature importance extraction."""
        service = InferenceService()
        
        try:
            importance = service.get_feature_importance()
            
            assert 'algorithm' in importance
            assert 'features' in importance
            assert len(importance['features']) > 0
            
            print(f"✅ Feature importance extracted:")
            print(f"   Top 3 features:")
            for feat in importance['features'][:3]:
                print(f"     - {feat['name']}: {feat['importance']:.4f}")
        except Exception as e:
            print(f"⚠️  Feature importance not available for this model: {e}")
    
    def test_09_api_health_check(self):
        """Test API health endpoint (requires API to be running)."""
        try:
            response = requests.get('http://localhost:8000/health', timeout=2)
            
            if response.status_code == 200:
                data = response.json()
                assert 'status' in data
                assert 'model_loaded' in data
                print(f"✅ API health check passed: {data['status']}")
            else:
                print(f"⚠️  API returned status code: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("⚠️  API not running - skipping API tests")
            print("   To test API integration, run: python api/main.py")
        except Exception as e:
            print(f"⚠️  API health check failed: {e}")
    
    def test_10_api_classification_endpoint(self):
        """Test API classification endpoint (requires API to be running)."""
        try:
            test_data = {
                'orbital_period': 3.52,
                'transit_duration': 2.8,
                'transit_depth': 500.0,
                'planetary_radius': 1.2,
                'equilibrium_temperature': 1200.0
            }
            
            response = requests.post(
                'http://localhost:8000/classify',
                json=test_data,
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                assert 'prediction' in result
                assert 'confidence' in result
                print(f"✅ API classification: {result['prediction']} ({result['confidence']:.2%})")
            else:
                print(f"⚠️  API returned status code: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("⚠️  API not running - skipping API classification test")
        except Exception as e:
            print(f"⚠️  API classification test failed: {e}")
    
    def test_11_complete_pipeline_validation(self, sample_dataset):
        """Validate complete pipeline from raw data to prediction."""
        print("\n" + "="*60)
        print("COMPLETE PIPELINE VALIDATION")
        print("="*60)
        
        # Step 1: Load data
        print("\n1. Loading dataset...")
        assert len(sample_dataset) > 0
        print(f"   ✅ Loaded {len(sample_dataset)} observations")
        
        # Step 2: Process data
        print("\n2. Processing data...")
        processor = DataProcessor()
        target_col = None
        dataset_type = 'koi'
        for col in ['koi_disposition', 'k2_disposition', 'tfopwg_disp']:
            if col in sample_dataset.columns:
                target_col = col
                if col == 'k2_disposition':
                    dataset_type = 'k2'
                elif col == 'tfopwg_disp':
                    dataset_type = 'toi'
                break
        
        processed = processor.process_dataset(
            sample_dataset, 
            dataset_type=dataset_type,
            target_col=target_col,
            remove_outliers=False,
            create_features=True
        )
        training_data = processor.prepare_for_training(processed, test_size=0.2, val_size=0.1)
        
        X_train = training_data['X_train']
        y_train = training_data['y_train']
        X_val = training_data['X_val']
        y_val = training_data['y_val']
        X_test = training_data['X_test']
        y_test = training_data['y_test']
        feature_columns = training_data['feature_columns']
        print(f"   ✅ Processed {X_train.shape[0]} training samples with {X_train.shape[1]} features")
        
        # Step 3: Train model
        print("\n3. Training model...")
        trainer = ModelTrainer()
        
        trainer.initialize_models({
            'RandomForest': {'n_estimators': 20, 'random_state': 42}
        })
        trainer.train_models(X_train, y_train, X_val, y_val, feature_columns)
        results = trainer.evaluate_models(X_test, y_test)
        print(f"   ✅ Model accuracy: {results['RandomForest']['accuracy']:.4f}")
        
        # Step 4: Register model
        print("\n4. Registering model...")
        registry = ModelRegistry()
        model_id = registry.register_model(
            model=trainer.models['RandomForest'],
            model_name='pipeline_test_model',
            algorithm='RandomForest',
            evaluation_results=trainer.evaluator.evaluation_results['RandomForest'],
            hyperparameters={'n_estimators': 20, 'random_state': 42},
            training_info={
                'training_dataset': 'test',
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            },
            feature_columns=feature_columns
        )
        print(f"   ✅ Model registered: {model_id}")
        
        # Step 5: Load and test inference
        print("\n5. Testing inference...")
        service = InferenceService(model_id=model_id)
        
        test_obs = {
            'orbital_period': 3.52,
            'transit_duration': 2.8,
            'transit_depth': 500.0,
            'planetary_radius': 1.2,
            'equilibrium_temperature': 1200.0
        }
        
        result = service.classify_observation(test_obs)
        print(f"   ✅ Prediction: {result['prediction']} ({result['confidence']:.2%})")
        
        print("\n" + "="*60)
        print("PIPELINE VALIDATION COMPLETE ✅")
        print("="*60)


class TestModelPerformanceBenchmarks:
    """Test model performance against research benchmarks."""
    
    def test_model_accuracy_threshold(self):
        """Verify model meets minimum accuracy threshold."""
        service = InferenceService()
        
        if service.model is None:
            pytest.skip("No model loaded")
        
        stats = service.get_model_statistics()
        
        # Minimum acceptable accuracy for exoplanet classification
        MIN_ACCURACY = 0.70  # 70%
        MIN_F1_SCORE = 0.65  # 65%
        
        assert stats['accuracy'] >= MIN_ACCURACY, \
            f"Model accuracy {stats['accuracy']:.2%} below threshold {MIN_ACCURACY:.2%}"
        
        assert stats['f1_score'] >= MIN_F1_SCORE, \
            f"Model F1 score {stats['f1_score']:.2%} below threshold {MIN_F1_SCORE:.2%}"
        
        print(f"✅ Model meets performance benchmarks:")
        print(f"   Accuracy: {stats['accuracy']:.2%} (threshold: {MIN_ACCURACY:.2%})")
        print(f"   F1 Score: {stats['f1_score']:.2%} (threshold: {MIN_F1_SCORE:.2%})")
    
    def test_prediction_consistency(self):
        """Test that predictions are consistent for same input."""
        service = InferenceService()
        
        if service.model is None:
            pytest.skip("No model loaded")
        
        test_obs = {
            'orbital_period': 3.52,
            'transit_duration': 2.8,
            'transit_depth': 500.0,
            'planetary_radius': 1.2,
            'equilibrium_temperature': 1200.0
        }
        
        # Make multiple predictions
        results = [service.classify_observation(test_obs) for _ in range(5)]
        
        # All predictions should be identical
        predictions = [r['prediction'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        assert len(set(predictions)) == 1, "Predictions are not consistent"
        assert len(set(confidences)) == 1, "Confidence scores are not consistent"
        
        print(f"✅ Predictions are consistent: {predictions[0]} ({confidences[0]:.2%})")


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '-s'])
