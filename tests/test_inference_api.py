"""
Test script for the exoplanet classification inference API.
Tests all endpoints and validates functionality.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from api.inference_service import InferenceService
from models.model_registry import ModelRegistry


def test_inference_service():
    """Test the inference service functionality."""
    print("="*80)
    print("TESTING INFERENCE SERVICE")
    print("="*80)
    
    # Check if models are available
    registry = ModelRegistry()
    models = registry.list_models()
    
    if not models:
        print("\n❌ No models found in registry. Please train a model first.")
        print("Run: python test_model_training.py")
        return False
    
    print(f"\n✓ Found {len(models)} model(s) in registry")
    
    # Initialize inference service
    print("\n1. Initializing inference service...")
    try:
        service = InferenceService()
        if service.model is None:
            print("❌ Failed to load model")
            return False
        print(f"✓ Loaded model: {service.metadata.model_id}")
        print(f"  Algorithm: {service.metadata.algorithm}")
        print(f"  F1 Score: {service.metadata.f1_score:.4f}")
    except Exception as e:
        print(f"❌ Error initializing service: {e}")
        return False
    
    # Test single observation classification
    print("\n2. Testing single observation classification...")
    test_observation = {
        'orbital_period': 3.52,
        'transit_duration': 2.8,
        'transit_depth': 500.0,
        'planetary_radius': 1.2,
        'equilibrium_temperature': 1200.0
    }
    
    try:
        result = service.classify_observation(test_observation)
        print(f"✓ Classification successful")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Probabilities: {result['probabilities']}")
        print(f"  Explanation: {result['explanation'][:100]}...")
    except Exception as e:
        print(f"❌ Classification failed: {e}")
        return False
    
    # Test batch classification
    print("\n3. Testing batch classification...")
    batch_observations = [
        {
            'orbital_period': 3.52,
            'transit_duration': 2.8,
            'transit_depth': 500.0,
            'planetary_radius': 1.2,
            'equilibrium_temperature': 1200.0
        },
        {
            'orbital_period': 365.25,
            'transit_duration': 6.5,
            'transit_depth': 84.0,
            'planetary_radius': 1.0,
            'equilibrium_temperature': 288.0
        },
        {
            'orbital_period': 0.5,
            'transit_duration': 1.2,
            'transit_depth': 2000.0,
            'planetary_radius': 0.8,
            'equilibrium_temperature': 2500.0
        }
    ]
    
    try:
        results = service.classify_batch(batch_observations)
        print(f"✓ Batch classification successful")
        print(f"  Processed: {len(results)} observations")
        
        confirmed_count = sum(1 for r in results if r['prediction'] == 'CONFIRMED')
        false_positive_count = sum(1 for r in results if r['prediction'] == 'FALSE_POSITIVE')
        
        print(f"  Results: {confirmed_count} CONFIRMED, {false_positive_count} FALSE_POSITIVE")
        
        for i, result in enumerate(results, 1):
            print(f"\n  Observation {i}:")
            print(f"    Prediction: {result['prediction']}")
            print(f"    Confidence: {result['confidence']:.2%}")
    except Exception as e:
        print(f"❌ Batch classification failed: {e}")
        return False
    
    # Test model statistics
    print("\n4. Testing model statistics retrieval...")
    try:
        stats = service.get_model_statistics()
        print(f"✓ Statistics retrieved successfully")
        print(f"  Model: {stats['model_name']} v{stats['version']}")
        print(f"  Algorithm: {stats['algorithm']}")
        print(f"  Accuracy: {stats['accuracy']:.4f}")
        print(f"  Precision: {stats['precision']:.4f}")
        print(f"  Recall: {stats['recall']:.4f}")
        print(f"  F1 Score: {stats['f1_score']:.4f}")
        print(f"  Training Samples: {stats['training_samples']}")
    except Exception as e:
        print(f"❌ Failed to retrieve statistics: {e}")
        return False
    
    # Test derived features
    print("\n5. Testing derived feature creation...")
    try:
        derived = service._create_derived_features(test_observation)
        print(f"✓ Created {len(derived)} derived features")
        print(f"  Features: {list(derived.keys())}")
    except Exception as e:
        print(f"❌ Failed to create derived features: {e}")
        return False
    
    print("\n" + "="*80)
    print("✓ ALL INFERENCE SERVICE TESTS PASSED")
    print("="*80)
    return True


def test_api_models():
    """Test API request/response models."""
    print("\n" + "="*80)
    print("TESTING API MODELS")
    print("="*80)
    
    from api.main import ExoplanetFeatures, BatchClassificationRequest
    from pydantic import ValidationError
    
    # Test valid input
    print("\n1. Testing valid input validation...")
    try:
        features = ExoplanetFeatures(
            orbital_period=3.52,
            transit_duration=2.8,
            transit_depth=500.0,
            planetary_radius=1.2,
            equilibrium_temperature=1200.0
        )
        print("✓ Valid input accepted")
    except ValidationError as e:
        print(f"❌ Valid input rejected: {e}")
        return False
    
    # Test invalid input (negative values)
    print("\n2. Testing invalid input validation (negative values)...")
    try:
        features = ExoplanetFeatures(
            orbital_period=-3.52,
            transit_duration=2.8,
            transit_depth=500.0,
            planetary_radius=1.2
        )
        print("❌ Invalid input accepted (should have been rejected)")
        return False
    except ValidationError:
        print("✓ Invalid input correctly rejected")
    
    # Test batch request
    print("\n3. Testing batch request validation...")
    try:
        batch = BatchClassificationRequest(
            observations=[
                ExoplanetFeatures(
                    orbital_period=3.52,
                    transit_duration=2.8,
                    transit_depth=500.0,
                    planetary_radius=1.2
                ),
                ExoplanetFeatures(
                    orbital_period=365.25,
                    transit_duration=6.5,
                    transit_depth=84.0,
                    planetary_radius=1.0
                )
            ]
        )
        print(f"✓ Batch request validated ({len(batch.observations)} observations)")
    except ValidationError as e:
        print(f"❌ Batch validation failed: {e}")
        return False
    
    # Test empty batch
    print("\n4. Testing empty batch validation...")
    try:
        batch = BatchClassificationRequest(observations=[])
        print("❌ Empty batch accepted (should have been rejected)")
        return False
    except ValidationError:
        print("✓ Empty batch correctly rejected")
    
    print("\n" + "="*80)
    print("✓ ALL API MODEL TESTS PASSED")
    print("="*80)
    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("EXOPLANET CLASSIFICATION API TESTS")
    print("="*80)
    
    # Test inference service
    service_passed = test_inference_service()
    
    # Test API models
    models_passed = test_api_models()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Inference Service Tests: {'✓ PASSED' if service_passed else '❌ FAILED'}")
    print(f"API Model Tests: {'✓ PASSED' if models_passed else '❌ FAILED'}")
    
    if service_passed and models_passed:
        print("\n✓ ALL TESTS PASSED")
        print("\nTo start the API server, run:")
        print("  python -m uvicorn api.main:app --reload")
        print("\nAPI will be available at:")
        print("  http://localhost:8000")
        print("  http://localhost:8000/docs (Interactive API documentation)")
        return True
    else:
        print("\n❌ SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
