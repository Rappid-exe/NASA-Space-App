"""
Test script to validate the API structure and input validation.
Tests API models and validation without requiring a trained model.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))


def test_api_models():
    """Test API request/response models."""
    print("="*80)
    print("TESTING API MODELS AND VALIDATION")
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
        print(f"  Features: {features.dict()}")
    except ValidationError as e:
        print(f"❌ Valid input rejected: {e}")
        return False
    
    # Test valid input without optional field
    print("\n2. Testing valid input without optional temperature...")
    try:
        features = ExoplanetFeatures(
            orbital_period=3.52,
            transit_duration=2.8,
            transit_depth=500.0,
            planetary_radius=1.2
        )
        print("✓ Valid input accepted (without optional field)")
    except ValidationError as e:
        print(f"❌ Valid input rejected: {e}")
        return False
    
    # Test invalid input (negative orbital period)
    print("\n3. Testing invalid input validation (negative orbital period)...")
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
    
    # Test invalid input (negative transit duration)
    print("\n4. Testing invalid input validation (negative transit duration)...")
    try:
        features = ExoplanetFeatures(
            orbital_period=3.52,
            transit_duration=-2.8,
            transit_depth=500.0,
            planetary_radius=1.2
        )
        print("❌ Invalid input accepted (should have been rejected)")
        return False
    except ValidationError:
        print("✓ Invalid input correctly rejected")
    
    # Test invalid input (unreasonably large orbital period)
    print("\n5. Testing invalid input validation (unreasonably large period)...")
    try:
        features = ExoplanetFeatures(
            orbital_period=15000.0,
            transit_duration=2.8,
            transit_depth=500.0,
            planetary_radius=1.2
        )
        print("❌ Invalid input accepted (should have been rejected)")
        return False
    except ValidationError:
        print("✓ Invalid input correctly rejected")
    
    # Test batch request
    print("\n6. Testing batch request validation...")
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
    print("\n7. Testing empty batch validation...")
    try:
        batch = BatchClassificationRequest(observations=[])
        print("❌ Empty batch accepted (should have been rejected)")
        return False
    except ValidationError:
        print("✓ Empty batch correctly rejected")
    
    # Test large batch (over limit)
    print("\n8. Testing oversized batch validation...")
    try:
        large_observations = [
            ExoplanetFeatures(
                orbital_period=3.52,
                transit_duration=2.8,
                transit_depth=500.0,
                planetary_radius=1.2
            )
            for _ in range(1001)
        ]
        batch = BatchClassificationRequest(observations=large_observations)
        print("❌ Oversized batch accepted (should have been rejected)")
        return False
    except ValidationError:
        print("✓ Oversized batch correctly rejected")
    
    print("\n" + "="*80)
    print("✓ ALL API MODEL TESTS PASSED")
    print("="*80)
    return True


def test_api_structure():
    """Test API structure and endpoints."""
    print("\n" + "="*80)
    print("TESTING API STRUCTURE")
    print("="*80)
    
    from api.main import app
    from fastapi.testclient import TestClient
    
    client = TestClient(app)
    
    # Test root endpoint
    print("\n1. Testing root endpoint...")
    try:
        response = client.get("/")
        if response.status_code == 200:
            print("✓ Root endpoint accessible")
            print(f"  Response: {response.json()}")
        else:
            print(f"❌ Root endpoint returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        return False
    
    # Test health endpoint
    print("\n2. Testing health endpoint...")
    try:
        response = client.get("/health")
        if response.status_code == 200:
            print("✓ Health endpoint accessible")
            data = response.json()
            print(f"  Status: {data['status']}")
            print(f"  Model Loaded: {data['model_loaded']}")
        else:
            print(f"❌ Health endpoint returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint failed: {e}")
        return False
    
    # Test models list endpoint
    print("\n3. Testing models list endpoint...")
    try:
        response = client.get("/models/list")
        if response.status_code == 200:
            print("✓ Models list endpoint accessible")
            models = response.json()
            print(f"  Found {len(models)} model(s)")
        else:
            print(f"❌ Models list endpoint returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Models list endpoint failed: {e}")
        return False
    
    print("\n" + "="*80)
    print("✓ ALL API STRUCTURE TESTS PASSED")
    print("="*80)
    return True


def test_inference_service_structure():
    """Test inference service structure without requiring a model."""
    print("\n" + "="*80)
    print("TESTING INFERENCE SERVICE STRUCTURE")
    print("="*80)
    
    from api.inference_service import InferenceService
    
    # Test initialization
    print("\n1. Testing inference service initialization...")
    try:
        service = InferenceService()
        print("✓ Inference service initialized")
        print(f"  Model loaded: {service.model is not None}")
    except Exception as e:
        print(f"⚠ Inference service initialized with warning: {e}")
    
    # Test derived features creation
    print("\n2. Testing derived features creation...")
    try:
        service = InferenceService()
        test_features = {
            'orbital_period': 3.52,
            'transit_duration': 2.8,
            'transit_depth': 500.0,
            'planetary_radius': 1.2,
            'equilibrium_temperature': 1200.0
        }
        derived = service._create_derived_features(test_features)
        print(f"✓ Created {len(derived)} derived features")
        print(f"  Features: {list(derived.keys())}")
    except Exception as e:
        print(f"❌ Derived features creation failed: {e}")
        return False
    
    print("\n" + "="*80)
    print("✓ ALL INFERENCE SERVICE STRUCTURE TESTS PASSED")
    print("="*80)
    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("EXOPLANET CLASSIFICATION API STRUCTURE TESTS")
    print("="*80)
    
    # Test API models
    models_passed = test_api_models()
    
    # Test API structure
    structure_passed = test_api_structure()
    
    # Test inference service structure
    service_passed = test_inference_service_structure()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"API Model Tests: {'✓ PASSED' if models_passed else '❌ FAILED'}")
    print(f"API Structure Tests: {'✓ PASSED' if structure_passed else '❌ FAILED'}")
    print(f"Inference Service Tests: {'✓ PASSED' if service_passed else '❌ FAILED'}")
    
    if models_passed and structure_passed and service_passed:
        print("\n✓ ALL STRUCTURE TESTS PASSED")
        print("\nThe API is properly structured and ready to use.")
        print("\nNote: To test full functionality, train a model first:")
        print("  python test_model_training.py")
        print("\nThen start the API server:")
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
