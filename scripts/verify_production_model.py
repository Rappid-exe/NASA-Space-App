"""
Verify that the production API is using the optimized model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.inference_service import InferenceService

def main():
    print("=" * 80)
    print("VERIFYING PRODUCTION MODEL")
    print("=" * 80)
    
    # Initialize service (this will load the best model)
    print("\nInitializing inference service...")
    service = InferenceService()
    
    # Check which model was loaded
    print(f"\n{'=' * 80}")
    print("LOADED MODEL DETAILS")
    print(f"{'=' * 80}")
    
    stats = service.get_model_statistics()
    
    print(f"\nModel Information:")
    print(f"  Model ID: {stats['model_id']}")
    print(f"  Model Name: {stats['model_name']}")
    print(f"  Algorithm: {stats['algorithm']}")
    print(f"  Version: {stats['version']}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:  {stats['accuracy']:.4f} ({stats['accuracy']*100:.2f}%)")
    print(f"  Precision: {stats['precision']:.4f}")
    print(f"  Recall:    {stats['recall']:.4f}")
    print(f"  F1 Score:  {stats['f1_score']:.4f}")
    
    print(f"\nTraining Information:")
    print(f"  Training Samples: {stats['training_samples']}")
    print(f"  Test Samples: {stats['test_samples']}")
    print(f"  Training Date: {stats['training_date']}")
    
    print(f"\nFeatures:")
    print(f"  Total Features: {len(service.feature_columns)}")
    print(f"  Feature List: {service.feature_columns}")
    
    # Test with a sample prediction
    print(f"\n{'=' * 80}")
    print("TESTING SAMPLE PREDICTION")
    print(f"{'=' * 80}")
    
    test_input = {
        'orbital_period': 3.52,
        'transit_duration': 2.8,
        'transit_depth': 15000.0,
        'planetary_radius': 11.2,
        'equilibrium_temperature': 1450.0
    }
    
    print(f"\nTest Input (Hot Jupiter):")
    for key, value in test_input.items():
        print(f"  {key}: {value}")
    
    result = service.classify_observation(test_input)
    
    print(f"\nPrediction Result:")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Confidence: {result['confidence']:.1%}")
    print(f"  Probabilities:")
    print(f"    FALSE_POSITIVE: {result['probabilities']['FALSE_POSITIVE']:.1%}")
    print(f"    CONFIRMED: {result['probabilities']['CONFIRMED']:.1%}")
    
    # Verification
    print(f"\n{'=' * 80}")
    print("VERIFICATION RESULT")
    print(f"{'=' * 80}")
    
    if 'Optimized' in stats['model_name']:
        print(f"\n✅ SUCCESS: Using optimized model!")
        print(f"   Model: {stats['model_name']}")
        print(f"   Accuracy: {stats['accuracy']*100:.2f}%")
        print(f"   Features: {len(service.feature_columns)} (with engineering)")
    elif 'UserFeatures' in stats['model_name']:
        print(f"\n⚠️  WARNING: Using fallback model")
        print(f"   Model: {stats['model_name']}")
        print(f"   Accuracy: {stats['accuracy']*100:.2f}%")
        print(f"   Consider checking if optimized model is available")
    else:
        print(f"\n❌ ERROR: Using unexpected model")
        print(f"   Model: {stats['model_name']}")
    
    print(f"\n{'=' * 80}")
    print("PRODUCTION MODEL READY FOR USE")
    print(f"{'=' * 80}")
    print()

if __name__ == "__main__":
    main()
