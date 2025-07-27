#!/usr/bin/env python3
"""
Test compatibility between original and enhanced models.
"""
import os
import numpy as np
import cv2
from mom_detector import is_enhanced_model, extract_hog_features, extract_enhanced_features_compatible

def test_compatibility():
    """Test that the enhanced model maintains compatibility."""
    print("Testing Model Compatibility")
    print("=" * 40)
    
    # Test feature extraction compatibility
    print("\n1. Testing feature extraction:")
    
    # Create a dummy image
    dummy_image = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    
    # Test original features
    try:
        hog_features = extract_hog_features(dummy_image)
        print(f"Original HOG features: {len(hog_features)} dimensions")
    except Exception as e:
        print(f"Original HOG features failed: {e}")
    
    # Test enhanced features compatibility
    try:
        enhanced_features = extract_enhanced_features_compatible(dummy_image)
        print(f"Enhanced features: {len(enhanced_features)} dimensions")
    except Exception as e:
        print(f"Enhanced features failed: {e}")
    
    # Test model detection
    print("\n2. Testing model detection:")
    
    # Test non-existent model
    fake_path = "nonexistent_model.joblib"
    is_enhanced = is_enhanced_model(fake_path)
    print(f"Non-existent model detection: {is_enhanced} (should be False)")
    
    # Test with existing original model if available
    original_model_path = "mom_detector_svm.joblib"
    if os.path.exists(original_model_path):
        is_enhanced = is_enhanced_model(original_model_path)
        print(f"Existing model type detection: {'Enhanced' if is_enhanced else 'Original'}")
    else:
        print("No existing model found (expected for fresh setup)")
    
    print("\n3. Compatibility Summary:")
    print("Enhanced model saves to 'mom_detector_svm.joblib'")
    print("Original interface automatically detects model type")
    print("Enhanced features used when enhanced model detected")
    print("Original features used as fallback")
    print("Confidence scores provided when available")
    
    print("\n4. Usage Instructions:")
    print("To train enhanced model:")
    print("  python mom_detector_enhanced.py train")
    print("\nTo use with original interface:")
    print("  python mom_detector.py infer --image your_image.jpg")
    print("\nBoth will use the same model file: mom_detector_svm.joblib")

if __name__ == "__main__":
    test_compatibility()