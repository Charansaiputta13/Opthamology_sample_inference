"""
Quick verification script to test the DR Classification system installation.
Run this after installing dependencies to validate setup.
"""

print("=" * 80)
print("DR Classification System - Installation Verification")
print("=" * 80)
print()

# Test 1: Import core modules
print("Test 1: Importing core modules...")
try:
    from src.config import Config
    from src.model import DRClassifier
    from src.preprocessing import preprocess_image, check_image_quality
    from src.inference import predict_single_image
    from src.evaluation import evaluate_model
    from src.utils.logging_utils import setup_logging
    from src.utils.validation import validate_image_file
    print(f"[OK] All modules imported successfully")
except Exception as e:
    print(f"[FAIL] Module import failed: {e}")
    exit(1)

print()

# Test 2: Load configuration
print("Test 2: Loading configuration...")
try:
    config = Config()
    print(f"✅ Configuration loaded")
    print(f"   Model path: {config.MODEL_PATH}")
    print(f"   Image size: {config.IMG_SIZE}")
    print(f"   Classes: {config.CLASS_NAMES}")
    print(f"   Confidence threshold: {config.CONFIDENCE_THRESHOLD}")
except Exception as e:
    print(f"❌ Configuration failed: {e}")
    exit(1)

print()

# Test 3: Load model
print("Test 3: Loading model...")
try:
    model = DRClassifier(config.MODEL_PATH)
    print("✅ Model loaded successfully")
    info = model.get_model_info()
    print(f"   Input shape: {info['input_shape']}")
    print(f"   Output shape: {info['output_shape']}")
    print(f"   Last conv layer: {info['last_conv_layer']}")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    print(f"   Make sure {config.MODEL_PATH} exists")
    exit(1)

print()

# Test 4: Check directories
print("Test 4: Checking output directories...")
try:
    import os
    for dir_name, dir_path in [
        ("Logs", config.LOG_DIR),
        ("Predictions", config.PREDICTION_DIR),
        ("Evaluation", config.EVAL_DIR)
    ]:
        if dir_path.exists():
            print(f"   ✅ {dir_name}: {dir_path}")
        else:
            print(f"   ⚠️ {dir_name}: {dir_path} (will be created)")
except Exception as e:
    print(f"❌ Directory check failed: {e}")

print()

# Test 5: Test preprocessing
print("Test 5: Testing preprocessing functions...")
try:
    import numpy as np
    from PIL import Image
    
    # Create dummy image
    dummy_img = Image.new('RGB', (512, 512), color='red')
    
    # Test preprocessing
    processed = preprocess_image(dummy_img, target_size=config.IMG_SIZE)
    assert processed is not None, "Preprocessing returned None"
    assert processed.shape == (224, 224, 3), f"Wrong shape: {processed.shape}"
    
    # Test quality check
    issues, metrics = check_image_quality(dummy_img)
    assert isinstance(issues, list), "Quality issues not a list"
    assert isinstance(metrics, dict), "Quality metrics not a dict"
    
    print("✅ Preprocessing functions work correctly")
    print(f"   Processed shape: {processed.shape}")
    print(f"   Quality metrics: blur={metrics.get('blur_score', 0):.1f}")
except Exception as e:
    print(f"❌ Preprocessing test failed: {e}")

print()

# Test 6: Dependencies
print("Test 6: Checking key dependencies...")
dependencies = [
    ("tensorflow", "2.15.0"),
    ("streamlit", "1.28.0"),
    ("numpy", "1.24.3"),
    ("pandas", "2.0.3"),
    ("pillow", "10.0.0"),
    ("opencv-python", "4.8.0"),
]

for pkg, expected_version in dependencies:
    try:
        if pkg == "opencv-python":
            import cv2
            version = cv2.__version__
            pkg_display = "opencv"
        elif pkg == "pillow":
            from PIL import __version__ as pil_version
            version = pil_version
            pkg_display = "PIL"
        else:
            module = __import__(pkg)
            version = module.__version__
            pkg_display = pkg
        
        print(f"   ✅ {pkg_display}: {version}")
    except Exception as e:
        print(f"   ⚠️ {pkg}: Not found or version mismatch")

print()
print("=" * 80)
print("🎉 Installation verification complete!")
print("=" * 80)
print()
print("Next steps:")
print("1. Run the Streamlit app: streamlit run app.py")
print("2. Upload a retinal image in the Single Image tab")
print("3. Or process a batch of images from a ZIP file")
print()
print("For help, see README.md")
print()
