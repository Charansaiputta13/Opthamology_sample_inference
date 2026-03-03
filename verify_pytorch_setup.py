"""
PyTorch Setup Verification Script
==================================
Verifies that all PyTorch dependencies are installed and model loads correctly.

Run with: python verify_pytorch_setup.py
"""

import sys
from pathlib import Path

def check_imports():
    """Check if all required packages are importable."""
    print("=" * 60)
    print("CHECKING PYTORCH DEPENDENCIES")
    print("=" * 60)
    
    packages = {
        'torch': 'PyTorch (Deep Learning Framework)',
        'torchvision': 'Torchvision (CV Models)',
        'pytorch_grad_cam': 'Grad-CAM (Explainability)',
        'numpy': 'NumPy (Numerical Computing)',
        'cv2': 'OpenCV (Image Processing)',
        'PIL': 'Pillow (Image Loading)',
        'pandas': 'Pandas (Data Handling)',
        'sklearn': 'Scikit-learn (Metrics)',
        'streamlit': 'Streamlit (Web UI)',
        'tqdm': 'TQDM (Progress Bars)',
    }
    
    missing = []
    installed = []
    
    for import_name, display_name in packages.items():
        try:
            __import__(import_name)
            print(f"✅ {display_name:<35} - {import_name}")
            installed.append(import_name)
        except ImportError:
            print(f"❌ {display_name:<35} - {import_name}")
            missing.append(import_name)
    
    print()
    return missing, installed


def check_pytorch_details():
    """Print PyTorch version and capabilities."""
    print("=" * 60)
    print("PYTORCH DETAILS")
    print("=" * 60)
    
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"Python Version: {sys.version}")
        print()
    except Exception as e:
        print(f"⚠️  Error getting PyTorch details: {e}")
        print()


def check_model_file():
    """Verify that the .pth model file exists."""
    print("=" * 60)
    print("MODEL FILE VERIFICATION")
    print("=" * 60)
    
    model_path = Path(__file__).parent / "pretrained" / "dr_mobilenetv2_5class.pth"
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model file exists")
        print(f"   Path: {model_path}")
        print(f"   Size: {size_mb:.2f} MB")
        print()
        return True
    else:
        print(f"❌ Model file not found")
        print(f"   Expected path: {model_path}")
        print()
        return False


def test_model_loading():
    """Test actual model loading."""
    print("=" * 60)
    print("MODEL LOADING TEST")
    print("=" * 60)
    
    try:
        print("Importing DRClassifier...")
        sys.path.insert(0, str(Path(__file__).parent))
        from src.config import Config
        from src.model import DRClassifier
        
        print("✅ Imports successful")
        print()
        
        print("Loading configuration...")
        config = Config()
        print(f"✅ Config loaded: {config.MODEL_PATH}")
        print()
        
        print("Loading model with .pth weights...")
        model = DRClassifier(config.MODEL_PATH)
        print("✅ Model loaded successfully!")
        
        # Print model info
        info = model.get_model_info()
        print()
        print("Model Information:")
        for key, value in info.items():
            if key == 'cam_methods':
                print(f"  {key}: {', '.join(value)}")
            else:
                print(f"  {key}: {value}")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_cam_methods():
    """Test CAM method availability."""
    print("=" * 60)
    print("CAM METHODS AVAILABILITY")
    print("=" * 60)
    
    try:
        from pytorch_grad_cam import (
            GradCAM, GradCAMPlusPlus, ScoreCAM, EigenCAM, LayerCAM
        )
        
        methods = {
            'GradCAM': GradCAM,
            'GradCAM++': GradCAMPlusPlus,
            'ScoreCAM': ScoreCAM,
            'EigenCAM': EigenCAM,
            'LayerCAM': LayerCAM,
        }
        
        print("Available CAM Methods:")
        for method_name, method_class in methods.items():
            print(f"  ✅ {method_name:<12} - {method_class.__name__}")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ CAM import failed: {e}")
        print()
        return False


def test_inference():
    """Test basic inference on a dummy image."""
    print("=" * 60)
    print("INFERENCE TEST (Dummy Image)")
    print("=" * 60)
    
    try:
        import numpy as np
        from src.config import Config
        from src.model import DRClassifier
        
        config = Config()
        model = DRClassifier(config.MODEL_PATH)
        
        # Create dummy image (ImageNet normalized)
        dummy_img = np.random.randn(224, 224, 3).astype(np.float32) * 0.2 + 0.5
        dummy_img = np.clip(dummy_img, 0, 1)
        
        print("Running inference on dummy image...")
        predicted_class, probs = model.predict(dummy_img)
        
        print(f"✅ Inference successful!")
        print(f"   Predicted class: {predicted_class} ({config.CLASS_NAMES[predicted_class]})")
        print(f"   Confidence: {probs[predicted_class]:.3f}")
        print(f"   All probabilities:")
        for i, prob in enumerate(probs):
            print(f"      {config.CLASS_NAMES[i]:<15}: {prob:.4f}")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_cam_computation():
    """Test CAM computation."""
    print("=" * 60)
    print("CAM COMPUTATION TEST (Dummy Image)")
    print("=" * 60)
    
    try:
        import numpy as np
        from src.config import Config
        from src.model import DRClassifier
        
        config = Config()
        model = DRClassifier(config.MODEL_PATH)
        
        # Create dummy image
        dummy_img = np.random.randn(224, 224, 3).astype(np.float32) * 0.2 + 0.5
        dummy_img = np.clip(dummy_img, 0, 1)
        
        print("Computing CAM with each method...")
        for method in config.CAM_METHODS:
            try:
                heatmap = model.compute_cam(dummy_img, method=method)
                status = "✅" if heatmap is not None else "⚠️"
                shape = heatmap.shape if heatmap is not None else "None"
                print(f"  {status} {method:<12} - Shape: {shape}")
            except Exception as e:
                print(f"  ❌ {method:<12} - Error: {e}")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ CAM test failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def main():
    """Run all verification tests."""
    print()
    print("╔" + "=" * 58 + "╗")
    print("║" + " DR CLASSIFICATION SYSTEM - PYTORCH SETUP VERIFICATION ".center(58) + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # Run checks
    missing, installed = check_imports()
    check_pytorch_details()
    model_exists = check_model_file()
    model_loads = test_model_loading()
    cam_available = test_cam_methods()
    inference_ok = test_inference()
    cam_ok = test_cam_computation()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if missing:
        print(f"❌ Missing {len(missing)} package(s):")
        for pkg in missing:
            print(f"   - {pkg}")
        print()
        print("Install missing packages with:")
        print(f"   pip install {' '.join(missing)}")
        print()
    else:
        print("✅ All packages installed!")
        print()
    
    all_ok = (
        not missing and
        model_exists and
        model_loads and
        cam_available and
        inference_ok and
        cam_ok
    )
    
    if all_ok:
        print("🎉 SETUP COMPLETE - All checks passed!")
        print()
        print("Next steps:")
        print("  1. Review configuration in src/config.py")
        print("  2. Add your retinal images to data/images/")
        print("  3. Run the Streamlit app: streamlit run app.py")
        print()
        return 0
    else:
        print("⚠️  Setup incomplete - please fix errors above")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
