"""
Inference Examples
==================
Demonstrates various inference patterns with the PyTorch model.

Usage:
    python examples_inference.py --single /path/to/image.jpg
    python examples_inference.py --batch /path/to/images.zip
"""

import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.model import DRClassifier
from src.preprocessing import load_image_safe, preprocess_image
from src.inference import predict_single_image, predict_batch_from_zip


def example_single_inference():
    """Example: Single image inference with CAM."""
    print("=" * 70)
    print("EXAMPLE 1: Single Image Inference with CAM")
    print("=" * 70)
    
    config = Config()
    model = DRClassifier(config.MODEL_PATH)
    
    # Create a dummy image
    print("\nGenerating dummy retinal image...")
    dummy_img = Image.new('RGB', (512, 512), color=(100, 50, 30))
    
    # Add some texture
    img_array = np.array(dummy_img)
    for _ in range(100):
        y, x = np.random.randint(0, 512, 2)
        r = np.random.randint(5, 50)
        yy, xx = np.ogrid[:512, :512]
        mask = (yy - y) ** 2 + (xx - x) ** 2 <= r ** 2
        img_array[mask] = np.random.randint(0, 256, 3)
    
    dummy_img = Image.fromarray(img_array)
    dummy_img.save("dummy_retinal.jpg")
    
    print("Saved dummy_retinal.jpg")
    print("\nRunning inference with GradCAM...")
    
    result = predict_single_image(
        img_path="dummy_retinal.jpg",
        model=model,
        config=config,
        cam_method="GradCAM"
    )
    
    if result['success']:
        print(f"✅ Inference successful!")
        print(f"\n   Filename: {result['filename']}")
        print(f"   Predicted Class: {result['class_name']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"\n   All Probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"      {class_name:<15}: {prob:.4f}")
        
        print(f"\n   Quality Issues: {result['quality_issues']}")
        print(f"   Needs Review: {result['needs_review']}")
        
        if 'cam_heatmap' in result and result['cam_heatmap'] is not None:
            print(f"   CAM Heatmap Shape: {result['cam_heatmap'].shape}")
            print(f"   CAM Method Used: {result['cam_method']}")
    else:
        print(f"❌ Inference failed: {result['error']}")
    
    print()


def example_multiple_cam_methods():
    """Example: Compare multiple CAM methods on same image."""
    print("=" * 70)
    print("EXAMPLE 2: Multiple CAM Methods Comparison")
    print("=" * 70)
    
    config = Config()
    model = DRClassifier(config.MODEL_PATH)
    
    # Create dummy image
    print("\nGenerating dummy retinal image...")
    img_path = "dummy_retinal.jpg"
    
    img_pil = load_image_safe(img_path)
    img_array = preprocess_image(img_pil, target_size=config.IMG_SIZE)
    
    if img_array is None:
        print("❌ Failed to preprocess image")
        return
    
    print("Computing CAM with each method...")
    timing_results = {}
    
    import time
    
    for method in config.CAM_METHODS:
        start = time.time()
        heatmap = model.compute_cam(img_array, method=method)
        elapsed = time.time() - start
        
        timing_results[method] = elapsed
        
        if heatmap is not None:
            print(f"  ✅ {method:<12} - Shape: {heatmap.shape}, Time: {elapsed:.3f}s")
        else:
            print(f"  ❌ {method:<12} - Failed")
    
    print("\nCAM Method Comparison:")
    print("-" * 40)
    print(f"{'Method':<12} | {'Time (s)':<10} | {'Speed Rank'}")
    print("-" * 40)
    
    sorted_methods = sorted(timing_results.items(), key=lambda x: x[1])
    for rank, (method, elapsed) in enumerate(sorted_methods, 1):
        print(f"{method:<12} | {elapsed:<10.4f} | #{rank}")
    
    print()


def example_batch_processing():
    """Example: Batch processing from ZIP file."""
    print("=" * 70)
    print("EXAMPLE 3: Batch Processing from ZIP")
    print("=" * 70)
    
    config = Config()
    model = DRClassifier(config.MODEL_PATH)
    
    # Create dummy ZIP with multiple images
    import zipfile
    
    print("\nCreating dummy ZIP file with images...")
    zip_path = Path("dummy_images.zip")
    
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for i in range(3):
            # Create dummy image
            img_array = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            
            # Save to memory
            import io
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            # Add to ZIP
            zf.writestr(f'image_{i:03d}.jpg', img_bytes.read())
    
    print(f"Created {zip_path} with 3 dummy images")
    
    print("\nRunning batch inference...")
    try:
        results_df = predict_batch_from_zip(
            zip_path=str(zip_path),
            model=model,
            config=config,
            use_cam=False  # Skip CAM for speed
        )
        
        print(f"✅ Batch processing complete!")
        print(f"\n   Processed: {len(results_df)} images")
        print(f"\n   Results Summary:")
        print(results_df[['filename', 'class_name', 'confidence']].to_string())
        
        # Save results
        results_df.to_csv("batch_results.csv", index=False)
        print(f"\n   Saved results to: batch_results.csv")
        
    except Exception as e:
        print(f"⚠️  Note: Batch processing requires real images")
        print(f"   Error: {e}")
    
    print()


def example_advanced_usage():
    """Example: Advanced usage patterns."""
    print("=" * 70)
    print("EXAMPLE 4: Advanced Usage Patterns")
    print("=" * 70)
    
    config = Config()
    model = DRClassifier(config.MODEL_PATH)
    
    # Create dummy image
    img = Image.new('RGB', (224, 224), color=(128, 128, 128))
    img.save("test_image.jpg")
    
    img_pil = load_image_safe("test_image.jpg")
    img_array = preprocess_image(img_pil, target_size=config.IMG_SIZE)
    
    print("\n1. Direct Model Prediction")
    print("-" * 40)
    predicted_class, probs = model.predict(img_array)
    print(f"   Predicted: {config.CLASS_NAMES[predicted_class]}")
    print(f"   Probabilities: {probs}")
    
    print("\n2. Target-Specific CAM")
    print("-" * 40)
    for target_class in range(len(config.CLASS_NAMES)):
        heatmap = model.compute_cam(
            img_array,
            method="GradCAM",
            target_class=target_class
        )
        if heatmap is not None:
            print(f"   {config.CLASS_NAMES[target_class]:<15} - Heatmap generated")
    
    print("\n3. Model Information")
    print("-" * 40)
    info = model.get_model_info()
    for key, value in info.items():
        if isinstance(value, list):
            print(f"   {key}: {', '.join(value)}")
        else:
            print(f"   {key}: {value}")
    
    print("\n4. Batch Prediction")
    print("-" * 40)
    batch_imgs = np.random.randn(4, 224, 224, 3).astype(np.float32) * 0.2 + 0.5
    batch_probs = model.predict_batch(batch_imgs, batch_size=2)
    print(f"   Input batch shape: {batch_imgs.shape}")
    print(f"   Output probs shape: {batch_probs.shape}")
    print(f"   Sample predictions: {np.argmax(batch_probs, axis=1)}")
    
    print()


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Inference examples")
    parser.add_argument('--single', help='Single inference example')
    parser.add_argument('--batch', help='Batch inference example')
    parser.add_argument('--all', action='store_true', help='Run all examples')
    
    args = parser.parse_args()
    
    try:
        if args.all or (not args.single and not args.batch):
            example_single_inference()
            example_multiple_cam_methods()
            example_batch_processing()
            example_advanced_usage()
        
        if args.single:
            example_single_inference()
        
        if args.batch:
            example_batch_processing()
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
