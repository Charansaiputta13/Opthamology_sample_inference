"""
CAM Visualization Examples
==========================
Demonstrates all 5 CAM methods with sample images.

Usage:
    python examples_cam_visualization.py /path/to/image.jpg
    python examples_cam_visualization.py --random  # Generate dummy image
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.model import DRClassifier
from src.preprocessing import load_image_safe, preprocess_image


def create_dummy_image():
    """Create a dummy retinal-like image for testing."""
    print("Generating dummy retinal image...")
    
    # Create random image with some structure
    img = np.random.rand(512, 512, 3) * 255
    
    # Add circular gradient to simulate vessel patterns
    y, x = np.ogrid[:512, :512]
    mask = (x - 256) ** 2 + (y - 256) ** 2 <= 200 ** 2
    img[mask] = img[mask] * 0.8  # Darken center
    
    # Add some noise
    img += np.random.randn(512, 512, 3) * 20
    
    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def visualize_all_cam_methods(img_path: str, model: DRClassifier, config: Config):
    """Visualize all 5 CAM methods for a single image."""
    
    # Load image
    print(f"\nLoading image: {img_path}")
    img_pil = load_image_safe(img_path)
    
    if img_pil is None:
        print(f"❌ Failed to load image: {img_path}")
        return
    
    # Preprocess
    print("Preprocessing image...")
    img_array = preprocess_image(img_pil, target_size=config.IMG_SIZE)
    
    if img_array is None:
        print("❌ Failed to preprocess image")
        return
    
    # Get prediction
    predicted_class, probs = model.predict(img_array)
    class_name = config.CLASS_NAMES[predicted_class]
    confidence = probs[predicted_class]
    
    print(f"\n✅ Prediction: {class_name} (confidence: {confidence:.2%})")
    
    # Create subplots for all CAM methods
    n_methods = len(config.CAM_METHODS)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Add original image
    axes[0].imshow(img_pil)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Compute and display each CAM method
    print("\nComputing CAM for each method...")
    for idx, method in enumerate(config.CAM_METHODS, start=1):
        print(f"  Processing {method}...")
        
        try:
            # Compute CAM
            heatmap = model.compute_cam(img_array, method=method)
            
            if heatmap is None:
                axes[idx].text(
                    0.5, 0.5,
                    f"CAM computation\nfailed for {method}",
                    ha='center', va='center',
                    transform=axes[idx].transAxes
                )
                axes[idx].axis('off')
                continue
            
            # Overlay CAM on original image
            overlay_img = model.overlay_cam(img_pil, heatmap, alpha=0.5)
            
            axes[idx].imshow(overlay_img)
            axes[idx].set_title(f"{method}\nOverlay", fontsize=12, fontweight='bold')
            axes[idx].axis('off')
            
        except Exception as e:
            print(f"    ⚠️  Error with {method}: {e}")
            axes[idx].text(
                0.5, 0.5,
                f"Error: {str(e)[:50]}",
                ha='center', va='center',
                transform=axes[idx].transAxes,
                fontsize=10
            )
            axes[idx].axis('off')
    
    # Main title
    fig.suptitle(
        f"CAM Explainability - All 5 Methods\n"
        f"Prediction: {class_name} (Confidence: {confidence:.2%})",
        fontsize=16, fontweight='bold', y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def create_heatmap_comparison(img_path: str, model: DRClassifier, config: Config):
    """Create a detailed comparison of pure heatmaps."""
    
    # Load and preprocess
    img_pil = load_image_safe(img_path)
    if img_pil is None:
        print(f"❌ Failed to load image: {img_path}")
        return
    
    img_array = preprocess_image(img_pil, target_size=config.IMG_SIZE)
    if img_array is None:
        print("❌ Failed to preprocess image")
        return
    
    # Create figure for heatmap comparison
    fig, axes = plt.subplots(1, len(config.CAM_METHODS), figsize=(20, 4))
    
    print("\nGenerating heatmap comparisons...")
    for ax, method in zip(axes, config.CAM_METHODS):
        try:
            heatmap = model.compute_cam(img_array, method=method)
            
            if heatmap is None:
                ax.text(0.5, 0.5, f"Failed:\n{method}", ha='center', va='center')
                continue
            
            im = ax.imshow(heatmap, cmap='jet')
            ax.set_title(method, fontsize=12, fontweight='bold')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
        except Exception as e:
            print(f"  ⚠️  {method}: {e}")
            ax.axis('off')
    
    fig.suptitle("Heatmap Comparison - All CAM Methods", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Visualize all CAM methods for DR classification"
    )
    parser.add_argument(
        'image',
        nargs='?',
        default=None,
        help='Path to retinal fundus image'
    )
    parser.add_argument(
        '--random',
        action='store_true',
        help='Use randomly generated dummy image instead'
    )
    parser.add_argument(
        '--heatmap-only',
        action='store_true',
        help='Show only heatmap comparison (faster)'
    )
    
    args = parser.parse_args()
    
    # Load configuration and model
    print("=" * 60)
    print("CAM VISUALIZATION DEMO")
    print("=" * 60)
    
    try:
        print("\nInitializing model...")
        config = Config()
        model = DRClassifier(config.MODEL_PATH)
        print("✅ Model loaded")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return 1
    
    # Get or create image
    if args.random:
        img_pil = create_dummy_image()
        img_path = "dummy_retinal_image"
    elif args.image:
        img_path = args.image
    else:
        print("\n❌ Please provide an image path or use --random flag")
        print("   Usage: python examples_cam_visualization.py /path/to/image.jpg")
        print("   Or:    python examples_cam_visualization.py --random")
        return 1
    
    # Visualize
    try:
        if args.heatmap_only:
            fig = create_heatmap_comparison(img_path, model, config)
        else:
            fig = visualize_all_cam_methods(img_path, model, config)
        
        if fig:
            # Save figure
            output_path = Path("outputs") / "cam_visualization.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n✅ Visualization saved: {output_path}")
            
            # Show plot
            plt.show()
    
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
