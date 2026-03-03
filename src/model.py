"""
Model management for DR classification (PyTorch backend).
Handles model loading, prediction, and CAM explainability with 5 methods:
  - GradCAM, GradCAM++, ScoreCAM, EigenCAM, LayerCAM
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import torchvision.models as tv_models

# pytorch-grad-cam library
from pytorch_grad_cam import (
    GradCAM,
    GradCAMPlusPlus,
    ScoreCAM,
    EigenCAM,
    LayerCAM,
)
from pytorch_grad_cam.utils.image import show_cam_on_image

# ─── CAM method registry ──────────────────────────────────────────────────────
_CAM_CLASSES = {
    "GradCAM":    GradCAM,
    "GradCAM++":  GradCAMPlusPlus,
    "ScoreCAM":   ScoreCAM,
    "EigenCAM":   EigenCAM,
    "LayerCAM":   LayerCAM,
}


class DRClassifier:
    """
    Diabetic Retinopathy classifier (PyTorch) with multi-method CAM explainability.

    Supports GradCAM, GradCAM++, ScoreCAM, EigenCAM, and LayerCAM.
    Uses singleton pattern for efficient model caching.
    """

    _instance = None
    _model: Optional[nn.Module] = None
    _model_path = None

    def __new__(cls, model_path: str, *args, **kwargs):
        """Singleton: re-create only if the model path changes."""
        if cls._instance is None or str(cls._model_path) != str(model_path):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_path: str, num_classes: int = 5):
        """
        Initialize DR Classifier.

        Args:
            model_path: Path to trained .pth model file
            num_classes: Number of output classes (default 5 for DR grading)
        """
        if DRClassifier._model is None or str(DRClassifier._model_path) != str(model_path):
            self.model_path = Path(model_path)
            self.num_classes = num_classes
            self.device = torch.device("cpu")  # CPU-first; GPU transparent
            self._load_model()
            DRClassifier._model_path = model_path

        self.model = DRClassifier._model

        # Target layer for CAM: last conv block of MobileNetV2
        # features[-1] is the last ConvBNActivation block
        self._target_layers: List[nn.Module] = [self.model.features[-1]]

    # ── Model Loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """
        Load MobileNetV2 from .pth checkpoint with auto-format detection.

        Supports:
          - state_dict (OrderedDict) checkpoints
          - Full serialized model objects
          - Checkpoint dicts with 'state_dict' or 'model_state_dict' keys
        """
        try:
            print(f"Loading model from: {self.model_path}")
            checkpoint = torch.load(str(self.model_path), map_location=self.device, weights_only=False)

            # Build fresh MobileNetV2 with correct head
            model = tv_models.mobilenet_v2(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, self.num_classes)

            # Determine how weights are stored
            if isinstance(checkpoint, dict):
                # Check for nested state dict keys
                if "state_dict" in checkpoint:
                    state = checkpoint["state_dict"]
                elif "model_state_dict" in checkpoint:
                    state = checkpoint["model_state_dict"]
                else:
                    # Assume it IS the state_dict
                    state = checkpoint

                # Strip any 'module.' prefix (from DataParallel)
                state = {k.replace("module.", ""): v for k, v in state.items()}
                model.load_state_dict(state, strict=False)

            else:
                # Full model object — use directly
                model = checkpoint

            model.to(self.device)
            model.eval()

            DRClassifier._model = model
            print("✅ Model loaded successfully (PyTorch)")
            print(f"   Architecture : MobileNetV2")
            print(f"   Classes      : {self.num_classes}")
            print(f"   Device       : {self.device}")

        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, img: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Predict DR class for a single preprocessed image.

        Args:
            img: ImageNet-normalized numpy array of shape (H, W, 3)

        Returns:
            (predicted_class_index, probabilities_array)
        """
        try:
            from .preprocessing import to_tensor
            tensor = to_tensor(img).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            predicted_class = int(np.argmax(probs))
            return predicted_class, probs

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")

    def predict_batch(self, imgs: np.ndarray, batch_size: int = 16) -> np.ndarray:
        """
        Predict classes for a batch of images.

        Args:
            imgs: Batch of ImageNet-normalized arrays, shape (N, H, W, 3)
            batch_size: Batch size for inference

        Returns:
            Probabilities array of shape (N, num_classes)
        """
        try:
            all_probs = []
            n = imgs.shape[0]

            for start in range(0, n, batch_size):
                chunk = imgs[start:start + batch_size]
                # (N, H, W, C) → (N, C, H, W)
                tensor = torch.from_numpy(chunk).permute(0, 3, 1, 2).float().to(self.device)
                with torch.no_grad():
                    logits = self.model(tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)

            return np.concatenate(all_probs, axis=0)

        except Exception as e:
            raise RuntimeError(f"Batch prediction failed: {e}")

    # ── CAM Explainability ────────────────────────────────────────────────────

    def compute_cam(
        self,
        img: np.ndarray,
        method: str = "GradCAM",
        target_class: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Compute a CAM heatmap using the specified method.

        Args:
            img: ImageNet-normalized numpy array of shape (H, W, 3)
            method: One of 'GradCAM', 'GradCAM++', 'ScoreCAM', 'EigenCAM', 'LayerCAM'
            target_class: Target class index (None → uses predicted class)

        Returns:
            Heatmap array of shape (H, W) with values in [0, 1], or None on failure
        """
        if method not in _CAM_CLASSES:
            print(f"[WARNING] Unknown CAM method '{method}'. Falling back to GradCAM.")
            method = "GradCAM"

        try:
            from .preprocessing import to_tensor
            tensor = to_tensor(img).to(self.device)

            cam_class = _CAM_CLASSES[method]
            targets = None  # None → pytorch-grad-cam uses the top predicted class
            if target_class is not None:
                from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
                targets = [ClassifierOutputTarget(target_class)]

            with cam_class(model=self.model, target_layers=self._target_layers) as cam_obj:
                heatmap = cam_obj(input_tensor=tensor, targets=targets)

            # heatmap shape: (1, H, W) — squeeze to (H, W)
            return heatmap[0]

        except Exception as e:
            print(f"[WARNING] CAM ({method}) computation failed: {e}")
            return None

    # ── Visualization ─────────────────────────────────────────────────────────

    def overlay_cam(
        self,
        original_img: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.4,
    ) -> Image.Image:
        """
        Overlay a CAM heatmap on the original image using show_cam_on_image.

        Args:
            original_img: Original PIL Image (any size)
            heatmap: CAM heatmap array (H, W) in [0, 1]
            alpha: Heatmap opacity (0 invisible → 1 fully opaque)

        Returns:
            PIL Image with colored CAM overlay
        """
        try:
            # Resize PIL to match heatmap size
            h, w = heatmap.shape
            img_resized = original_img.resize((w, h), Image.LANCZOS)
            rgb_img = np.array(img_resized, dtype=np.float32) / 255.0

            # show_cam_on_image expects float32 [0,1] RGB + float32 heatmap [0,1]
            overlayed = show_cam_on_image(rgb_img, heatmap, use_rgb=True, image_weight=1 - alpha)
            return Image.fromarray(overlayed)

        except Exception as e:
            print(f"[WARNING] CAM overlay failed: {e}")
            return original_img

    def create_cam_figure(
        self,
        original_img: Image.Image,
        heatmap: np.ndarray,
        method: str = "GradCAM",
        alpha: float = 0.4,
    ) -> plt.Figure:
        """
        Create a 3-panel matplotlib figure: Original | Heatmap | Overlay.

        Args:
            original_img: Original PIL Image
            heatmap: CAM heatmap array (H, W)
            method: CAM method name (used for title)
            alpha: Overlay transparency

        Returns:
            Matplotlib Figure
        """
        overlayed = self.overlay_cam(original_img, heatmap, alpha)

        # Resize heatmap to display size
        h, w = heatmap.shape
        heatmap_img = Image.fromarray(np.uint8(heatmap * 255)).resize(
            original_img.size, Image.LANCZOS
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original_img)
        axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
        axes[0].axis('off')

        im = axes[1].imshow(np.array(heatmap_img), cmap='jet')
        axes[1].set_title(f"{method} Heatmap", fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].imshow(overlayed)
        axes[2].set_title(f"{method} Overlay", fontsize=12, fontweight='bold')
        axes[2].axis('off')

        plt.suptitle(f"CAM Explainability — {method}", fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()

        return fig

    # ── backwards compat alias ────────────────────────────────────────────────

    def compute_gradcam(self, img: np.ndarray, target_class: Optional[int] = None) -> Optional[np.ndarray]:
        """Backwards-compatible alias → compute_cam with GradCAM."""
        return self.compute_cam(img, method="GradCAM", target_class=target_class)

    def overlay_gradcam(self, original_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.4, **kwargs) -> Image.Image:
        """Backwards-compatible alias → overlay_cam."""
        return self.overlay_cam(original_img, heatmap, alpha)

    def create_gradcam_figure(self, original_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> plt.Figure:
        """Backwards-compatible alias → create_cam_figure."""
        return self.create_cam_figure(original_img, heatmap, method="GradCAM", alpha=alpha)

    # ── Model info ────────────────────────────────────────────────────────────

    def get_model_info(self) -> dict:
        """
        Get model information.

        Returns:
            Dictionary with model metadata
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            'model_path': str(self.model_path),
            'architecture': 'MobileNetV2',
            'framework': 'PyTorch',
            'device': str(self.device),
            'num_classes': self.num_classes,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'cam_methods': list(_CAM_CLASSES.keys()),
        }
