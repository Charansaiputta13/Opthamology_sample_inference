"""
Image preprocessing and quality check utilities.
Handles loading, resizing, normalization, and quality validation.
PyTorch-compatible: uses ImageNet normalization instead of tf.keras preprocessing.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, List, Optional, Dict
import torch

# ImageNet normalization constants (standard for MobileNetV2 in PyTorch/torchvision)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image_safe(path: Union[str, Path]) -> Optional[Image.Image]:
    """
    Safely load an image from file with error handling.

    Args:
        path: Path to image file

    Returns:
        PIL Image object or None if loading fails
    """
    try:
        path = Path(path)
        if not path.exists():
            return None

        img = Image.open(path).convert("RGB")
        return img

    except Exception as e:
        print(f"[WARNING] Failed to load image {path}: {e}")
        return None


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.

    This is highly effective for enhancing details in retinal images.
    Applies CLAHE to each channel of an RGB image in LAB space for better results.

    Args:
        img: Input image as numpy array (RGB)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization

    Returns:
        Enhanced image as numpy array
    """
    try:
        # Convert RGB to LAB color space
        lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)

        # Merge back and convert to RGB
        limg = cv2.merge((cl, a, b))
        final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

        return final_img
    except Exception as e:
        print(f"[WARNING] CLAHE processing failed: {e}")
        return img


def preprocess_image(
    img: Union[Image.Image, np.ndarray, str, Path],
    target_size: Tuple[int, int] = (224, 224),
    use_clahe: bool = False
) -> Optional[np.ndarray]:
    """
    Preprocess image for MobileNetV2 model inference (PyTorch).

    Preprocessing pipeline:
    1. Resize to target size
    2. Convert to float32 in [0, 1]
    3. Apply ImageNet normalization: (x - mean) / std

    Args:
        img: Input image (PIL Image, numpy array, or file path)
        target_size: Target size (width, height)
        use_clahe: Whether to apply CLAHE enhancement

    Returns:
        Preprocessed numpy array of shape (H, W, 3), ImageNet-normalized.
        Returns None if processing fails.
    """
    try:
        # Handle different input types
        if isinstance(img, (str, Path)):
            img = load_image_safe(img)
            if img is None:
                return None

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))

        if not isinstance(img, Image.Image):
            return None

        # Resize
        img_resized = img.resize(target_size, Image.LANCZOS)

        # Convert to float32 array in [0, 1]
        img_array = np.array(img_resized, dtype=np.float32) / 255.0

        # Apply CLAHE if requested (operates on uint8, then convert back)
        if use_clahe:
            img_uint8 = (img_array * 255).astype(np.uint8)
            img_uint8 = apply_clahe(img_uint8)
            img_array = img_uint8.astype(np.float32) / 255.0

        # Apply ImageNet normalization: (pixel - mean) / std
        img_normalized = (img_array - IMAGENET_MEAN) / IMAGENET_STD

        return img_normalized.astype(np.float32)

    except Exception as e:
        print(f"[WARNING] Failed to preprocess image: {e}")
        return None


def to_tensor(img_array: np.ndarray) -> torch.Tensor:
    """
    Convert a preprocessed numpy image (H, W, C) to a PyTorch tensor (1, C, H, W).

    Args:
        img_array: Preprocessed numpy array of shape (H, W, 3)

    Returns:
        PyTorch tensor of shape (1, 3, H, W) ready for model inference
    """
    # HWC -> CHW -> add batch dimension
    tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    return tensor.float()


def compute_blur_score(img: Image.Image) -> float:
    """
    Compute blur score using Laplacian variance.

    Lower scores indicate more blur.
    Typical thresholds:
    - < 100: Very blurry
    - 100-200: Somewhat blurry
    - > 200: Sharp

    Args:
        img: PIL Image

    Returns:
        Blur score (Laplacian variance)
    """
    try:
        img_gray = np.array(img.convert('L'))
        laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
        variance = laplacian.var()
        return float(variance)

    except Exception as e:
        print(f"[WARNING] Failed to compute blur score: {e}")
        return 0.0


def check_image_quality(
    img: Image.Image,
    min_size: int = 256,
    blur_threshold: float = 100.0
) -> Tuple[List[str], Dict[str, float]]:
    """
    Check image quality for potential issues.

    Args:
        img: PIL Image
        min_size: Minimum acceptable dimension (width or height)
        blur_threshold: Laplacian variance threshold for blur detection

    Returns:
        Tuple of:
            - List of quality issue descriptions
            - Dictionary of quality metrics
    """
    issues = []
    metrics = {}

    # Check resolution
    width, height = img.size
    metrics['width'] = width
    metrics['height'] = height

    if width < min_size or height < min_size:
        issues.append(f"Low resolution ({width}x{height})")

    # Check blur
    blur_score = compute_blur_score(img)
    metrics['blur_score'] = blur_score

    if blur_score < blur_threshold:
        issues.append(f"Image appears blurry (score: {blur_score:.1f})")

    # Check aspect ratio
    aspect_ratio = width / height if height > 0 else 1.0
    metrics['aspect_ratio'] = aspect_ratio

    if aspect_ratio > 3.0 or aspect_ratio < 0.33:
        issues.append(f"Unusual aspect ratio ({aspect_ratio:.2f})")

    # Check brightness
    img_array = np.array(img.convert('L'))
    mean_brightness = img_array.mean()
    metrics['mean_brightness'] = mean_brightness

    if mean_brightness < 30:
        issues.append("Image is very dark")
    elif mean_brightness > 225:
        issues.append("Image is very bright")

    return issues, metrics


def preprocess_batch(
    images: List[Union[Image.Image, np.ndarray]],
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    """
    Preprocess a batch of images efficiently.

    Args:
        images: List of PIL Images or numpy arrays
        target_size: Target size for all images

    Returns:
        Numpy array of shape (N, H, W, 3) — ImageNet normalized
    """
    preprocessed = []

    for img in images:
        processed = preprocess_image(img, target_size)
        if processed is not None:
            preprocessed.append(processed)

    if len(preprocessed) == 0:
        return np.array([])

    return np.stack(preprocessed, axis=0)


def normalize_to_uint8(img_array: np.ndarray) -> np.ndarray:
    """
    Normalize a float array to uint8 [0, 255] for visualization.
    Handles both ImageNet-normalized (may be negative) and [0,1] arrays.

    Args:
        img_array: Float array

    Returns:
        Uint8 array in [0, 255] range
    """
    min_val = img_array.min()
    max_val = img_array.max()
    range_val = max_val - min_val

    if range_val < 1e-8:
        return np.zeros_like(img_array, dtype=np.uint8)

    img_normalized = (img_array - min_val) / range_val * 255.0
    return np.clip(img_normalized, 0, 255).astype(np.uint8)


def apply_test_time_augmentation(
    img: Image.Image,
    augmentations: List[str] = ['original', 'flip_horizontal']
) -> List[Image.Image]:
    """
    Apply test-time augmentation to an image.

    Args:
        img: PIL Image
        augmentations: List of augmentation types
                      Options: 'original', 'flip_horizontal', 'flip_vertical',
                               'rotate_90', 'rotate_180', 'rotate_270'

    Returns:
        List of augmented PIL Images
    """
    augmented = []

    for aug in augmentations:
        if aug == 'original':
            augmented.append(img.copy())
        elif aug == 'flip_horizontal':
            augmented.append(img.transpose(Image.FLIP_LEFT_RIGHT))
        elif aug == 'flip_vertical':
            augmented.append(img.transpose(Image.FLIP_TOP_BOTTOM))
        elif aug == 'rotate_90':
            augmented.append(img.transpose(Image.ROTATE_90))
        elif aug == 'rotate_180':
            augmented.append(img.transpose(Image.ROTATE_180))
        elif aug == 'rotate_270':
            augmented.append(img.transpose(Image.ROTATE_270))

    return augmented
