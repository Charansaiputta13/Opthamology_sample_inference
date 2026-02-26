"""
Inference orchestration for DR classification.
Handles single image and batch inference with quality checks and error handling.
"""

import io
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

from .config import Config
from .model import DRClassifier
from .preprocessing import (
    load_image_safe,
    preprocess_image,
    check_image_quality,
    apply_test_time_augmentation
)
from .utils.validation import get_valid_images_from_zip, sanitize_filename


def predict_single_image(
    img_path: str,
    model: DRClassifier,
    config: Config,
    gradcam: bool = False
) -> Dict[str, Any]:
    """
    Predict DR classification for a single image with quality checks.
    
    Args:
        img_path: Path to image file
        model: DRClassifier instance
        config: Configuration object
        gradcam: Whether to compute Grad-CAM heatmap
        
    Returns:
        Dictionary containing:
            - success: bool
            - filename: str
            - predicted_class: int
            - class_name: str
            - confidence: float
            - probabilities: dict
            - quality_issues: list
            - quality_metrics: dict
            - needs_review: bool
            - gradcam_heatmap: np.ndarray (if gradcam=True)
            - error: str (if failed)
    """
    result = {
        'success': False,
        'filename': Path(img_path).name,
        'error': None
    }
    
    try:
        # Load image
        img_pil = load_image_safe(img_path)
        if img_pil is None:
            result['error'] = "Failed to load image"
            return result
        
        # Quality checks
        quality_issues, quality_metrics = check_image_quality(
            img_pil,
            min_size=config.MIN_IMAGE_SIZE,
            blur_threshold=config.BLUR_THRESHOLD
        )
        
        result['quality_issues'] = quality_issues
        result['quality_metrics'] = quality_metrics
        
        # Preprocess
        img_preprocessed = preprocess_image(img_pil, target_size=config.IMG_SIZE)
        if img_preprocessed is None:
            result['error'] = "Failed to preprocess image"
            return result
        
        # Predict
        predicted_class, probabilities = model.predict(img_preprocessed)
        
        result['success'] = True
        result['predicted_class'] = predicted_class
        result['class_name'] = config.CLASS_NAMES[predicted_class]
        result['confidence'] = float(probabilities[predicted_class])
        result['probabilities'] = {
            config.CLASS_NAMES[i]: float(probabilities[i])
            for i in range(len(config.CLASS_NAMES))
        }
        
        # Determine if needs review
        result['needs_review'] = (
            result['confidence'] < config.CONFIDENCE_THRESHOLD
            or len(quality_issues) > 0
        )
        
        # Grad-CAM if requested
        if gradcam:
            heatmap = model.compute_gradcam(img_preprocessed, target_class=predicted_class)
            result['gradcam_heatmap'] = heatmap
        
        return result
    
    except Exception as e:
        result['error'] = str(e)
        return result


def predict_with_tta(
    img: Image.Image,
    model: DRClassifier,
    config: Config,
    augmentations: List[str] = ['original', 'flip_horizontal']
) -> np.ndarray:
    """
    Predict with Test-Time Augmentation (TTA) for improved robustness.
    
    Args:
        img: PIL Image
        model: DRClassifier instance
        config: Configuration object
        augmentations: List of augmentation types
        
    Returns:
        Averaged probabilities across augmentations
    """
    # Generate augmented images
    augmented_images = apply_test_time_augmentation(img, augmentations)
    
    # Preprocess all
    preprocessed = []
    for aug_img in augmented_images:
        proc_img = preprocess_image(aug_img, target_size=config.IMG_SIZE)
        if proc_img is not None:
            preprocessed.append(proc_img)
    
    if len(preprocessed) == 0:
        return np.zeros(config.NUM_CLASSES)
    
    # Batch predict
    batch = np.stack(preprocessed, axis=0)
    all_probs = model.predict_batch(batch, batch_size=len(batch))
    
    # Average probabilities
    avg_probs = np.mean(all_probs, axis=0)
    
    return avg_probs


def predict_batch_from_zip(
    zip_path: str,
    model: DRClassifier,
    config: Config,
    progress_callback: Optional[Callable[[float], None]] = None
) -> pd.DataFrame:
    """
    Run inference on all images in a ZIP file with chunked batch processing.
    
    This function is memory-efficient:
    - Streams ZIP contents (no full extraction)
    - Processes images in batches
    - Skips corrupt images gracefully
    
    Args:
        zip_path: Path to ZIP file
        model: DRClassifier instance
        config: Configuration object
        progress_callback: Optional callback function(progress: float [0-1])
        
    Returns:
        DataFrame with columns:
            - filename
            - predicted_class
            - class_name
            - confidence
            - prob_class_0, prob_class_1, ... (per-class probabilities)
            - blur_score
            - quality_issues
            - needs_review
            - error (if any)
    """
    results = []
    
    # Get list of images in ZIP
    image_names = get_valid_images_from_zip(zip_path)
    total = len(image_names)
    
    if total == 0:
        return pd.DataFrame()
    
    processed = 0
    batch_images = []
    batch_filenames = []
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        for idx, img_name in enumerate(image_names):
            try:
                # Read image from ZIP
                with z.open(img_name) as f:
                    img_bytes = f.read()
                    img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                
                # Quality check
                quality_issues, quality_metrics = check_image_quality(
                    img_pil,
                    min_size=config.MIN_IMAGE_SIZE,
                    blur_threshold=config.BLUR_THRESHOLD
                )
                
                # Preprocess
                img_preprocessed = preprocess_image(img_pil, target_size=config.IMG_SIZE)
                
                if img_preprocessed is not None:
                    batch_images.append(img_preprocessed)
                    batch_filenames.append({
                        'filename': sanitize_filename(img_name),
                        'quality_issues': quality_issues,
                        'blur_score': quality_metrics.get('blur_score', 0.0)
                    })
                else:
                    # Failed to preprocess
                    results.append({
                        'filename': sanitize_filename(img_name),
                        'predicted_class': -1,
                        'class_name': 'ERROR',
                        'confidence': 0.0,
                        'blur_score': 0.0,
                        'quality_issues': 'Preprocessing failed',
                        'needs_review': True,
                        'error': 'Preprocessing failed'
                    })
                    processed += 1
                    if progress_callback:
                        progress_callback(processed / total)
                    continue
            
            except Exception as e:
                # Failed to load image
                results.append({
                    'filename': sanitize_filename(img_name),
                    'predicted_class': -1,
                    'class_name': 'ERROR',
                    'confidence': 0.0,
                    'blur_score': 0.0,
                    'quality_issues': f'Load error: {str(e)}',
                    'needs_review': True,
                    'error': str(e)
                })
                processed += 1
                if progress_callback:
                    progress_callback(processed / total)
                continue
            
            # Process batch when full or at end
            if len(batch_images) >= config.BATCH_SIZE or idx == total - 1:
                if len(batch_images) > 0:
                    # Batch predict
                    batch_tensor = np.stack(batch_images, axis=0)
                    probs_batch = model.predict_batch(batch_tensor, batch_size=len(batch_tensor))
                    
                    # Parse results
                    for i, probs in enumerate(probs_batch):
                        pred_class = int(np.argmax(probs))
                        confidence = float(probs[pred_class])
                        
                        row = {
                            'filename': batch_filenames[i]['filename'],
                            'predicted_class': pred_class,
                            'class_name': config.CLASS_NAMES[pred_class],
                            'confidence': confidence,
                            'blur_score': batch_filenames[i]['blur_score'],
                            'quality_issues': ', '.join(batch_filenames[i]['quality_issues']) or 'OK',
                            'needs_review': (
                                confidence < config.CONFIDENCE_THRESHOLD
                                or len(batch_filenames[i]['quality_issues']) > 0
                            )
                        }
                        
                        # Add per-class probabilities
                        for j, class_name in enumerate(config.CLASS_NAMES):
                            row[f'prob_{class_name.replace(" ", "_")}'] = float(probs[j])
                        
                        results.append(row)
                    
                    processed += len(batch_images)
                    if progress_callback:
                        progress_callback(min(processed / total, 1.0))
                    
                    # Clear batch
                    batch_images = []
                    batch_filenames = []
    
    return pd.DataFrame(results)


def predict_batch_from_folder(
    folder_path: str,
    model: DRClassifier,
    config: Config,
    progress_callback: Optional[Callable[[float], None]] = None
) -> pd.DataFrame:
    """
    Run inference on all images in a folder.
    
    Args:
        folder_path: Path to folder containing images
        model: DRClassifier instance
        config: Configuration object
        progress_callback: Optional progress callback
        
    Returns:
        DataFrame with predictions (same format as predict_batch_from_zip)
    """
    results = []
    folder = Path(folder_path)
    
    # Get all image files
    image_files = []
    for ext in config.VALID_IMAGE_EXTENSIONS:
        image_files.extend(folder.glob(f'*{ext}'))
        image_files.extend(folder.glob(f'*{ext.upper()}'))
    
    total = len(image_files)
    if total == 0:
        return pd.DataFrame()
    
    processed = 0
    batch_images = []
    batch_filenames = []
    
    for idx, img_path in enumerate(image_files):
        try:
            # Load image
            img_pil = load_image_safe(img_path)
            if img_pil is None:
                raise ValueError("Failed to load image")
            
            # Quality check
            quality_issues, quality_metrics = check_image_quality(
                img_pil,
                min_size=config.MIN_IMAGE_SIZE,
                blur_threshold=config.BLUR_THRESHOLD
            )
            
            # Preprocess
            img_preprocessed = preprocess_image(img_pil, target_size=config.IMG_SIZE)
            
            if img_preprocessed is not None:
                batch_images.append(img_preprocessed)
                batch_filenames.append({
                    'filename': img_path.name,
                    'quality_issues': quality_issues,
                    'blur_score': quality_metrics.get('blur_score', 0.0)
                })
            else:
                results.append({
                    'filename': img_path.name,
                    'predicted_class': -1,
                    'class_name': 'ERROR',
                    'confidence': 0.0,
                    'blur_score': 0.0,
                    'quality_issues': 'Preprocessing failed',
                    'needs_review': True,
                    'error': 'Preprocessing failed'
                })
                processed += 1
                if progress_callback:
                    progress_callback(processed / total)
                continue
        
        except Exception as e:
            results.append({
                'filename': img_path.name,
                'predicted_class': -1,
                'class_name': 'ERROR',
                'confidence': 0.0,
                'blur_score': 0.0,
                'quality_issues': f'Error: {str(e)}',
                'needs_review': True,
                'error': str(e)
            })
            processed += 1
            if progress_callback:
                progress_callback(processed / total)
            continue
        
        # Process batch
        if len(batch_images) >= config.BATCH_SIZE or idx == total - 1:
            if len(batch_images) > 0:
                batch_tensor = np.stack(batch_images, axis=0)
                probs_batch = model.predict_batch(batch_tensor, batch_size=len(batch_tensor))
                
                for i, probs in enumerate(probs_batch):
                    pred_class = int(np.argmax(probs))
                    confidence = float(probs[pred_class])
                    
                    row = {
                        'filename': batch_filenames[i]['filename'],
                        'predicted_class': pred_class,
                        'class_name': config.CLASS_NAMES[pred_class],
                        'confidence': confidence,
                        'blur_score': batch_filenames[i]['blur_score'],
                        'quality_issues': ', '.join(batch_filenames[i]['quality_issues']) or 'OK',
                        'needs_review': (
                            confidence < config.CONFIDENCE_THRESHOLD
                            or len(batch_filenames[i]['quality_issues']) > 0
                        )
                    }
                    
                    for j, class_name in enumerate(config.CLASS_NAMES):
                        row[f'prob_{class_name.replace(" ", "_")}'] = float(probs[j])
                    
                    results.append(row)
                
                processed += len(batch_images)
                if progress_callback:
                    progress_callback(min(processed / total, 1.0))
                
                batch_images = []
                batch_filenames = []
    
    return pd.DataFrame(results)
