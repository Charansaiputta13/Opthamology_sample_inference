"""
Model management for DR classification.
Handles model loading, prediction, and Grad-CAM explainability.
"""

import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


class DRClassifier:
    """
    Diabetic Retinopathy classifier with Grad-CAM explainability.
    
    This class manages model loading, caching, prediction, and visualization.
    Uses singleton pattern for efficient model caching.
    """
    
    _instance = None
    _model = None
    _model_path = None
    
    def __new__(cls, model_path: str, *args, **kwargs):
        """Singleton pattern to cache model."""
        if cls._instance is None or str(cls._model_path) != str(model_path):
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_path: str, cache: bool = True):
        """
        Initialize DR Classifier.
        
        Args:
            model_path: Path to trained model (.h5 or SavedModel)
            cache: Whether to cache the loaded model
        """
        # Only load if not already loaded or path changed
        if DRClassifier._model is None or str(DRClassifier._model_path) != str(model_path):
            self.model_path = Path(model_path)
            self._load_model()
            DRClassifier._model_path = model_path
        
        self.model = DRClassifier._model
        self.last_conv_layer_name = self._find_last_conv_layer()
    
    def _load_model(self) -> None:
        """Load TensorFlow model with error handling."""
        try:
            print(f"Loading model from: {self.model_path}")
            DRClassifier._model = tf.keras.models.load_model(str(self.model_path))
            print( " Model loaded successfully")
            print(f"   Input shape: {DRClassifier._model.input_shape}")
            print(f"   Output shape: {DRClassifier._model.output_shape}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")
    
    def _find_last_conv_layer(self) -> Optional[str]:
        """
        Find the last convolutional layer in the model.
        
        Returns:
            Name of last Conv2D layer or None if not found
        """
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        
        # No Conv2D found - might be in a nested model
        for layer in reversed(self.model.layers):
            if hasattr(layer, 'layers'):
                for sublayer in reversed(layer.layers):
                    if isinstance(sublayer, tf.keras.layers.Conv2D):
                        return sublayer.name
        
        print("[WARNING] No Conv2D layer found in model. Grad-CAM will not work.")
        return None
    
    def predict(self, img: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Predict class for a single preprocessed image.
        
        Args:
            img: Preprocessed image array of shape (H, W, 3)
            
        Returns:
            Tuple of (predicted_class_index, probabilities)
        """
        try:
            # Add batch dimension if needed
            if len(img.shape) == 3:
                img = np.expand_dims(img, axis=0)
            
            # Predict
            probabilities = self.model.predict(img, verbose=0)[0]
            predicted_class = int(np.argmax(probabilities))
            
            return predicted_class, probabilities
        
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
    
    def predict_batch(self, imgs: np.ndarray, batch_size: int = 16) -> np.ndarray:
        """
        Predict classes for a batch of preprocessed images.
        
        Args:
            imgs: Batch of preprocessed images, shape (N, H, W, 3)
            batch_size: Batch size for prediction
            
        Returns:
            Probabilities array of shape (N, num_classes)
        """
        try:
            return self.model.predict(imgs, batch_size=batch_size, verbose=0)
        except Exception as e:
            raise RuntimeError(f"Batch prediction failed: {e}")
    
    def compute_gradcam(
        self,
        img: np.ndarray,
        target_class: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Compute Grad-CAM heatmap for explainability.
        
        Args:
            img: Preprocessed image array of shape (H, W, 3) or (1, H, W, 3)
            target_class: Target class index (if None, uses predicted class)
            
        Returns:
            Heatmap array of shape (H, W) or None if Grad-CAM fails
        """
        if self.last_conv_layer_name is None:
            print("[WARNING] Cannot compute Grad-CAM: No Conv2D layer found")
            return None
        
        try:
            # Add batch dimension if needed
            if len(img.shape) == 3:
                img_batch = np.expand_dims(img, axis=0)
            else:
                img_batch = img
            
            # Create Grad-CAM model
            grad_model = tf.keras.models.Model(
                inputs=self.model.inputs,
                outputs=[
                    self.model.get_layer(self.last_conv_layer_name).output,
                    self.model.output
                ]
            )
            
            # Compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_batch)
                
                # Handle list output
                if isinstance(predictions, (list, tuple)):
                    predictions = predictions[0]
                
                # Use predicted class if target not specified
                if target_class is None:
                    target_class = tf.argmax(predictions[0])
                
                # Loss for target class
                loss = predictions[:, target_class]
            
            # Compute gradients
            grads = tape.gradient(loss, conv_outputs)
            
            # Global average pooling on gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight conv outputs by gradients
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
            
            # Normalize heatmap [0, 1]
            heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
            
            return heatmap.numpy()
        
        except Exception as e:
            print(f"[WARNING] Grad-CAM computation failed: {e}")
            return None
    
    def overlay_gradcam(
        self,
        original_img: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: str = 'jet'
    ) -> Image.Image:
        """
        Overlay Grad-CAM heatmap on original image.
        
        Args:
            original_img: Original PIL Image
            heatmap: Grad-CAM heatmap array (H, W)
            alpha: Overlay transparency (0=invisible, 1=opaque)
            colormap: Matplotlib colormap name
            
        Returns:
            PIL Image with heatmap overlay
        """
        try:
            # Resize heatmap to match original image size
            heatmap_resized = np.array(
                Image.fromarray(np.uint8(255 * heatmap)).resize(
                    original_img.size,
                    Image.LANCZOS
                )
            )
            
            # Apply colormap
            cmap = plt.get_cmap(colormap)
            heatmap_colored = cmap(heatmap_resized / 255.0)
            heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
            
            # Convert original to array
            original_array = np.array(original_img)
            
            # Blend
            overlayed = (alpha * heatmap_colored + (1 - alpha) * original_array).astype(np.uint8)
            
            return Image.fromarray(overlayed)
        
        except Exception as e:
            print(f"[WARNING] Heatmap overlay failed: {e}")
            return original_img
    
    def create_gradcam_figure(
        self,
        original_img: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.4
    ) -> plt.Figure:
        """
        Create matplotlib figure with Grad-CAM visualization.
        
        Args:
            original_img: Original PIL Image
            heatmap: Grad-CAM heatmap
            alpha: Overlay transparency
            
        Returns:
            Matplotlib figure
        """
        overlayed = self.overlay_gradcam(original_img, heatmap, alpha)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(overlayed)
        ax.axis('off')
        ax.set_title('Grad-CAM Explainability', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def get_model_info(self) -> dict:
        """
        Get model information.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            'model_path': str(self.model_path),
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'last_conv_layer': self.last_conv_layer_name,
            'total_params': self.model.count_params(),
            'trainable_params': sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        }
