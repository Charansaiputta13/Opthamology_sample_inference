"""
Configuration module for Diabetic Retinopathy Classification System.
Centralizes all paths, model settings, thresholds, and constants.
Supports environment variable overrides via .env file.
"""

import os
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()


class Config:
    """
    Centralized configuration for DR classification system (PyTorch backend).

    Attributes:
        BASE_DIR: Project root directory
        MODEL_PATH: Path to trained model file (.pth)
        DATA_DIR: Directory containing image data
        OUTPUT_DIR: Directory for outputs (logs, predictions, evaluation)
        IMG_SIZE: Input image size for model (width, height)
        CLASS_NAMES: List of DR classification categories
        CONFIDENCE_THRESHOLD: Minimum confidence for "reliable" predictions
        BLUR_THRESHOLD: Laplacian variance threshold for blur detection
        MIN_IMAGE_SIZE: Minimum acceptable image resolution
        BATCH_SIZE: Batch size for inference
        LOG_LEVEL: Logging level
        CAM_METHODS: Available CAM explainability methods
        DEFAULT_CAM_METHOD: Default CAM method used in UI
    """

    # =====================
    # DIRECTORY PATHS
    # =====================
    BASE_DIR: Path = Path(__file__).parent.parent.resolve()

    # Model path (can be overridden by MODEL_PATH env var)
    MODEL_PATH: Path = Path(os.getenv(
        "MODEL_PATH",
        str(BASE_DIR / "pretrained" / "dr_mobilenetv2_5class.pth")
    ))

    # Data directories
    DATA_DIR: Path = Path(os.getenv(
        "DATA_DIR",
        str(BASE_DIR / "data" / "images")
    ))

    # Output directories
    OUTPUT_DIR: Path = Path(os.getenv(
        "OUTPUT_DIR",
        str(BASE_DIR / "outputs")
    ))

    LOG_DIR: Path = OUTPUT_DIR / "logs"
    PREDICTION_DIR: Path = OUTPUT_DIR / "predictions"
    EVAL_DIR: Path = OUTPUT_DIR / "eval"

    # =====================
    # MODEL SETTINGS
    # =====================
    IMG_SIZE: Tuple[int, int] = (224, 224)  # MobileNetV2 input size

    CLASS_NAMES: List[str] = [
        "No DR",
        "Mild",
        "Moderate",
        "Severe",
        "PDR"
    ]

    NUM_CLASSES: int = len(CLASS_NAMES)

    # =====================
    # INFERENCE THRESHOLDS
    # =====================
    CONFIDENCE_THRESHOLD: float = float(os.getenv(
        "CONFIDENCE_THRESHOLD",
        "0.6"
    ))

    BLUR_THRESHOLD: float = float(os.getenv(
        "BLUR_THRESHOLD",
        "100.0"
    ))

    MIN_IMAGE_SIZE: int = int(os.getenv(
        "MIN_IMAGE_SIZE",
        "256"
    ))

    # =====================
    # BATCH PROCESSING
    # =====================
    BATCH_SIZE: int = int(os.getenv(
        "BATCH_SIZE",
        "16"
    ))

    # =====================
    # LOGGING
    # =====================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # =====================
    # CAM EXPLAINABILITY SETTINGS
    # =====================
    # Available Class Activation Mapping methods
    CAM_METHODS: List[str] = [
        "GradCAM",
        "GradCAM++",
        "ScoreCAM",
        "EigenCAM",
        "LayerCAM",
    ]
    DEFAULT_CAM_METHOD: str = "GradCAM"

    # Overlay transparency for CAM visualization
    CAM_ALPHA: float = 0.4

    # =====================
    # VALID IMAGE EXTENSIONS
    # =====================
    VALID_IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

    def __init__(self):
        """Initialize config and create necessary directories."""
        self._create_directories()
        self._validate_paths()

    def _create_directories(self) -> None:
        """Create output directories if they don't exist."""
        for directory in [self.OUTPUT_DIR, self.LOG_DIR, self.PREDICTION_DIR, self.EVAL_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

    def _validate_paths(self) -> None:
        """Validate critical paths exist."""
        if not self.MODEL_PATH.exists():
            raise FileNotFoundError(
                f"❌ Model file not found at: {self.MODEL_PATH}\n"
                f"Please ensure the trained .pth model is present or set MODEL_PATH environment variable."
            )

    def __repr__(self) -> str:
        """String representation of config."""
        return (
            f"Config(\n"
            f"  MODEL_PATH={self.MODEL_PATH},\n"
            f"  DATA_DIR={self.DATA_DIR},\n"
            f"  OUTPUT_DIR={self.OUTPUT_DIR},\n"
            f"  IMG_SIZE={self.IMG_SIZE},\n"
            f"  CONFIDENCE_THRESHOLD={self.CONFIDENCE_THRESHOLD},\n"
            f"  BATCH_SIZE={self.BATCH_SIZE},\n"
            f"  CAM_METHODS={self.CAM_METHODS}\n"
            f")"
        )


# Singleton instance
_config_instance = None

def get_config() -> Config:
    """
    Get singleton config instance.

    Returns:
        Config: Global configuration object
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance
