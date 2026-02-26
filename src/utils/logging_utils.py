"""
Logging utilities for DR classification system.
Provides structured logging with file rotation and console output.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import List, Optional
from datetime import datetime


def setup_logging(
    log_dir: Path,
    log_level: str = "INFO",
    log_file: str = "dr_inference.log",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Configure logging with file rotation and console output.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Name of log file
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("dr_classifier")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # File handler with rotation
    log_path = log_dir / log_file
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def log_prediction(
    logger: logging.Logger,
    filename: str,
    prediction: int,
    class_name: str,
    confidence: float,
    quality_issues: List[str],
    needs_review: bool
) -> None:
    """
    Log a prediction with structured format.
    
    Args:
        logger: Logger instance
        filename: Image filename
        prediction: Predicted class index
        class_name: Predicted class name
        confidence: Prediction confidence (0-1)
        quality_issues: List of image quality issues
        needs_review: Whether prediction needs human review
    """
    quality_str = ", ".join(quality_issues) if quality_issues else "OK"
    review_flag = "[NEEDS REVIEW]" if needs_review else "[OK]"
    
    logger.info(
        f"PREDICTION | {filename} | Class: {prediction} ({class_name}) | "
        f"Confidence: {confidence:.3f} | Quality: {quality_str} | {review_flag}"
    )


def log_error(
    logger: logging.Logger,
    filename: str,
    error: Exception,
    context: str = ""
) -> None:
    """
    Log an error with traceback.
    
    Args:
        logger: Logger instance
        filename: Image filename that caused error
        error: Exception object
        context: Additional context about the error
    """
    context_str = f" ({context})" if context else ""
    logger.error(
        f"ERROR{context_str} | {filename} | {type(error).__name__}: {str(error)}",
        exc_info=True
    )


def log_batch_summary(
    logger: logging.Logger,
    total: int,
    successful: int,
    errors: int,
    needs_review: int,
    duration_seconds: float
) -> None:
    """
    Log batch processing summary.
    
    Args:
        logger: Logger instance
        total: Total images processed
        successful: Number of successful predictions
        errors: Number of errors
        needs_review: Number of predictions needing review
        duration_seconds: Processing duration in seconds
    """
    logger.info(
        f"BATCH SUMMARY | Total: {total} | Successful: {successful} | "
        f"Errors: {errors} | Needs Review: {needs_review} | "
        f"Duration: {duration_seconds:.2f}s | "
        f"Speed: {total/duration_seconds:.2f} images/sec"
    )


class PredictionLogger:
    """
    CSV-based prediction logger for audit trail.
    """
    
    def __init__(self, log_dir: Path):
        """
        Initialize prediction logger.
        
        Args:
            log_dir: Directory for prediction logs
        """
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"predictions_{timestamp}.csv"
        
        # Write header
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("timestamp,filename,predicted_class,class_name,confidence,blur_score,quality_issues,needs_review\n")
    
    def log(
        self,
        filename: str,
        predicted_class: int,
        class_name: str,
        confidence: float,
        blur_score: float,
        quality_issues: List[str],
        needs_review: bool
    ) -> None:
        """
        Log a prediction to CSV file.
        
        Args:
            filename: Image filename
            predicted_class: Predicted class index
            class_name: Predicted class name
            confidence: Prediction confidence
            blur_score: Blur detection score
            quality_issues: List of quality issues
            needs_review: Review flag
        """
        timestamp = datetime.now().isoformat()
        quality_str = "|".join(quality_issues) if quality_issues else "OK"
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(
                f"{timestamp},{filename},{predicted_class},{class_name},"
                f"{confidence:.4f},{blur_score:.2f},{quality_str},{needs_review}\n"
            )
