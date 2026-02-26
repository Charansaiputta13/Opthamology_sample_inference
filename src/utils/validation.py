"""
Input validation utilities for DR classification system.
Validates files, paths, and inputs with graceful error handling.
"""

import zipfile
from pathlib import Path
from typing import Tuple, List, Optional


def validate_image_file(path: str) -> Tuple[bool, str]:
    """
    Validate that a file is a readable image.
    
    Args:
        path: Path to image file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    file_path = Path(path)
    
    # Check existence
    if not file_path.exists():
        return False, f"File does not exist: {path}"
    
    # Check it's a file
    if not file_path.is_file():
        return False, f"Path is not a file: {path}"
    
    # Check extension
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    if file_path.suffix.lower() not in valid_extensions:
        return False, f"Invalid file extension: {file_path.suffix}. Expected: {', '.join(valid_extensions)}"
    
    # Check readability
    try:
        with open(file_path, 'rb') as f:
            # Try to read first few bytes
            f.read(10)
        return True, ""
    except Exception as e:
        return False, f"Cannot read file: {str(e)}"


def validate_zip_file(path: str) -> Tuple[bool, str, int]:
    """
    Validate that a file is a readable ZIP containing images.
    
    Args:
        path: Path to ZIP file
        
    Returns:
        Tuple of (is_valid, error_message, num_images)
    """
    file_path = Path(path)
    
    # Check existence
    if not file_path.exists():
        return False, f"File does not exist: {path}", 0
    
    # Check it's a file
    if not file_path.is_file():
        return False, f"Path is not a file: {path}", 0
    
    # Check extension
    if file_path.suffix.lower() != '.zip':
        return False, f"File is not a ZIP: {file_path.suffix}", 0
    
    # Try to open ZIP
    try:
        with zipfile.ZipFile(file_path, 'r') as z:
            # Get list of files
            all_files = z.namelist()
            
            # Filter for images
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
            image_files = [
                f for f in all_files
                if not f.startswith('__MACOSX')  # Ignore macOS metadata
                and not f.startswith('.')  # Ignore hidden files
                and f.lower().endswith(valid_extensions)
            ]
            
            if len(image_files) == 0:
                return False, "ZIP contains no valid image files", 0
            
            return True, "", len(image_files)
    
    except zipfile.BadZipFile:
        return False, "File is not a valid ZIP archive", 0
    except Exception as e:
        return False, f"Error reading ZIP: {str(e)}", 0


def validate_model_path(path: str) -> Tuple[bool, str]:
    """
    Validate that model file exists and has correct format.
    
    Args:
        path: Path to model file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    model_path = Path(path)
    
    # Check existence
    if not model_path.exists():
        return False, f"Model file does not exist: {path}"
    
    # Check it's a file or directory (SavedModel format)
    if not (model_path.is_file() or model_path.is_dir()):
        return False, f"Invalid model path: {path}"
    
    # Check format
    if model_path.is_file():
        # Should be .h5 or .keras
        if model_path.suffix.lower() not in ['.h5', '.keras']:
            return False, f"Invalid model format: {model_path.suffix}. Expected .h5 or .keras"
    else:
        # SavedModel directory should contain saved_model.pb
        pb_file = model_path / "saved_model.pb"
        if not pb_file.exists():
            return False, f"SavedModel directory missing saved_model.pb: {path}"
    
    return True, ""


def validate_csv_file(path: str, required_columns: Optional[List[str]] = None) -> Tuple[bool, str]:
    """
    Validate that CSV file exists and contains required columns.
    
    Args:
        path: Path to CSV file
        required_columns: List of required column names
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    import pandas as pd
    
    file_path = Path(path)
    
    # Check existence
    if not file_path.exists():
        return False, f"CSV file does not exist: {path}"
    
    # Check it's a file
    if not file_path.is_file():
        return False, f"Path is not a file: {path}"
    
    # Check extension
    if file_path.suffix.lower() != '.csv':
        return False, f"File is not a CSV: {file_path.suffix}"
    
    # Try to read and check columns
    if required_columns:
        try:
            df = pd.read_csv(file_path, nrows=1)
            df.columns = df.columns.str.strip()  # Clean column names
            
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                return False, f"CSV missing required columns: {', '.join(missing_cols)}"
            
        except Exception as e:
            return False, f"Error reading CSV: {str(e)}"
    
    return True, ""


def get_valid_images_from_zip(zip_path: str) -> List[str]:
    """
    Extract list of valid image filenames from ZIP.
    
    Args:
        zip_path: Path to ZIP file
        
    Returns:
        List of valid image filenames
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            all_files = z.namelist()
            image_files = [
                f for f in all_files
                if not f.startswith('__MACOSX')
                and not f.startswith('.')
                and not f.endswith('/')  # Ignore directories
                and f.lower().endswith(valid_extensions)
            ]
            return image_files
    except Exception:
        return []


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file operations.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove path components (keep only basename)
    filename = Path(filename).name
    
    # Remove or replace unsafe characters
    unsafe_chars = '<>:"|?*\\'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    return filename
