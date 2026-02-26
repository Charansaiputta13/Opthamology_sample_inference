"""
Model evaluation utilities for DR classification.
Generates metrics, confusion matrices, and classification reports.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from pathlib import Path
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score
)

from .config import Config
from .model import DRClassifier
from .preprocessing import load_image_safe, preprocess_image


def evaluate_model(
    csv_path: str,
    img_dir: str,
    model: DRClassifier,
    config: Config,
    output_dir: str
) -> Dict[str, Any]:
    """
    Evaluate model on labeled dataset and generate reports.
    
    Args:
        csv_path: Path to CSV with columns: 'Image name', 'Retinopathy grade'
        img_dir: Directory containing images
        model: DRClassifier instance
        config: Configuration object
        output_dir: Directory to save evaluation outputs
        
    Returns:
        Dictionary with metrics:
            - accuracy
            - balanced_accuracy
            - cohen_kappa
            - per_class_metrics
            - confusion_matrix_path
            - classification_report_path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load CSV
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    if 'Image name' not in df.columns or 'Retinopathy grade' not in df.columns:
        raise ValueError("CSV must contain 'Image name' and 'Retinopathy grade' columns")
    
    image_names = df['Image name'].values
    gt_labels = df['Retinopathy grade'].values
    
    # Run inference
    y_true = []
    y_pred = []
    successful = 0
    failed = 0
    
    print(f"Running evaluation on {len(image_names)} images...")
    
    for img_name, gt_label in zip(image_names, gt_labels):
        img_path = Path(img_dir) / img_name
        
        try:
            # Load and preprocess
            img_pil = load_image_safe(img_path)
            if img_pil is None:
                failed += 1
                continue
            
            img_preprocessed = preprocess_image(img_pil, target_size=config.IMG_SIZE)
            if img_preprocessed is None:
                failed += 1
                continue
            
            # Predict
            pred_class, _ = model.predict(img_preprocessed)
            
            y_true.append(int(gt_label))
            y_pred.append(pred_class)
            successful += 1
        
        except Exception as e:
            print(f"[WARNING] Failed on {img_name}: {e}")
            failed += 1
            continue
    
    if len(y_true) == 0:
        raise RuntimeError("No successful predictions. Evaluation failed.")
    
    print(f"Successfully evaluated {successful} images ({failed} failed)")
    
    # Convert to numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute metrics
    metrics = {
        'total_images': len(image_names),
        'successful': successful,
        'failed': failed,
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred)
    }
    
    # Generate classification report
    report_path = output_path / 'classification_report.txt'
    report = classification_report(
        y_true,
        y_pred,
        target_names=config.CLASS_NAMES,
        digits=4
    )
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DIABETIC RETINOPATHY CLASSIFICATION - EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Images: {len(image_names)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}\n")
        f.write(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}\n\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)
    
    metrics['classification_report_path'] = str(report_path)
    print(f"Saved classification report: {report_path}")
    
    # Generate confusion matrix
    cm_path = output_path / 'confusion_matrix.png'
    generate_confusion_matrix_plot(
        y_true,
        y_pred,
        class_names=config.CLASS_NAMES,
        save_path=str(cm_path)
    )
    metrics['confusion_matrix_path'] = str(cm_path)
    print(f"Saved confusion matrix: {cm_path}")
    
    # Per-class metrics
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(config.NUM_CLASSES))
    )
    
    per_class = {}
    for i, class_name in enumerate(config.CLASS_NAMES):
        per_class[class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }
    
    metrics['per_class_metrics'] = per_class
    
    return metrics


def generate_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: str,
    normalize: bool = False
) -> None:
    """
    Generate and save confusion matrix visualization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save plot
        normalize: Whether to normalize confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap='Blues', ax=ax, xticks_rotation=45, values_format='.2f' if normalize else 'd')
    
    ax.set_title(
        'Confusion Matrix - Diabetic Retinopathy Classification',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_metrics_summary_plot(
    per_class_metrics: Dict[str, Dict[str, float]],
    save_path: str
) -> None:
    """
    Generate bar chart of per-class metrics.
    
    Args:
        per_class_metrics: Dictionary of per-class metrics
        save_path: Path to save plot
    """
    classes = list(per_class_metrics.keys())
    precision = [per_class_metrics[c]['precision'] for c in classes]
    recall = [per_class_metrics[c]['recall'] for c in classes]
    f1 = [per_class_metrics[c]['f1_score'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width, precision, width, label='Precision', color='#4CAF50')
    ax.bar(x, recall, width, label='Recall', color='#2196F3')
    ax.bar(x + width, f1, width, label='F1-Score', color='#FF9800')
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_predictions_csv(
    image_names: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    class_names: List[str],
    save_path: str
) -> None:
    """
    Save detailed predictions to CSV.
    
    Args:
        image_names: List of image filenames
        y_true: True labels
        y_pred: Predicted labels
        probabilities: Prediction probabilities (N, num_classes)
        class_names: List of class names
        save_path: Path to save CSV
    """
    data = {
        'image_name': image_names,
        'true_label': y_true,
        'true_class': [class_names[i] for i in y_true],
        'pred_label': y_pred,
        'pred_class': [class_names[i] for i in y_pred],
        'correct': y_true == y_pred,
        'confidence': [probabilities[i, y_pred[i]] for i in range(len(y_pred))]
    }
    
    # Add per-class probabilities
    for i, class_name in enumerate(class_names):
        data[f'prob_{class_name.replace(" ", "_")}'] = probabilities[:, i]
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"💾 Saved detailed predictions: {save_path}")


if __name__ == "__main__":
    # Import here to avoid circular imports if any
    from src.config import get_config
    
    # Initialize configuration
    config = get_config()
    
    # Load model
    print(f"Loading model from: {config.MODEL_PATH}")
    try:
        model = DRClassifier(config.MODEL_PATH)
        
        # Set paths
        csv_path = config.BASE_DIR / "data" / "images" / "labels.csv"
        img_dir = config.BASE_DIR / "data" / "images"
        
        if not csv_path.exists():
            print(f"Label file not found: {csv_path}")
        elif not img_dir.exists():
            print(f"Image directory not found: {img_dir}")
        else:
            # Run evaluation
            results = evaluate_model(
                csv_path=str(csv_path),
                img_dir=str(img_dir),
                model=model,
                config=config,
                output_dir=str(config.EVAL_DIR)
            )
            
            print("\n" + "="*50)
            print("         EVALUATION SUMMARY")
            print("="*50)
            print(f"Accuracy:          {results['accuracy']:.4%}")
            print(f"Balanced Accuracy: {results['balanced_accuracy']:.4%}")
            print(f"Cohen's Kappa:     {results['cohen_kappa']:.4f}")
            print(f"Reports saved to:  {config.EVAL_DIR}")
            print("="*50)
            
    except Exception as e:
        print(f"Evaluation failed: {e}")
