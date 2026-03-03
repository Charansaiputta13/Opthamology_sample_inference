# 🩺 Diabetic Retinopathy Classification System

A production-ready AI system for automated Diabetic Retinopathy (DR) screening from retinal fundus images using deep learning with explainability.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

> ⚡ **Status:** PyTorch Migration Complete ✅  
> Advanced CAM methods (GradCAM, GradCAM++, ScoreCAM, EigenCAM, LayerCAM) now available for explainability

---

## � Live Demo

**Try the application right now - no installation required!**

👉 **[Launch Live Demo on Streamlit Cloud](https://opthamologysampleinference-p8usjvppwcnpwcnxscxaqv.streamlit.app/)**

Simply click the link above to access the fully functional DR classification system in your browser. Upload retinal images, get predictions, visualize CAM heatmaps, and process batch data - all live!

---

## �🎯 Features

- **5-Class DR Classification:** No DR, Mild, Moderate, Severe, Proliferative DR (PDR)
- **PyTorch Backend:** Modern, efficient deep learning framework
- **Single Image Inference:** Real-time prediction with confidence scoring
- **Batch Processing:** Process hundreds of images from ZIP files efficiently
- **🎨 5 CAM Methods:** GradCAM, GradCAM++, ScoreCAM, EigenCAM, LayerCAM for explainability
- **Quality Validation:** Automatic blur detection and resolution checks
- **Confidence Thresholding:** Flag low-confidence predictions for expert review
- **Production Architecture:** Modular, tested, maintainable codebase
- **Web Interface:** Beautiful Streamlit UI with interactive visualization
- **CSV Reports:** Downloadable batch results with detailed metrics

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd Opthamology_sample_inference

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy the example environment file and customize:

```bash
cp .env.example .env
# Edit .env with your paths and settings
```

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📂 Project Structure

```
Opthamology_sample_inference/
├── src/                        # Core application modules
│   ├── config.py              # Configuration management (PyTorch)
│   ├── model.py               # Model loading & 5 CAM methods
│   ├── preprocessing.py       # Image preprocessing & quality checks
│   ├── inference.py           # Single & batch inference
│   ├── evaluation.py          # Metrics & evaluation
│   └── utils/                 # Utility modules
│       ├── logging_utils.py   # Structured logging
│       └── validation.py      # Input validation
├── app.py                      # Streamlit web application
├── pretrained/                 # Trained model files
│   └── dr_mobilenetv2_5class.pth  # PyTorch model weights
├── outputs/                    # Generated outputs
│   ├── logs/                  # Application logs
│   ├── predictions/           # Prediction CSVs
│   └── eval/                  # Evaluation results
├── data/                       # Data directory (optional)
├── requirements.txt           # Python dependencies
├── .env.example               # Environment configuration template
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

---

## 🖥️ Usage

### Web Application (Recommended)

The Streamlit app provides a complete interface with 4 tabs:

#### 📸 Single Image Tab
1. Upload a retinal fundus image (JPG/PNG)
2. View prediction with confidence percentage
3. See Grad-CAM heatmap highlighting important regions
4. Check image quality metrics (blur, resolution, brightness)
5. Get automatic "needs review" flag for uncertain cases

#### 📦 Batch Inference Tab
1. Prepare a ZIP file with multiple retinal images
2. Upload and process all images in batches
3. Monitor progress with real-time progress bar
4. Download complete CSV report with all predictions
5. View summary statistics and class distribution

#### 📊 Evaluation Tab
- View confusion matrix visualization
- Read detailed classification report
- Analyze per-class performance metrics

#### ℹ️ About Tab
- Model information and architecture details
- Medical disclaimer and usage guidelines
- Configuration documentation

### Python API (Programmatic Use)

```python
from src.config import Config
from src.model import DRClassifier
from src.inference import predict_single_image

# Initialize
config = Config()
model = DRClassifier(config.MODEL_PATH)

# Single prediction
result = predict_single_image(
    img_path="path/to/image.jpg",
    model=model,
    config=config,
    gradcam=True
)

print(f"Prediction: {result['class_name']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Quality Issues: {result['quality_issues']}")
```

### Batch Processing

```python
from src.inference import predict_batch_from_zip

# Batch from ZIP
results_df = predict_batch_from_zip(
    zip_path="images.zip",
    model=model,
    config=config
)

# Save results
results_df.to_csv("batch_results.csv", index=False)
```

### Model Evaluation

```python
from src.evaluation import evaluate_model

# Evaluate on test set
metrics = evaluate_model(
    csv_path="data/test_labels.csv",
    img_dir="data/test_images",
    model=model,
    config=config,
    output_dir="outputs/eval"
)

print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Cohen's Kappa: {metrics['cohen_kappa']:.3f}")
```

---

## 🤖 Model Information

| Property | Value |
|----------|-------|
| **Architecture** | MobileNetV2 (fine-tuned) |
| **Input Size** | 224×224 pixels |
| **Output Classes** | 5 (No DR, Mild, Moderate, Severe, PDR) |
| **Framework** | **PyTorch 2.1+** |
| **Preprocessing** | ImageNet normalization (mean/std) |
| **Deployment** | CPU-optimized (GPU-compatible) |
| **Model File** | `.pth` (PyTorch state dict) |

### Class Definitions

| Class Index | Class Name | Description |
|-------------|------------|-------------|
| 0 | No DR | No Diabetic Retinopathy detected |
| 1 | Mild | Mild non-proliferative DR |
| 2 | Moderate | Moderate non-proliferative DR |
| 3 | Severe | Severe non-proliferative DR |
| 4 | PDR | Proliferative Diabetic Retinopathy |

---

## ⚙️ Configuration

Configuration is managed via environment variables (`.env` file) or `src/config.py`.

### Key Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `pretrained/dr_mobilenetv2_5class.h5` | Path to model file |
| `CONFIDENCE_THRESHOLD` | `0.6` | Minimum confidence for reliable predictions |
| `BLUR_THRESHOLD` | `100.0` | Laplacian variance threshold (lower = more blur) |
| `MIN_IMAGE_SIZE` | `256` | Minimum acceptable image dimension |
| `BATCH_SIZE` | `16` | Batch size for inference |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## 🔬 Quality Checks

The system performs automatic quality validation:

### Blur Detection
- **Method:** Laplacian variance
- **Threshold:** < 100 flagged as blurry
- **Action:** Warning displayed, "needs review" flag set

### Resolution Check
- **Minimum:** 256×256 pixels
- **Action:** Warning for low-resolution images

### Brightness Analysis
- **Range:** Mean brightness checked
- **Action:** Flag very dark (< 30) or very bright (> 225) images

---

## 📊 Evaluation Metrics

The system provides comprehensive evaluation:

- **Accuracy:** Overall classification accuracy
- **Balanced Accuracy:** Weighted accuracy for imbalanced datasets
- **Cohen's Kappa:** Inter-rater agreement metric
- **Per-Class Metrics:** Precision, Recall, F1-Score for each class
- **Confusion Matrix:** Visual representation of predictions vs. ground truth
- **Classification Report:** Detailed sklearn report

---

## 🛠️ Development

### Code Quality

- **Type Hints:** Full type annotations throughout
- **Docstrings:** Google-style docstrings for all functions/classes
- **Error Handling:** Comprehensive try/except with graceful fallbacks
- **Logging:** Structured logging with file rotation
- **Validation:** Input validation for all user inputs

### Architecture Principles

1. **Separation of Concerns:** Each module has single responsibility
2. **Dependency Injection:** Config and models passed as parameters
3. **Fail Gracefully:** Never crash on bad input, always log errors
4. **Memory Efficient:** Streaming ZIP processing, chunked batches
5. **CPU-First:** No GPU assumptions, works on any machine

---

## ⚠️ Important Disclaimers

### Medical Disclaimer

> **THIS SYSTEM IS FOR EDUCATIONAL, RESEARCH, AND DEMONSTRATION PURPOSES ONLY.**
>
> **NOT FOR CLINICAL USE**
>
> - ❌ NOT FDA approved or clinically validated
> - ❌ NOT a substitute for professional medical examination
> - ❌ NOT intended for diagnosis or treatment decisions
> - ✅ Intended for screening support and research only
>
> **Always consult qualified ophthalmologists and healthcare professionals for medical decisions.**

### Limitations

- **Training Data:** Model performance depends on training data quality and diversity
- **Generalization:** May not generalize well to images from different capture devices
- **Edge Cases:** May fail on severely degraded or atypical images
- **No Warranty:** Provided as-is without guarantees of accuracy

---

## 🧪 Testing

### Manual Testing

```bash
# Test configuration loading
python -c "from src.config import Config; print(Config())"

# Test model loading
python -c "from src.model import DRClassifier; from src.config import Config; DRClassifier(Config().MODEL_PATH)"

# Run single inference
streamlit run app.py  # Use Single Image tab
```

### Batch Testing

Create a test ZIP with sample images and use the Batch tab in the web app.

---

## 🚢 Deployment

### Local Deployment

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Docker (Optional)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t dr-classifier .
docker run -p 8501:8501 dr-classifier
```

---

## 📝 License

This project is provided for educational purposes. See LICENSE file for details.

---

## 🎨 CAM Explainability Methods

The system supports **5 state-of-the-art CAM methods** for visual explanations:

### 1. **GradCAM** (Gradient-weighted Class Activation Mapping)
- Uses gradients flowing into the last convolutional layer
- Fast and widely used for model interpretability
- Good for understanding spatial attention regions

### 2. **GradCAM++**
- Improved version of GradCAM with better gradient weighting
- Provides more fine-grained visualizations
- Better at handling multiple objects/regions of interest

### 3. **ScoreCAM** (Score-weighted Class Activation Mapping)
- Doesn't rely on gradients, uses forward pass only
- More stable and computationally intensive
- Useful when gradient-based methods are unreliable

### 4. **EigenCAM**
- Unsupervised version of ScoreCAM
- Uses principal components of activations
- Good for general feature visualization

### 5. **LayerCAM**
- Alternative to GradCAM focusing on activation maps only
- Minimal gradient dependency
- Useful for verification of gradient-based methods

### Usage Example

```python
from src.model import DRClassifier
from src.preprocessing import preprocess_image, load_image_safe

config = Config()
model = DRClassifier(config.MODEL_PATH)

# Load and preprocess image
img_pil = load_image_safe("retina.jpg")
img_array = preprocess_image(img_pil)

# Compute CAM using different methods
methods = ["GradCAM", "GradCAM++", "ScoreCAM", "EigenCAM", "LayerCAM"]

for method in methods:
    heatmap = model.compute_cam(img_array, method=method)
    overlayed = model.overlay_cam(img_pil, heatmap)
    fig = model.create_cam_figure(img_pil, heatmap, method=method)
```

---

## 📊 Acknowledgments

- **PyTorch Team** - Deep learning framework
- **Torchvision** - Computer vision models
- **pytorch-grad-cam** - CAM implementations
- **Streamlit** - Web app framework
- **Scikit-learn** - Evaluation metrics
- **OpenCV & PIL** - Image processing

---

## 🔑 Keywords

Diabetic Retinopathy, Medical AI, PyTorch, Deep Learning, Computer Vision, Explainable AI, CAM, GradCAM, MobileNetV2, Streamlit, Healthcare ML, Medical Imaging, Retinal Imaging, Clinical Decision Support, ML Engineering, Production ML

---

**Made with ❤️ for advancing AI in healthcare**
