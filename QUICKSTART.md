# Quick Start Guide

## 30-Second Setup

```bash
# 1. Clone/navigate to project
cd Opthamology_sample_inference

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python verify_pytorch_setup.py

# 5. Run Streamlit app
streamlit run app.py
```

---

## What You'll Get

✅ **5 CAM Methods** - GradCAM, GradCAM++, ScoreCAM, EigenCAM, LayerCAM  
✅ **Web Interface** - Beautiful Streamlit UI  
✅ **Real-time Predictions** - Fast PyTorch inference  
✅ **Batch Processing** - Process 100s of images  
✅ **Quality Checks** - Blur detection, resolution validation  
✅ **Explainability** - Visual explanations via CAM heatmaps  

---

## System Requirements

| Component | Requirement |
|-----------|------------|
| **Python** | 3.9 - 3.11 |
| **RAM** | 4GB minimum (8GB recommended) |
| **Disk** | 500MB free |
| **GPU** | Optional (can use CPU) |
| **OS** | Windows, macOS, Linux |

---

## Installation Steps

### Step 1: Prerequisites
```bash
# Ensure you have Python 3.9+
python --version  # Should be 3.9 or higher

# Ensure pip is updated
python -m pip install --upgrade pip
```

### Step 2: Clone Repository
```bash
git clone <repository-url>
cd Opthamology_sample_inference
```

### Step 3: Create Virtual Environment
```bash
# Create
python -m venv venv

# Activate
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

**What's being installed:**
- `torch` - PyTorch deep learning framework
- `torchvision` - Pre-trained vision models
- `grad-cam` - CAM visualization library
- `streamlit` - Web UI framework
- `pandas`, `numpy`, `opencv-python` - Data processing
- `scikit-learn` - Evaluation metrics

### Step 5: Verify Installation
```bash
python verify_pytorch_setup.py
```

**Expected output:**
```
✅ All packages installed!
✅ Model file exists
✅ Model loaded successfully!
✅ All CAM methods available
```

---

## Running the Application

### Option A: Web Interface (Recommended)
```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

**Features:**
- 📸 Single image prediction with CAM
- 📦 Batch processing from ZIP
- 📊 Model evaluation & metrics
- ℹ️ System information

### Option B: Command Line Examples
```bash
# Run all examples
python examples_inference.py --all

# Single image inference
python examples_inference.py --single path/to/image.jpg

# Batch from ZIP
python examples_inference.py --batch path/to/images.zip

# CAM visualization
python examples_cam_visualization.py --random
python examples_cam_visualization.py path/to/image.jpg
```

### Option C: Python API
```python
from src.config import Config
from src.model import DRClassifier
from src.inference import predict_single_image

# Setup
config = Config()
model = DRClassifier(config.MODEL_PATH)

# Predict single image
result = predict_single_image(
    img_path="image.jpg",
    model=model,
    config=config,
    cam_method="GradCAM"  # Choose CAM method
)

print(f"Prediction: {result['class_name']}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

## Key Files

| File | Purpose |
|------|---------|
| `app.py` | 🌐 Streamlit web application |
| `src/model.py` | 🤖 PyTorch model + 5 CAM methods |
| `src/inference.py` | 📊 Single & batch inference |
| `src/config.py` | ⚙️ Configuration management |
| `src/preprocessing.py` | 🖼️ Image loading & preprocessing |
| `src/evaluation.py` | 📈 Metrics & evaluation |
| `verify_pytorch_setup.py` | ✅ Setup verification |
| `examples_*.py` | 📚 Usage examples |

---

## Configuration

### Using Environment Variables

Create `.env` file from template:
```bash
cp .env.example .env
```

Edit `.env` with your settings:
```dotenv
# Model path
MODEL_PATH=pretrained/dr_mobilenetv2_5class.pth

# CAM method
DEFAULT_CAM_METHOD=GradCAM

# Thresholds
CONFIDENCE_THRESHOLD=0.6
BLUR_THRESHOLD=100.0
MIN_IMAGE_SIZE=256
```

### Programmatic Configuration

```python
from src.config import Config

config = Config()
print(config.MODEL_PATH)
print(config.CAM_METHODS)
print(config.IMG_SIZE)
```

---

## Usage Examples

### Single Image Prediction
```python
from src.inference import predict_single_image
from src.model import DRClassifier
from src.config import Config

config = Config()
model = DRClassifier(config.MODEL_PATH)

result = predict_single_image(
    img_path="retinal_image.jpg",
    model=model,
    config=config,
    cam_method="GradCAM"  # or "GradCAM++", etc
)

print(f"Class: {result['class_name']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Multiple CAM Methods
```python
for method in ["GradCAM", "GradCAM++", "ScoreCAM", "EigenCAM", "LayerCAM"]:
    heatmap = model.compute_cam(img_array, method=method)
    overlay = model.overlay_cam(original_img, heatmap)
    print(f"{method}: computed")
```

### Batch Processing
```python
from src.inference import predict_batch_from_zip

results_df = predict_batch_from_zip(
    zip_path="images.zip",
    model=model,
    config=config,
    use_cam=False  # Skip CAM for speed
)

results_df.to_csv("predictions.csv", index=False)
```

### Model Evaluation
```python
from src.evaluation import evaluate_model

metrics = evaluate_model(
    csv_path="labels.csv",
    img_dir="images",
    model=model,
    config=config,
    output_dir="outputs/eval"
)

print(f"Accuracy: {metrics['accuracy']:.3f}")
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution:**
```bash
pip install torch torchvision
```

### Issue: "Model file not found"
**Solution:**
- Ensure `.pth` file exists at: `pretrained/dr_mobilenetv2_5class.pth`
- Or set `MODEL_PATH` environment variable

### Issue: "CAM computation failed"
**Solution:**
- Check image is properly normalized
- Verify model loaded successfully
- Try different CAM method

### Issue: Streamlit app won't start
**Solution:**
```bash
# Clear cache and restart
streamlit run app.py --logger.level=debug
```

### Issue: Slow batch processing
**Solution:**
```python
# Disable CAM for speed
predict_batch_from_zip(..., use_cam=False)

# Or use faster CAM method
cam_method="LayerCAM"  # Instead of ScoreCAM
```

---

## CAM Methods at a Glance

| Method | Speed | Accuracy | Best For |
|--------|-------|----------|----------|
| **GradCAM** | ⚡ Fastest | ⭐⭐⭐⭐ | Quick explanations |
| **GradCAM++** | ⚡⚡ Fast | ⭐⭐⭐⭐⭐ | Fine details |
| **ScoreCAM** | 🐢 Slow | ⭐⭐⭐⭐⭐ | Maximum accuracy |
| **EigenCAM** | ⚡⚡⚡ Medium | ⭐⭐⭐⭐ | Exploration |
| **LayerCAM** | ⚡⚡ Fast | ⭐⭐⭐⭐ | Verification |

👉 **Start with:** GradCAM (fast & good)  
👉 **For details:** GradCAM++ (fine-grained)  
👉 **For accuracy:** ScoreCAM (most faithful)  

---

## Directory Structure

```
Opthamology_sample_inference/
├── app.py                          # Streamlit app
├── requirements.txt                # Dependencies
├── .env.example                    # Config template
├── verify_pytorch_setup.py        # Verification
├── examples_*.py                  # Example scripts
├── src/
│   ├── config.py                  # Configuration
│   ├── model.py                   # PyTorch model + CAM
│   ├── preprocessing.py           # Image processing
│   ├── inference.py               # Inference pipeline
│   ├── evaluation.py              # Metrics
│   └── utils/
│       ├── logging_utils.py
│       └── validation.py
├── pretrained/
│   └── dr_mobilenetv2_5class.pth  # Model weights
├── outputs/                        # Generated outputs
│   ├── logs/
│   ├── predictions/
│   └── eval/
└── README.md                       # Full documentation
```

---

## Common Commands

```bash
# Verify setup
python verify_pytorch_setup.py

# Run web app
streamlit run app.py

# Single image
python examples_inference.py --single image.jpg

# All examples
python examples_inference.py --all

# CAM visualization
python examples_cam_visualization.py --random

# Check version
python -c "import torch; print(torch.__version__)"
```

---

## Performance Expectations

| Task | Time | Hardware |
|------|------|----------|
| Model Load | 1-2 sec | CPU |
| Single Pred | 0.5 sec | CPU |
| GradCAM | 0.2 sec | CPU |
| ScoreCAM | 2-5 sec | CPU |
| Batch (100 imgs) | 30 sec | CPU |

**With GPU:** ~5-10x faster

---

## Next Steps

1. ✅ Install dependencies (`pip install -r requirements.txt`)
2. ✅ Verify setup (`python verify_pytorch_setup.py`)
3. ✅ Try examples (`python examples_inference.py --all`)
4. ✅ Run web app (`streamlit run app.py`)
5. ✅ Add your images to `data/images/`
6. ✅ Make predictions and explore CAM methods
7. ✅ Review [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for in-depth info
8. ✅ Check [CAM_METHODS_GUIDE.md](CAM_METHODS_GUIDE.md) for CAM details

---

## Getting Help

| Issue | Resource |
|-------|----------|
| **Setup** | `python verify_pytorch_setup.py` |
| **Examples** | `python examples_*.py` |
| **CAM Details** | [CAM_METHODS_GUIDE.md](CAM_METHODS_GUIDE.md) |
| **Migration** | [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) |
| **API Details** | Code docstrings in `src/*.py` |

---

## Environment Info

- **Framework:** PyTorch 2.1+
- **Model:** MobileNetV2 (224×224 input)
- **Classes:** 5 (DR severity levels)
- **CAM Methods:** 5 (GradCAM, GradCAM++, ScoreCAM, EigenCAM, LayerCAM)
- **Python:** 3.9+
- **License:** MIT

---

🚀 **Ready to start?**

```bash
# In your terminal:
pip install -r requirements.txt
python verify_pytorch_setup.py
streamlit run app.py
```

**That's it!** Visit http://localhost:8501 and start making predictions.

---

**Last Updated:** 2025-03-03

