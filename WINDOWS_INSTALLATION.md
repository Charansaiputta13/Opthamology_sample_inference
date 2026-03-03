# Windows Installation Guide

## Complete Step-by-Step Setup for Windows

This guide covers installing and running the Diabetic Retinopathy Classification System on Windows 10/11.

---

## Prerequisites

### Check Python Installation
1. Open Command Prompt (Win + R, type `cmd`)
2. Verify Python is installed:
   ```cmd
   python --version
   ```
   - Should show Python 3.9 or higher
   - If not installed, download from https://www.python.org/downloads/

### Check Pip
```cmd
python -m pip --version
```

---

## Step 1: Open Command Prompt

1. Press `Win + R`
2. Type `cmd` and press Enter
3. Navigate to project directory:
   ```cmd
   cd C:\Users\YourUsername\PycharmProjects\Opthamology_sample_inference
   ```

---

## Step 2: Create Virtual Environment

Creating a virtual environment isolates your project dependencies.

```cmd
python -m venv venv
```

**Activate the virtual environment:**
```cmd
venv\Scripts\activate
```

You should see `(venv)` prefix in your prompt:
```
(venv) C:\Users\...\Opthamology_sample_inference>
```

---

## Step 3: Upgrade Pip

Ensure pip is up to date:
```cmd
python -m pip install --upgrade pip
```

---

## Step 4: Install Dependencies

Install all required packages:
```cmd
pip install -r requirements.txt
```

**What's being installed:**
- PyTorch (Deep Learning Framework)
- Torchvision (Vision Models)
- Grad-CAM (Explainability)
- Streamlit (Web UI)
- And more...

⏱️ **This takes 5-15 minutes** depending on internet speed.

**Alternative (if above fails):**
```cmd
pip install --upgrade --user -r requirements.txt
```

---

## Step 5: Verify Installation

Run the verification script:
```cmd
python verify_pytorch_setup.py
```

### Expected Output
```
============================================================
PYTORCH DETAILS
============================================================
PyTorch Version: 2.1.0
CUDA Available: False
Python Version: 3.x.x

============================================================
MODEL FILE VERIFICATION
============================================================
✅ Model file exists
   Path: C:\...\pretrained\dr_mobilenetv2_5class.pth
   Size: 14.20 MB

============================================================
MODEL LOADING TEST
============================================================
✅ Model loaded successfully!

Model Information:
  model_path: pretrained\dr_mobilenetv2_5class.pth
  architecture: MobileNetV2
  cam_methods: GradCAM, GradCAM++, ScoreCAM, EigenCAM, LayerCAM

============================================================
SUMMARY
============================================================
🎉 SETUP COMPLETE - All checks passed!
```

### Troubleshooting Verification

**If you see errors:**

1. **ModuleNotFoundError: torch**
   ```cmd
   pip install torch torchvision
   ```

2. **ModuleNotFoundError: grad_cam**
   ```cmd
   pip install grad-cam
   ```

3. **Model file not found**
   - Check that `dr_mobilenetv2_5class.pth` exists in `pretrained/` folder
   - File size should be ~14 MB

---

## Step 6: Run the Web Application

### Start Streamlit
```cmd
streamlit run app.py
```

### Open in Browser
- Streamlit typically opens automatically
- If not, visit: http://localhost:8501

### First Load
- Takes ~5-10 seconds to load the model
- Wait for "✅ Model loaded successfully"
- Then use the interface

---

## Testing the Installation

### Test 1: Run Examples
```cmd
python examples_inference.py --all
```

This runs 4 different inference examples and shows CAM computation.

### Test 2: CAM Visualization
```cmd
python examples_cam_visualization.py --random
```

This creates CAM visualizations with a dummy image.

### Test 3: Single Image
```cmd
python examples_inference.py --single C:\path\to\image.jpg
```

Replace with actual image path.

---

## File Operations in Windows

### Create .env File
1. Open File Explorer
2. Navigate to project folder
3. Right-click → New → Text Document
4. Name it `.env` (not `.env.txt`)
5. Edit with Notepad and copy contents from `.env.example`

Or via Command Prompt:
```cmd
copy .env.example .env
```

### Add Images
1. Folder: `data/images/`
2. Copy your retinal fundus images here
3. Supported formats: `.jpg`, `.png`, `.bmp`, `.tiff`

---

## Common Windows Issues & Solutions

### Issue: "python command not found"
**Solution:**
- Add Python to PATH
- Or use full path: `C:\Python312\python.exe verify_pytorch_setup.py`
- Or reinstall Python with "Add Python to PATH" option

### Issue: "Permission Denied" when creating venv
**Solution:**
```cmd
# Run Command Prompt as Administrator
# Right-click → Run as Administrator
```

### Issue: Slow package installation
**Solution:**
```cmd
# Use faster mirror
pip install -r requirements.txt -i https://pypi.tsinghua.edu.cn/simple
```

### Issue: "pip install" hangs
**Solution:**
1. Press Ctrl+C to cancel
2. Try again with:
   ```cmd
   pip install --no-cache-dir -r requirements.txt
   ```

### Issue: Streamlit "port already in use"
**Solution:**
```cmd
streamlit run app.py --server.port 8502
```
(Uses port 8502 instead of 8501)

### Issue: GPU not detected (if available)
**Verify:**
```cmd
python -c "import torch; print(torch.cuda.is_available())"
```

If `False`, you need CUDA drivers. It's optional - system works fine with CPU.

---

## Command Reference

| Task | Command |
|------|---------|
| **Activate venv** | `venv\Scripts\activate` |
| **Deactivate venv** | `deactivate` |
| **Install packages** | `pip install -r requirements.txt` |
| **Verify setup** | `python verify_pytorch_setup.py` |
| **Run web app** | `streamlit run app.py` |
| **Test examples** | `python examples_inference.py --all` |
| **View Python version** | `python --version` |
| **Check pip** | `pip --version` |
| **Update pip** | `python -m pip install --upgrade pip` |

---

## Folder Structure After Setup

```
C:\Users\YourUsername\PycharmProjects\Opthamology_sample_inference\
│
├── venv\                          # Virtual environment
│   ├── Scripts\                   # Executables
│   │   ├── python.exe
│   │   ├── streamlit.exe
│   │   └── pip.exe
│   └── Lib\                       # Installed packages
│
├── src\                           # Source code
│   ├── model.py                   # PyTorch model
│   ├── inference.py
│   ├── preprocessing.py
│   └── ...
│
├── pretrained\
│   └── dr_mobilenetv2_5class.pth  # Model weights (14 MB)
│
├── data\
│   └── images\                    # Your images here
│
├── outputs\
│   ├── logs\
│   ├── predictions\               # CSV results
│   └── eval\
│
├── app.py                         # Streamlit app
├── requirements.txt               # Dependencies
├── .env                           # Config (you create from .env.example)
├── QUICKSTART.md                  # Quick reference
├── README.md                      # Full documentation
└── ...
```

---

## Performance on Windows

| Component | Performance |
|-----------|-------------|
| Model Load | 1-2 sec |
| Single Prediction | 0.5 sec |
| CAM (GradCAM) | 0.2 sec |
| Batch (100 imgs) | 30-60 sec |

*Times vary based on system specs*

---

## GPU Support (Optional)

### Check if GPU Available
```cmd
python -c "import torch; print(torch.cuda.is_available())"
```

### If GPU Available (NVIDIA)
```cmd
# Install GPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### If GPU Available (AMD)
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

---

## IDE Setup (Optional)

### PyCharm
1. Open Project
2. File → Settings → Project → Python Interpreter
3. Select `C:\...\venv\Scripts\python.exe`
4. Terminal should automatically activate venv

### VS Code
1. Open project folder
2. Install "Python" extension
3. Select interpreter: `./venv/Scripts/python.exe`
4. Terminal → New Terminal (auto-activates venv)

---

## Typical Workflow

### Day 1: Setup
```cmd
# First time setup only
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python verify_pytorch_setup.py
```

### Daily Use
```cmd
# Each time you use the project
venv\Scripts\activate

# Option A: Web interface
streamlit run app.py

# Option B: Python API
python your_script.py

# Option C: Examples
python examples_inference.py --all
```

### Deactivate
```cmd
deactivate
```

---

## Next Steps

1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Verify setup: `python verify_pytorch_setup.py`
3. ✅ Try examples: `python examples_inference.py --all`
4. ✅ Run web app: `streamlit run app.py`
5. ✅ Read documentation: [QUICKSTART.md](QUICKSTART.md)

---

## Getting Help

### Check Installation
```cmd
python verify_pytorch_setup.py
```

### Check Dependencies
```cmd
pip list
```

### Check PyTorch
```cmd
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

### Check CAM
```cmd
python -c "from pytorch_grad_cam import GradCAM; print('CAM OK')"
```

### Check Streamlit
```cmd
streamlit --version
```

---

## Documentation Files

Keep these bookmarks handy:
- 📖 **QUICKSTART.md** - Quick reference (30 seconds)
- 📖 **README.md** - Full documentation
- 📖 **MIGRATION_GUIDE.md** - PyTorch details
- 📖 **CAM_METHODS_GUIDE.md** - CAM explanation

---

## Support

If you encounter issues:
1. Run `python verify_pytorch_setup.py`
2. Check `QUICKSTART.md` troubleshooting
3. See example scripts: `examples_*.py`
4. Review code docstrings in `src/`

---

## Summary

You now have a complete, production-ready Diabetic Retinopathy Classification system with:

✅ PyTorch backend  
✅ 5 CAM methods for explainability  
✅ Web interface  
✅ Batch processing  
✅ Quality validation  

**Ready to predict retinal images!** 🎉

---

**Last Updated:** 2025-03-03  
**Framework:** PyTorch 2.1+  
**Python:** 3.9+  
**OS:** Windows 10/11

