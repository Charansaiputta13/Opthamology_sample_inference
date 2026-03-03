# Docker & Cloud Deployment Guide

## Overview

This guide covers deploying the DR Classification System using Docker, Docker Compose, and cloud platforms.

---

## Problem: libGL.so.1 Error

**Error:**
```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

**Cause:** OpenCV with graphical support requires OpenGL libraries, which aren't available in headless (no-display) environments like Docker, Streamlit Cloud, or Heroku.

**Solution:** Use `opencv-python-headless` instead of `opencv-python`.

---

## Quick Fix for Your Current Deployment

### If Using Streamlit Cloud:
1. Delete `requirements.txt` from your repo (or update it)
2. Ensure you're using `opencv-python-headless>=4.8.0`
3. Redeploy

### If Using Docker:
Update your Dockerfile to include headless OpenCV in `requirements.txt`:
```
opencv-python-headless>=4.8.0
```

---

## Solution: OpenCV Version Handling

### Local Development (Windows/macOS with display)
Use `requirements-dev.txt`:
```bash
pip install -r requirements-dev.txt
```
This includes `opencv-python` for full graphical support.

### Server Deployment (Linux, Docker, Cloud)
Use default `requirements.txt`:
```bash
pip install -r requirements.txt
```
This includes `opencv-python-headless` for headless environments.

---

## Docker Deployment

### Option 1: Build and Run Locally

```bash
# Build Docker image
docker build -t dr-classifier:latest .

# Run container
docker run -p 8501:8501 dr-classifier:latest

# Access at: http://localhost:8501
```

### Option 2: Docker Compose (Recommended)

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  dr-classifier:
    build: .
    ports:
      - "8501:8501"
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
      - CONFIDENCE_THRESHOLD=0.6
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
    restart: always
```

Run with:
```bash
docker-compose up -d
```

---

## Cloud Deployment

### 1. Streamlit Cloud (Easiest)

**Steps:**
1. Push code to GitHub
2. Go to https://share.streamlit.io/
3. Sign in with GitHub
4. Paste repository URL
5. Select branch and `app.py`
6. Click Deploy

**Note:** Streamlit Cloud automatically uses headless OpenCV. No changes needed.

**Configuration file** (`streamlit/config.toml`):
```toml
[server]
port = 8501
enableXsrfProtection = false

[client]
toolbarMode = "minimal"

[logger]
level = "info"
```

### 2. Heroku (Docker Deployment)

**Setup:**
```bash
# Install Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli

# Login
heroku login

# Create app
heroku create dr-classifier

# Set buildpack
heroku buildpacks:set heroku/docker

# Deploy
git push heroku master

# View logs
heroku logs --tail
```

**For Heroku, add `Procfile`:**
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

### 3. AWS (ECS/Fargate)

**Push to ECR (Elastic Container Registry):**
```bash
# Create ECR repository
aws ecr create-repository --repository-name dr-classifier

# Build and push
docker build -t dr-classifier:latest .
docker tag dr-classifier:latest <account-id>.dkr.ecr.<region>.amazonaws.com/dr-classifier:latest
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/dr-classifier:latest

# Launch ECS task with this image
```

### 4. Google Cloud Run

**Deploy:**
```bash
# Build
gcloud builds submit --tag gcr.io/<project-id>/dr-classifier

# Deploy
gcloud run deploy dr-classifier \
  --image gcr.io/<project-id>/dr-classifier \
  --platform managed \
  --region us-central1 \
  --port 8501 \
  --memory 2Gi \
  --cpu 2

# Access
gcloud run services describe dr-classifier
```

### 5. Azure Container Instances

**Push to ACR (Azure Container Registry):**
```bash
# Create ACR
az acr create --resource-group mygroup --name drclassifier --sku Basic

# Build and push
az acr build --registry drclassifier --image dr-classifier:latest .

# Deploy
az container create \
  --resource-group mygroup \
  --name dr-classifier \
  --image drclassifier.azurecr.io/dr-classifier:latest \
  --ports 8501 \
  --environment-variables PORT=8501
```

### 6. DigitalOcean App Platform

Create `app.yaml`:
```yaml
name: dr-classifier
services:
- name: web
  github:
    repo: username/Opthamology_sample_inference
    branch: master
  build_command: pip install -r requirements.txt
  run_command: streamlit run app.py --server.port 8080 --server.address 0.0.0.0
  http_port: 8080
```

---

## Environment Variables for Deployment

Set these in your deployment platform:

```env
# Model
MODEL_PATH=/app/pretrained/dr_mobilenetv2_5class.pth

# Data
DATA_DIR=/app/data/images
OUTPUT_DIR=/app/outputs

# Inference
CONFIDENCE_THRESHOLD=0.6
BLUR_THRESHOLD=100.0
MIN_IMAGE_SIZE=256
BATCH_SIZE=16

# Logging
LOG_LEVEL=INFO

# CAM
DEFAULT_CAM_METHOD=GradCAM
CAM_ALPHA=0.4

# Streamlit
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

---

## Performance Optimization for Deployment

### 1. Reduce Image Size
```dockerfile
# Use slim base image (current Dockerfile already does this)
FROM python:3.11-slim
```

### 2. Use Model Quantization
```python
# In src/model.py, add quantization:
import torch.quantization as q
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
# Reduces model size from 14MB → 3.5MB
```

### 3. Enable Caching
```python
# In app.py, Streamlit caching is already enabled
@st.cache_resource
def load_model():
    ...
```

### 4. Minimize Dependencies
Try `requirements-minimal.txt`:
```
torch
torchvision
grad-cam
numpy
pillow
opencv-python-headless
pandas
streamlit
python-dotenv
```

---

## Troubleshooting Deployment Errors

### Error: libGL.so.1 not found
**Solution:** Ensure `opencv-python-headless` is in requirements.txt (not `opencv-python`)

### Error: CUDA not available
**Solution:** Normal for CPU deployments. Model works fine on CPU.

### Error: Out of memory
**Solution:** Reduce batch size or use smaller model. In `.env`:
```
BATCH_SIZE=8
```

### Error: Slow inference
**Solutions:**
1. Use GPU instance (AWS g4dn, Azure GPU)
2. Use model quantization
3. Enable caching

### Streamlit Cloud takes too long to load
**Solution:**
1. Clear Streamlit cache
2. Pin specific dependency versions
3. Use lighter dependencies

---

## Monitoring & Logging

### Docker Logs
```bash
# View logs
docker logs <container-id>

# Follow logs
docker logs -f <container-id>

# Save logs
docker logs <container-id> > app.log
```

### Streamlit Cloud Logs
- View in Streamlit Cloud dashboard
- Check GitHub Actions for deployment issues

### Cloud Provider Logging
- **AWS:** CloudWatch
- **Google Cloud:** Cloud Logging
- **Azure:** Monitor
- **DigitalOcean:** Application Logs

---

## Production Checklist

- [ ] Using `opencv-python-headless`
- [ ] Environment variables configured
- [ ] Health checks in place
- [ ] Logging enabled
- [ ] Error handling robust
- [ ] Model quantized (optional but recommended)
- [ ] Batch processing optimized
- [ ] Database/storage configured (if needed)
- [ ] SSL/TLS enabled
- [ ] Rate limiting configured
- [ ] Secrets managed properly
- [ ] Tested with production-like data

---

## Comparison: Deployment Platforms

| Platform | Cost | Setup Time | Scaling | Best For |
|----------|------|-----------|---------|----------|
| **Streamlit Cloud** | Free | 5 min | Auto | Demos, prototypes |
| **Heroku** | $5-50/mo | 15 min | Manual | Small apps |
| **AWS** | $10-100/mo | 30 min | Auto | Production, enterprise |
| **Google Cloud Run** | Pay-per-use | 20 min | Auto | Scale-on-demand |
| **Azure** | $10-100/mo | 25 min | Auto | Enterprise, Microsoft stack |
| **Docker Hub + VPS** | $5-20/mo | 20 min | Manual | Control, customization |

---

## Quick Start Templates

### Docker
```bash
docker build -t dr-classifier .
docker run -p 8501:8501 dr-classifier
```

### Docker Compose
```bash
docker-compose up -d
```

### Streamlit Cloud
1. Push to GitHub
2. Visit https://share.streamlit.io/
3. Deploy in 2 clicks

### Heroku
```bash
heroku login
heroku create dr-classifier
git push heroku master
```

---

## Next Steps

1. **Choose platform** based on your needs
2. **Update requirements.txt** (use headless OpenCV)
3. **Set environment variables**
4. **Deploy and test**
5. **Monitor logs**
6. **Scale as needed**

---

## Support

- **Streamlit Cloud Help:** https://docs.streamlit.io/streamlit-cloud
- **Docker Docs:** https://docs.docker.com/
- **Cloud Provider Docs:** See links above
- **Project Issues:** https://github.com/Charansaiputta13/Opthamology_sample_inference/issues

---

**Status:** Ready for production deployment  
**Framework:** PyTorch 2.1+  
**OpenCV:** Headless version for servers  
**Last Updated:** 2025-03-03

