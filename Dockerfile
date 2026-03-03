# Multi-stage Dockerfile for Diabetic Retinopathy Classification System
# Optimized for Streamlit Cloud, Docker, and production deployments

# Stage 1: Base image with minimal dependencies
FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies needed for image processing
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Build stage with all dependencies
FROM base as builder

# Copy requirements
COPY requirements.txt .

# Install Python dependencies with headless OpenCV
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 3: Runtime stage (minimal image)
FROM base

# Copy only necessary files from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Set working directory
WORKDIR /app

# Copy application files
COPY src/ src/
COPY pretrained/ pretrained/
COPY app.py .
COPY .env.example .env

# Create output directories
RUN mkdir -p outputs/logs outputs/predictions outputs/eval data/images

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
