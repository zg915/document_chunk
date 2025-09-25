# syntax=docker/dockerfile:1.4
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# ---------- Runtime env ----------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    XDG_CACHE_HOME=/tmp \
    PIP_CACHE_DIR=/tmp/pip \
    HF_HOME=/tmp/hf \
    MPLCONFIGDIR=/tmp/mpl \
    NLTK_DATA=/usr/local/nltk_data \
    CUDA_VISIBLE_DEVICES=0 \
    TORCH_DEVICE=cuda \
    MARKER_DEVICE=cuda \
    SURYA_DEVICE=cuda \
    TORCH_CUDNN_V8_API_ENABLED=1 \
    TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 \
    CUDA_LAUNCH_BLOCKING=0

WORKDIR /app


# ---------- System deps (rarely changes) ----------
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv \
    build-essential curl ca-certificates git \
    libmagic1 poppler-utils tesseract-ocr \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    # WeasyPrint dependencies
    libpango-1.0-0 libharfbuzz0b libpangoft2-1.0-0 \
    # LibreOffice for document conversion (required for .doc, .ppt, etc.)
    libreoffice \
    # Additional libraries for image processing
    libcairo2-dev pkg-config python3-dev \
 && ln -s /usr/bin/python3.10 /usr/bin/python


# ---------- Python deps ----------
COPY requirements.txt .
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt && \
    # Install marker-pdf separately to handle potential conflicts
    (pip3 install marker-pdf --no-deps --force-reinstall || echo "Marker installation skipped") && \
    pip3 install torch torchvision transformers --upgrade && \
    # Verify installation
    (python3 -c "import marker; print('Marker imported successfully')" || echo "Marker import failed") && \
    (python3 -c "from marker.converters.pdf import PdfConverter; print('PdfConverter available')" || echo "PdfConverter failed") && \
    # Verify critical packages
    (python3 -c "import torch; print('PyTorch installed')" || echo "PyTorch failed") && \
    (python3 -c "import uvicorn; print('Uvicorn installed')" || echo "Uvicorn failed") && \
    (python3 -c "import fastapi; print('FastAPI installed')" || echo "FastAPI failed")

# ---------- Pre-fetch runtime assets ----------
# Install NLTK if not already installed and download required data
RUN pip3 install nltk && \
    python3 -c "import os, nltk; os.makedirs('/usr/local/nltk_data', exist_ok=True); [nltk.download(p, download_dir='/usr/local/nltk_data', quiet=True) for p in ('punkt','punkt_tab','averaged_perceptron_tagger')]" && \
    (python3 -c "from marker.util import download_font; download_font(); print('Marker font pre-downloaded.')" || echo "Marker font download skipped")

# ---------- GPU optimization environment variables ----------
ENV MARKER_NUM_WORKERS=4 \
    MARKER_PAGE_BATCH=6 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    TORCH_CUDNN_V8_API_ENABLED=1 \
    TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 \
    INFERENCE_RAM=16 \
    NUM_DEVICES=1 \
    NUM_WORKERS=4 \
    VRAM_PER_TASK=4

# ---------- App files (changes frequently - keep last) ----------
COPY . .

# Create an unprivileged user and fix ownership
# Also create marker_output directory and cache directories with proper permissions
RUN useradd --create-home --shell /bin/bash app \
 && mkdir -p /app/marker_output /tmp/marker_output /home/app/.cache/datalab /home/app/.cache/surya \
 && chown -R app:app /app /tmp/marker_output /home/app \
 && chmod -R 777 /app/marker_output /tmp/marker_output /home/app/.cache


EXPOSE 8001


CMD ["bash", "-c", "mkdir -p /app/marker_output /home/app/.cache/datalab /home/app/.cache/surya && chmod -R 777 /app/marker_output /home/app/.cache && echo 'Starting API server...' && python3 preload_models.py && python3 -m uvicorn api_server:app --host 0.0.0.0 --port 8001"]
