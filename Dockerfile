# syntax=docker/dockerfile:1.4
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# ---------- Runtime env (Cloud Run-friendly) ----------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Writable caches at runtime
    XDG_CACHE_HOME=/tmp \
    PIP_CACHE_DIR=/tmp/pip \
    HF_HOME=/tmp/hf \
    MPLCONFIGDIR=/tmp/mpl \
    # NLTK data lives in the image (read-only at runtime, that's fine)
    NLTK_DATA=/usr/local/nltk_data

WORKDIR /app

# ---------- System deps ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv \
    build-essential curl ca-certificates git \
    libmagic1 poppler-utils tesseract-ocr \
 && rm -rf /var/lib/apt/lists/* \
 && ln -s /usr/bin/python3.10 /usr/bin/python

# ---------- Python deps ----------
# (Keep requirements.txt minimal & pinned for reproducible builds)
COPY requirements.txt .
# Install PyTorch with CUDA support first, then other requirements
RUN pip3 install --upgrade pip && \
    pip3 install --extra-index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.0+cu121 torchvision==0.19.0+cu121 && \
    pip3 install --no-cache-dir -r requirements.txt

# If marker/unstructured/nltk aren't in requirements.txt, install here:
RUN pip3 install --no-cache-dir \
    "transformers>=4.36.0" \
    "marker-pdf[gpu]==1.9.2" \
    "unstructured>=0.15.7" \
    "nltk>=3.9.0" \
    "uvicorn[standard]>=0.30.0"

# ---------- Pre-fetch runtime assets at BUILD time ----------
# NLTK packs (avoid runtime downloads on read-only FS)
RUN python3 -c "import os, nltk; os.makedirs('/usr/local/nltk_data', exist_ok=True); [nltk.download(p, download_dir='/usr/local/nltk_data', quiet=True) for p in ('punkt','punkt_tab','averaged_perceptron_tagger')]"

# Pre-download Marker font and models for GPU usage
RUN python3 -c "from marker.util import download_font; download_font(); print('Marker font pre-downloaded.')" && \
    python3 -c "from marker.models import load_all_models; load_all_models(); print('Marker models pre-downloaded.')"

# ---------- App files & non-root user ----------
# (Copy code after deps to leverage Docker layer caching)
COPY . .

# Create an unprivileged user and fix ownership
# Also create marker_output directory and cache directories with proper permissions
RUN useradd --create-home --shell /bin/bash app \
 && mkdir -p /app/marker_output /tmp/marker_output /home/app/.cache/datalab /home/app/.cache/surya \
 && chown -R app:app /app /tmp/marker_output /home/app \
 && chmod -R 777 /app/marker_output /tmp/marker_output /home/app/.cache


EXPOSE 8001


CMD bash -c "mkdir -p /app/marker_output /home/app/.cache/datalab /home/app/.cache/surya && chmod -R 777 /app/marker_output /home/app/.cache && chown -R app:app /app/marker_output /home/app && echo 'Checking CUDA availability...' && python3 -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"CUDA device count: {torch.cuda.device_count()}\"); print(f\"Current device: {torch.cuda.current_device() if torch.cuda.is_available() else None}\")' && su - app -c 'cd /app && CUDA_VISIBLE_DEVICES=0 python3 -m uvicorn api_server:app --host 0.0.0.0 --port 8001'"
