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
    # LibreOffice for document conversion (optional but recommended)
    libreoffice \
 && ln -s /usr/bin/python3.10 /usr/bin/python


# ---------- Python deps (use cache mount for pip) ----------
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --upgrade pip && \
    pip3 install -r requirements.txt && \
    # Ensure marker CLI commands are available
    which marker || echo "marker command not found" && \
    which marker_single || echo "marker_single command not found" && \
    python3 -c "import marker; print('Marker package installed successfully')"

# ---------- Pre-fetch runtime assets ----------
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -c "import os, nltk; os.makedirs('/usr/local/nltk_data', exist_ok=True); [nltk.download(p, download_dir='/usr/local/nltk_data', quiet=True) for p in ('punkt','punkt_tab','averaged_perceptron_tagger')]" && \
    python3 -c "from marker.util import download_font; download_font(); print('Marker font pre-downloaded.')"

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


CMD bash -c "mkdir -p /app/marker_output /home/app/.cache/datalab /home/app/.cache/surya && chmod -R 777 /app/marker_output /home/app/.cache && chown -R app:app /app/marker_output /home/app && echo 'Checking CUDA availability...' && python3 -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"CUDA device count: {torch.cuda.device_count()}\"); print(f\"Current device: {torch.cuda.current_device() if torch.cuda.is_available() else None}\")' && echo 'Preloading marker models for GPU optimization...' && python3 /app/preload_models.py && echo 'Starting API server...' && su - app -c 'cd /app && CUDA_VISIBLE_DEVICES=0 python3 -m uvicorn api_server:app --host 0.0.0.0 --port 8001'"
