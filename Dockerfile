# syntax=docker/dockerfile:1.4
FROM python:3.10-slim

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
    build-essential curl ca-certificates \
    libmagic1 poppler-utils \
 && rm -rf /var/lib/apt/lists/*

# ---------- Python deps ----------
# (Keep requirements.txt minimal & pinned for reproducible builds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# If marker/unstructured/nltk aren't in requirements.txt, install here:
RUN pip install --no-cache-dir \
    "marker-pdf==1.9.2" \
    "unstructured>=0.15.7" \
    "nltk>=3.9.0" \
    "uvicorn[standard]>=0.30.0"

# ---------- Pre-fetch runtime assets at BUILD time ----------
# NLTK packs (avoid runtime downloads on read-only FS)
RUN python -c "import os, nltk; os.makedirs('/usr/local/nltk_data', exist_ok=True); [nltk.download(p, download_dir='/usr/local/nltk_data', quiet=True) for p in ('punkt','punkt_tab','averaged_perceptron_tagger')]"

# Pre-download Marker font so it doesn't try writing into site-packages at runtime
RUN python -c "from marker.util import download_font; download_font(); print('Marker font pre-downloaded.')"

# ---------- App files & non-root user ----------
# (Copy code after deps to leverage Docker layer caching)
COPY . .

# Create an unprivileged user and fix ownership
RUN useradd --create-home --shell /bin/bash app \
 && chown -R app:app /app
USER app

# Cloud Run exposes exactly one container port; your app must bind 0.0.0.0
EXPOSE 8001

# ---------- Start command ----------
# Replace "api_server:app" with your actual ASGI app import path
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8001"]
