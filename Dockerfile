# syntax=docker/dockerfile:1.4
FROM python:3.10-slim

# ---------- Runtime env ----------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    XDG_CACHE_HOME=/tmp \
    PIP_CACHE_DIR=/tmp/pip \
    NLTK_DATA=/usr/local/nltk_data \
    UNSTRUCTURED_NLTK_DATA=/usr/local/nltk_data

WORKDIR /app


# ---------- System deps (rarely changes) ----------
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates git \
    libmagic1 poppler-utils tesseract-ocr \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*


# ---------- Python deps (use cache mount for pip) ----------
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

# ---------- Pre-fetch runtime assets ----------
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -c "import os, nltk; os.makedirs('/usr/local/nltk_data', exist_ok=True); [nltk.download(p, download_dir='/usr/local/nltk_data', quiet=True) for p in ('punkt','punkt_tab','averaged_perceptron_tagger','averaged_perceptron_tagger_eng')]" \
    && chmod -R 755 /usr/local/nltk_data

# ---------- App files (changes frequently - keep last) ----------
COPY . .

# Create an unprivileged user and fix ownership
RUN useradd --create-home --shell /bin/bash app \
 && mkdir -p /home/app/.cache \
 && chown -R app:app /app /home/app

# Switch to app user
USER app

EXPOSE 8001

CMD ["python3", "-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8001"]
