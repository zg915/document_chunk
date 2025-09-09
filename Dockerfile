# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8001

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install watchfiles for better file watching in Docker
RUN pip install --no-cache-dir watchfiles

# Install marker-pdf separately to avoid dependency conflicts
# Note: For Cloud Run, we'll use API mode instead of local marker
RUN pip install --no-cache-dir marker-pdf==1.9.2

# Make sure versions are compatible (works well together)
RUN pip install --no-cache-dir "unstructured>=0.15.7" "nltk>=3.9.0"

# Set where NLTK should look
ENV NLTK_DATA=/usr/local/nltk_data

# Pre-download required tokenizers into the image
RUN python - <<'PY'
import os, nltk
os.makedirs("/usr/local/nltk_data", exist_ok=True)
for pkg in ("punkt", "punkt_tab", "averaged_perceptron_tagger"):
    nltk.download(pkg, download_dir="/usr/local/nltk_data")
PY

# Pre-create marker's static directory and set permissions
RUN mkdir -p /usr/local/lib/python3.10/site-packages/static \
    && chmod 777 /usr/local/lib/python3.10/site-packages/static

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/marker_output /app/temp

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app

# Create cache directories with proper permissions before switching to non-root user
RUN mkdir -p /home/app/.cache/datalab /home/app/.cache/huggingface \
    && chown -R app:app /home/app/.cache

USER app


# Expose port (Cloud Run will override this)
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Run the application with proper Cloud Run configuration
CMD exec uvicorn api_server:app --host 0.0.0.0 --port $PORT --workers 1

