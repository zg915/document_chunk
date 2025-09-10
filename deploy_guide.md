# Google Cloud Run Deployment Guide

## Creating the Project

### Via Google Cloud Console (UI)

1. Go to **Google Cloud Console** → **IAM & Admin** → **Manage Resources**
2. Click **Create Project** → Name it, set billing account → **Create**
3. Enable APIs (Cloud Run, Artifact Registry, Cloud Build) from the UI
4. Open **Cloud Shell** (bottom of the UI) to run build/deploy commands

## Deployment Steps

### 1. Prepare Your Code

**Option A: Clone from GitHub**
```bash
git clone https://github.com/YOU/YOUR-REPO.git
cd YOUR-REPO
```

**Option B: Use Existing Files**
- If files are already in Cloud Shell, navigate to that folder

### 2. Build & Push Image to Artifact Registry

```bash
REGION=us-central1
REPO=app-repo
SERVICE=document-api
IMAGE="$REGION-docker.pkg.dev/$GOOGLE_CLOUD_PROJECT/$REPO/$SERVICE:v1"

# One-time repository creation (ignore error if already exists)
gcloud artifacts repositories create "$REPO" \
  --repository-format=docker \
  --location="$REGION" || true

# Build and push image
gcloud builds submit --tag "$IMAGE"
```

### 3. Prepare Environment Variables

Create a clean environment file:
```bash
# Extract KEY=VALUE pairs from .env
grep -v '^[# ]' .env > .env.cloud
```

**Important**: Verify each line is just `KEY=VALUE` (no quotes, no `${}`)

### 4. Deploy to Cloud Run

```bash
gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --allow-unauthenticated \
  --port 8001 \
  --env-vars-file env.yaml \
  --cpu 2 \
  --memory 8Gi
```

Cloud Run will print a public URL when deployment finishes.