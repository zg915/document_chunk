#!/bin/bash

# Google Cloud Run Deployment Script
# Usage: ./deploy.sh [PROJECT_ID] [REGION]

set -e

# Configuration
PROJECT_ID=${1:-"your-project-id"}
REGION=${2:-"us-central1"}
SERVICE_NAME="document-processing-api"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "🚀 Deploying Document Processing API to Google Cloud Run"
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI is not installed. Please install it first."
    exit 1
fi

# Check if user is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Not authenticated with gcloud. Please run 'gcloud auth login' first."
    exit 1
fi

# Set project
echo "📋 Setting project to $PROJECT_ID"
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable monitoring.googleapis.com
gcloud services enable logging.googleapis.com

# Create secrets if they don't exist
echo "🔐 Setting up secrets..."
create_secret_if_not_exists() {
    local secret_name=$1
    local description=$2
    
    if ! gcloud secrets describe $secret_name &> /dev/null; then
        echo "Creating secret: $secret_name"
        echo "placeholder" | gcloud secrets create $secret_name --data-file=- --replication-policy="automatic"
        echo "⚠️  Please update the secret value: gcloud secrets versions add $secret_name --data-file=-"
    else
        echo "Secret $secret_name already exists"
    fi
}

create_secret_if_not_exists "weaviate-url" "Weaviate database URL"
create_secret_if_not_exists "weaviate-api-key" "Weaviate API key"
create_secret_if_not_exists "marker-api-key" "Marker API key"

# Create service account
echo "👤 Creating service account..."
SERVICE_ACCOUNT_EMAIL="$SERVICE_NAME@$PROJECT_ID.iam.gserviceaccount.com"

if ! gcloud iam service-accounts describe $SERVICE_ACCOUNT_EMAIL &> /dev/null; then
    gcloud iam service-accounts create $SERVICE_NAME \
        --display-name="Document Processing API Service Account" \
        --description="Service account for Document Processing API"
    echo "Service account created: $SERVICE_ACCOUNT_EMAIL"
else
    echo "Service account already exists: $SERVICE_ACCOUNT_EMAIL"
fi

# Grant necessary permissions
echo "🔑 Granting permissions..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/secretmanager.secretAccessor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/monitoring.metricWriter"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/logging.logWriter"

# Build and push image
echo "🏗️  Building and pushing Docker image..."
gcloud builds submit --tag $IMAGE_NAME .

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --service-account $SERVICE_ACCOUNT_EMAIL \
    --set-env-vars "PORT=8000,USE_LOCAL_MARKER=false,LOG_LEVEL=INFO" \
    --set-secrets "WEAVIATE_URL=weaviate-url:latest,WEAVIATE_API_KEY=weaviate-api-key:latest,MARKER_API_KEY=marker-api-key:latest" \
    --cpu 2 \
    --memory 4Gi \
    --max-instances 10 \
    --min-instances 0 \
    --timeout 3600 \
    --concurrency 5

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")

echo "✅ Deployment completed successfully!"
echo "🌐 Service URL: $SERVICE_URL"
echo "📚 API Documentation: $SERVICE_URL/docs"
echo "❤️  Health Check: $SERVICE_URL/health"

echo ""
echo "🔧 Next steps:"
echo "1. Update your secrets with actual values:"
echo "   gcloud secrets versions add weaviate-url --data-file=-"
echo "   gcloud secrets versions add weaviate-api-key --data-file=-"
echo "   gcloud secrets versions add marker-api-key --data-file=-"
echo ""
echo "2. Test your deployment:"
echo "   curl $SERVICE_URL/health"
echo ""
echo "3. Monitor your service:"
echo "   gcloud run services describe $SERVICE_NAME --region=$REGION"
