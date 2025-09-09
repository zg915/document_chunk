# Google Cloud Run Deployment Guide

This guide will help you deploy the Document Processing API to Google Cloud Run.

## Prerequisites

1. **Google Cloud Account**: Sign up at [Google Cloud Console](https://console.cloud.google.com/)
2. **Google Cloud CLI**: Install [gcloud CLI](https://cloud.google.com/sdk/docs/install)
3. **Docker**: Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
4. **Project Setup**: Create a new GCP project or use an existing one

## Quick Start

### 1. Authentication
```bash
# Login to Google Cloud
gcloud auth login

# Set your project ID
gcloud config set project YOUR_PROJECT_ID
```

### 2. Deploy with Script
```bash
# Make script executable (Linux/Mac)
chmod +x deploy.sh

# Deploy to Cloud Run
./deploy.sh YOUR_PROJECT_ID us-central1
```

### 3. Manual Deployment

#### Step 1: Enable APIs
```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable secretmanager.googleapis.com
gcloud services enable monitoring.googleapis.com
gcloud services enable logging.googleapis.com
```

#### Step 2: Create Secrets
```bash
# Create Weaviate configuration secrets
echo "your-weaviate-url" | gcloud secrets create weaviate-url --data-file=-
echo "your-weaviate-api-key" | gcloud secrets create weaviate-api-key --data-file=-
echo "your-marker-api-key" | gcloud secrets create marker-api-key --data-file=-
```

#### Step 3: Create Service Account
```bash
# Create service account
gcloud iam service-accounts create document-processing-api \
    --display-name="Document Processing API Service Account"

# Grant permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:document-processing-api@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:document-processing-api@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/monitoring.metricWriter"
```

#### Step 4: Build and Deploy
```bash
# Build and push image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/document-processing-api .

# Deploy to Cloud Run
gcloud run deploy document-processing-api \
    --image gcr.io/YOUR_PROJECT_ID/document-processing-api \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --service-account document-processing-api@YOUR_PROJECT_ID.iam.gserviceaccount.com \
    --set-env-vars "PORT=8001,USE_LOCAL_MARKER=false,LOG_LEVEL=INFO" \
    --set-secrets "WEAVIATE_URL=weaviate-url:latest,WEAVIATE_API_KEY=weaviate-api-key:latest,MARKER_API_KEY=marker-api-key:latest" \
    --cpu 2 \
    --memory 4Gi \
    --max-instances 10 \
    --min-instances 0 \
    --timeout 3600 \
    --concurrency 5
```

## Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `PORT` | Server port | No | 8001 |
| `USE_LOCAL_MARKER` | Use local Marker | No | false |
| `LOG_LEVEL` | Logging level | No | INFO |
| `ALLOWED_ORIGINS` | CORS origins | No | * |

### Secrets (Managed by Secret Manager)

| Secret | Description | Required |
|--------|-------------|----------|
| `weaviate-url` | Weaviate database URL | Yes |
| `weaviate-api-key` | Weaviate API key | Yes |
| `marker-api-key` | Marker API key | Yes |

## Monitoring and Logging

### Cloud Logging
- All application logs are automatically sent to Cloud Logging
- Structured logging with JSON format
- Log levels: INFO, WARNING, ERROR

### Cloud Monitoring
- Custom metrics for document processing
- API request metrics
- Error tracking
- Performance monitoring

### Health Checks
- Endpoint: `GET /health`
- Checks Weaviate connection
- Returns service status

## API Endpoints

### Document Processing
- `POST /convert-to-markdown-upload` - Convert uploaded file to markdown
- `POST /process-document-upload` - Process uploaded file to Weaviate
- `POST /upload-and-process` - Upload and process file directly

### Document Management
- `GET /documents` - List documents
- `GET /document/{document_id}` - Get document info
- `DELETE /delete-document` - Delete document

### System
- `GET /health` - Health check
- `GET /docs` - API documentation (Swagger UI)
- `GET /redoc` - API documentation (ReDoc)

## Scaling Configuration

### Current Settings
- **CPU**: 2 vCPU
- **Memory**: 4 GiB
- **Max Instances**: 10
- **Min Instances**: 0
- **Concurrency**: 5 requests per instance
- **Timeout**: 3600 seconds (1 hour)

### Adjusting Resources
```bash
gcloud run services update document-processing-api \
    --region us-central1 \
    --cpu 4 \
    --memory 8Gi \
    --max-instances 20
```

## Security

### Network Security
- HTTPS only (automatic with Cloud Run)
- CORS configured for specific origins
- Trusted host middleware

### Access Control
- Service account with minimal permissions
- Secrets managed by Secret Manager
- No hardcoded credentials

### File Upload Security
- File type validation
- Temporary file cleanup
- Size limits (configured at Cloud Run level)

## Troubleshooting

### Common Issues

1. **Service won't start**
   - Check secrets are properly configured
   - Verify service account permissions
   - Check Cloud Run logs

2. **Weaviate connection failed**
   - Verify Weaviate URL and API key
   - Check network connectivity
   - Ensure Weaviate is accessible from Cloud Run

3. **High memory usage**
   - Increase memory allocation
   - Optimize file processing
   - Consider file size limits

### Viewing Logs
```bash
# View recent logs
gcloud logs read "resource.type=cloud_run_revision AND resource.labels.service_name=document-processing-api" --limit 50

# Follow logs in real-time
gcloud logs tail "resource.type=cloud_run_revision AND resource.labels.service_name=document-processing-api"
```

### Monitoring
```bash
# View service status
gcloud run services describe document-processing-api --region us-central1

# View metrics
gcloud monitoring metrics list --filter="resource.type=cloud_run_revision"
```

## Cost Optimization

### Recommendations
1. **Min Instances**: Keep at 0 for cost savings
2. **CPU/Memory**: Start with 2 CPU, 4GiB memory
3. **Timeout**: Set appropriate timeout based on use case
4. **Concurrency**: Adjust based on processing requirements

### Estimated Costs (us-central1)
- **CPU**: ~$0.00002400 per vCPU-second
- **Memory**: ~$0.00000250 per GiB-second
- **Requests**: $0.40 per million requests
- **Networking**: $0.12 per GB egress

## Updates and Maintenance

### Updating the Service
```bash
# Build new image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/document-processing-api .

# Deploy update
gcloud run deploy document-processing-api \
    --image gcr.io/YOUR_PROJECT_ID/document-processing-api \
    --region us-central1
```

### Rolling Back
```bash
# List revisions
gcloud run revisions list --service document-processing-api --region us-central1

# Rollback to previous revision
gcloud run services update-traffic document-processing-api \
    --to-revisions=REVISION_NAME=100 \
    --region us-central1
```

## Support

For issues and questions:
1. Check Cloud Run logs
2. Review monitoring metrics
3. Verify configuration
4. Check Google Cloud documentation

## Next Steps

1. Set up custom domain (optional)
2. Configure load balancing (if needed)
3. Set up CI/CD pipeline
4. Implement additional monitoring
5. Add authentication/authorization
