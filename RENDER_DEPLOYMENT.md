# Render Deployment Guide

## Prerequisites

1. **Render Account** - Sign up at [render.com](https://render.com)
2. **GitHub Repository** - Push your code to GitHub
3. **Environment Variables** - Set up required environment variables

## Deployment Steps

### 1. Push Code to GitHub

```bash
git add .
git commit -m "Add webhook-based document processing"
git push origin main
```

### 2. Create New Web Service on Render

1. Go to [render.com/dashboard](https://render.com/dashboard)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Select the repository with your code

### 3. Configure Service Settings

**Basic Settings:**
- **Name**: `document-chunk-api` (or your preferred name)
- **Runtime**: `Docker`
- **Build Command**: `docker-compose -f docker-compose.yml -f docker-compose.render.yml build`
- **Start Command**: `docker-compose -f docker-compose.yml -f docker-compose.render.yml up`
- **Dockerfile Path**: Leave empty (using docker-compose)

**Advanced Settings:**
- **Docker Context Directory**: `document_chunk`
- **Docker Compose File**: `docker-compose.yml,docker-compose.render.yml`

### 4. Set Environment Variables

In Render dashboard, go to Environment tab and add:

```
MARKER_API_KEY=your_marker_api_key
MARKER_API_URL=https://www.datalab.to/api/v1/marker
USE_LOCAL_MARKER=false
WEAVIATE_API_KEY=your_weaviate_api_key
WEAVIATE_URL=your_weaviate_url
GOOGLE_API_KEY=your_google_api_key
RENDER_SERVICE_NAME=your_service_name
```

### 5. Deploy

1. Click "Create Web Service"
2. Render will build and deploy both containers
3. Wait for deployment to complete (5-10 minutes)

## Service URLs

After deployment, you'll get:

- **API Server**: `https://your-service-name.onrender.com:8001`
- **Webhook Service**: `https://your-service-name.onrender.com:3000/webhook`

## Testing the Deployment

### Test API Health
```bash
curl https://your-service-name.onrender.com:8001/health
```

### Test Webhook
```bash
curl https://your-service-name.onrender.com:3000/
```

### Test Document Conversion
```bash
curl -X POST "https://your-service-name.onrender.com:8001/convert-doc-to-markdown-webhook?webhook_url=https://your-service-name.onrender.com:3000/webhook" \
  -F "file=@your-document.pdf"
```

## Troubleshooting

### Common Issues:

1. **Build Fails**: Check Docker logs in Render dashboard
2. **Port Issues**: Ensure services use `PORT` environment variable
3. **Webhook URL**: Verify webhook URL is correctly configured
4. **Memory Limits**: Render free tier has 512MB RAM limit

### Logs:
- View logs in Render dashboard under "Logs" tab
- Both services will show logs in the same view

## Environment Variables Reference

| Variable | Description | Required |
|----------|-------------|----------|
| `MARKER_API_KEY` | Datalab API key | Yes |
| `MARKER_API_URL` | Datalab API endpoint | Yes |
| `USE_LOCAL_MARKER` | Use local vs API processing | No (default: false) |
| `WEAVIATE_API_KEY` | Weaviate database API key | Yes |
| `WEAVIATE_URL` | Weaviate database URL | Yes |
| `GOOGLE_API_KEY` | Google API key for AI features | No |
| `RENDER_SERVICE_NAME` | Your Render service name | Yes |

## Scaling

- **Free Tier**: 1 instance, 512MB RAM
- **Paid Plans**: Multiple instances, more RAM
- **Auto-scaling**: Available on paid plans

## Monitoring

- **Health Checks**: Built-in health endpoints
- **Logs**: Real-time logs in dashboard
- **Metrics**: CPU, memory usage in dashboard
