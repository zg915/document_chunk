# Document Processing API with Webhook Support

A FastAPI-based document processing service that converts PDFs and images to markdown using webhook-based asynchronous processing with datalab.to.

## 🚀 Features

- **Webhook-based Processing**: Real-time document conversion without polling
- **Multiple Formats**: Support for PDF, images, Word, and Excel files
- **Asynchronous**: Non-blocking API responses
- **Docker Ready**: Easy deployment with Docker Compose
- **Cloud Deployable**: Works on Google Cloud, AWS, and other platforms

## 📋 Prerequisites

- Docker and Docker Compose
- Python 3.10+
- Node.js 18+ (for webhook service)
- datalab.to API key

## 🛠️ Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd document_chunk
```

### 2. Environment Configuration

```bash
# Copy environment template
cp env.example .env

# Edit .env with your API keys
nano .env
```

Required environment variables:
```env
MARKER_API_KEY=your_datalab_api_key
MARKER_API_URL=https://www.datalab.to/api/v1/marker
WEAVIATE_API_KEY=your_weaviate_key
WEAVIATE_URL=your_weaviate_url
GOOGLE_API_KEY=your_google_api_key
```

### 3. Start Services

```bash
# Build and start all services
docker compose up -d

# Check logs
docker compose logs -f
```

### 4. Access the API

- **API Documentation**: http://localhost:8001/docs
- **Webhook Service**: http://localhost:3000
- **Health Check**: http://localhost:3000/health

## 📡 API Endpoints

### Webhook-based Conversion (Recommended)

```bash
POST http://localhost:8001/convert-doc-to-markdown-webhook
```

**Parameters:**
- `webhook_url`: `http://your-server:3000/webhook`
- `file`: PDF or image file

**Response:**
```json
{
  "success": true,
  "markdown_content": "# Document Title\n\nContent...",
  "message": "Document converted successfully via webhook"
}
```

### Traditional Conversion

```bash
POST http://localhost:8001/convert-doc-to-markdown
```

**Parameters:**
- `use_webhook`: `true`
- `webhook_url`: `http://your-server:3000/webhook`
- `file`: PDF or image file

## 🔄 How Webhook Processing Works

1. **Upload**: Client uploads file to API
2. **Process**: API uploads file to datalab.to with webhook URL
3. **Wait**: API waits for webhook callback (up to 5 minutes)
4. **Callback**: datalab.to sends results to webhook service
5. **Extract**: Webhook fetches content and sends to API
6. **Return**: API returns markdown content to client

## 🐳 Docker Services

### API Service (Port 8001)
- **FastAPI application**
- **Document processing logic**
- **Webhook callback handling**

### Webhook Service (Port 3000)
- **Node.js Express server**
- **Receives datalab.to callbacks**
- **Fetches and processes results**

## 🌐 Deployment

### Google Cloud Platform

See [GCP_DEPLOYMENT.md](GCP_DEPLOYMENT.md) for detailed GCP deployment instructions.

### Other Platforms

1. **Update webhook URLs** in your environment
2. **Configure firewall** to allow ports 8001 and 3000
3. **Deploy with Docker Compose**

## 📝 Example Usage

### cURL Example

```bash
curl -X POST "http://localhost:8001/convert-doc-to-markdown-webhook?webhook_url=http://localhost:3000/webhook" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

### Python Example

```python
import requests

url = "http://localhost:8001/convert-doc-to-markdown-webhook"
params = {"webhook_url": "http://localhost:3000/webhook"}

with open("document.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post(url, params=params, files=files)
    
result = response.json()
print(result["markdown_content"])
```

## 🔧 Configuration

### API Configuration

- **Timeout**: 5 minutes for webhook responses
- **File Size**: 10MB limit
- **Supported Formats**: PDF, PNG, JPG, JPEG, DOC, DOCX, XLS, XLSX

### Webhook Configuration

- **Port**: 3000 (configurable via PORT env var)
- **API Callback**: http://document-api:8001/api-callback
- **Timeout**: 10 seconds for datalab API calls

## 🐛 Troubleshooting

### Common Issues

1. **Webhook not receiving callbacks**
   - Check firewall settings
   - Verify webhook URL is accessible
   - Check datalab.to API key

2. **API timeout errors**
   - Increase timeout in api_server.py
   - Check webhook service logs
   - Verify datalab.to processing status

3. **File upload errors**
   - Check file size (max 10MB)
   - Verify file format is supported
   - Check API service logs

### Logs

```bash
# API logs
docker compose logs -f document-api

# Webhook logs
docker compose logs -f webhook

# All logs
docker compose logs -f
```

## 📊 Monitoring

### Health Checks

- **API Health**: `GET http://localhost:8001/health`
- **Webhook Health**: `GET http://localhost:3000/health`
- **Webhook Status**: `GET http://localhost:3000/webhooks`

### Metrics

- **Processing Time**: Logged in API responses
- **Webhook Count**: Available via /webhooks endpoint
- **Error Rates**: Check service logs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review service logs
3. Open an issue on GitHub
4. Contact the development team

---

**Happy Document Processing! 🎉**
