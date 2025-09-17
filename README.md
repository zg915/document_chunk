# Document Processing API with Webhook Support

A FastAPI-based document processing service that converts PDFs and images to markdown using webhook-based asynchronous processing with datalab.to integration.

## 🚀 Features

- **Webhook-based Processing**: Real-time document conversion without polling
- **Multiple Formats**: Support for PDF, images, Word, and Excel files
- **Asynchronous**: Non-blocking API responses using asyncio
- **Docker Ready**: Containerized with Docker Compose
- **Cloud Deployable**: Ready for Google Cloud Platform deployment
- **Webhook Service**: Node.js webhook receiver for real-time callbacks

## 🏗️ Architecture

```
Client → API Server → Datalab.to → Webhook Service → API Server → Client
```

1. **API Server** (FastAPI): Handles file uploads and webhook callbacks
2. **Webhook Service** (Node.js): Receives datalab notifications and fetches content
3. **Datalab.to**: External service for document processing

## 📋 Prerequisites

- Docker and Docker Compose
- Google Cloud Platform account (for deployment)
- Datalab.to API key

## 🛠️ 10-Step Setup Guide

### Step 1: Clone the Repository
```bash
git clone <your-repo-url>
cd document_chunk
```

### Step 2: Set Up Environment Variables
```bash
cp env.example .env
# Edit .env with your API keys
```

Required environment variables:
```env
MARKER_API_KEY=your_datalab_api_key
MARKER_API_URL=https://www.datalab.to/api/v1/marker
WEAVIATE_API_KEY=your_weaviate_key
WEAVIATE_URL=your_weaviate_url
GOOGLE_API_KEY=your_google_api_key
```

### Step 3: Build Docker Images
```bash
docker compose build
```

### Step 4: Start Services Locally
```bash
docker compose up -d
```

### Step 5: Verify Services are Running
```bash
# Check API health
curl http://localhost:8001/health

# Check webhook health
curl http://localhost:3000/health
```

### Step 6: Test Webhook-based Conversion
```bash
curl -X POST "http://localhost:8001/convert-doc-to-markdown-webhook?webhook_url=http://localhost:3000/webhook" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your-document.pdf"
```

### Step 7: Deploy to Google Cloud Platform
```bash
# Create GCP VM instance
gcloud compute instances create document-process-vm \
  --zone=us-central1-a \
  --machine-type=e2-medium \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud

# SSH into VM
gcloud compute ssh document-process-vm --zone=us-central1-a
```

### Step 8: Set Up Firewall Rules
In GCP Console:
1. Go to VPC Network → Firewall
2. Create firewall rule:
   - **Name**: `allow-api-8001`
   - **Targets**: `api-8001`
   - **Protocols/Ports**: `tcp:3000,8001`
   - **Action**: Allow

### Step 9: Deploy on GCP VM
```bash
# On the VM
sudo apt update
sudo apt install -y docker.io docker-compose git

# Clone repository
git clone <your-repo-url>
cd document_chunk

# Set environment variables
export MARKER_API_KEY="your_api_key"
export WEAVIATE_API_KEY="your_weaviate_key"
export WEAVIATE_URL="your_weaviate_url"
export GOOGLE_API_KEY="your_google_key"

# Start services
docker compose up -d
```

### Step 10: Test Production Deployment
```bash
# Get VM external IP
gcloud compute instances describe document-process-vm \
  --zone=us-central1-a \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)'

# Test API (replace with your VM IP)
curl -X POST "http://YOUR_VM_IP:8001/convert-doc-to-markdown-webhook?webhook_url=http://YOUR_VM_IP:3000/webhook" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test-document.pdf"
```

## 🔄 How Webhook Processing Works

### 1. **File Upload**
- Client uploads PDF/image to API
- API generates unique request ID
- Creates asyncio.Future for waiting

### 2. **Datalab Upload**
- API uploads file to datalab.to with webhook URL
- Datalab returns its own request ID
- API updates pending requests mapping

### 3. **Asynchronous Waiting**
- API waits for webhook callback (5-minute timeout)
- Client sees "loading" state during processing

### 4. **Datalab Processing**
- Datalab processes the document
- When complete, sends webhook to your webhook service

### 5. **Webhook Reception**
- Webhook service receives datalab notification
- Extracts request ID from notification

### 6. **Content Fetching**
- Webhook fetches actual content from datalab API
- Extracts markdown from JSON response

### 7. **Callback to API**
- Webhook sends extracted content back to API
- Uses correct request ID for matching

### 8. **Result Delivery**
- API receives webhook callback
- Sets result in asyncio.Future
- API returns markdown content to client

## 📚 API Endpoints

### Webhook-based Conversion
```http
POST /convert-doc-to-markdown-webhook
```
**Parameters:**
- `webhook_url` (required): URL for webhook callbacks
- `file` (required): PDF or image file

### Regular Conversion
```http
POST /convert-doc-to-markdown
```
**Parameters:**
- `use_webhook` (optional): Enable webhook processing
- `webhook_url` (optional): Webhook URL when use_webhook=true
- `file` (required): PDF or image file

### Health Checks
```http
GET /health                    # API health
GET http://webhook:3000/health # Webhook health
```

## 🐳 Docker Services

### API Service (Port 8001)
- **Image**: Python FastAPI application
- **Dependencies**: marker-pdf, unstructured, weaviate
- **Environment**: API keys, webhook URL

### Webhook Service (Port 3000)
- **Image**: Node.js Express application
- **Dependencies**: axios, cheerio, undici
- **Function**: Receives datalab webhooks, fetches content

## 🔧 Configuration

### Environment Variables
```env
# Datalab Configuration
MARKER_API_KEY=your_datalab_api_key
MARKER_API_URL=https://www.datalab.to/api/v1/marker

# Weaviate Configuration
WEAVIATE_API_KEY=your_weaviate_key
WEAVIATE_URL=your_weaviate_url

# Google Services
GOOGLE_API_KEY=your_google_api_key

# Webhook Configuration
WEBHOOK_URL=http://webhook:3000/webhook
API_CALLBACK_URL=http://document-api:8001/api-callback
```

### Docker Compose
```yaml
services:
  document-api:
    build: .
    ports:
      - "8001:8001"
    environment:
      - WEBHOOK_URL=http://webhook:3000/webhook
    depends_on:
      - webhook

  webhook:
    build: ./webhook
    ports:
      - "3000:3000"
    environment:
      - API_CALLBACK_URL=http://document-api:8001/api-callback
```

## 🚨 Troubleshooting

### Common Issues

1. **Webhook not receiving callbacks**
   - Check firewall rules allow port 3000
   - Verify webhook URL is accessible externally
   - Check webhook service logs

2. **Request ID mismatch**
   - API generates UUID, datalab returns different ID
   - Fixed by updating pending_webhook_requests mapping

3. **File class error in webhook**
   - Node.js 18 compatibility issue with undici
   - Fixed by adding global File class mock

4. **Timeout errors**
   - Increase timeout in asyncio.wait_for()
   - Check datalab processing time

### Logs
```bash
# View all logs
docker compose logs -f

# View specific service logs
docker compose logs -f document-api
docker compose logs -f webhook
```

## 📝 Supported File Formats

- **PDF**: `.pdf`
- **Images**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`
- **Word**: `.doc`, `.docx`
- **Excel**: `.xls`, `.xlsx`

## 🔒 Security

- API keys stored in environment variables
- Webhook endpoints require proper authentication
- File validation before processing
- Request timeout protection

## 📈 Performance

- **Asynchronous processing**: Non-blocking API responses
- **Webhook-based**: No polling required
- **Docker optimized**: Efficient containerization
- **Cloud-ready**: Scalable deployment

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check troubleshooting section
2. Review logs for errors
3. Create GitHub issue with details
4. Include relevant log snippets

---

**Ready to process documents with webhook-based real-time conversion!** 🎉
