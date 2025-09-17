# Webhook Service

A Node.js Express service that receives webhook callbacks from datalab.to and processes document conversion results.

## 🚀 Purpose

This service acts as a bridge between datalab.to and the main API service, handling:
- Webhook callbacks from datalab.to
- Content extraction from datalab API
- Result forwarding to the main API

## 📋 Quick Start

### 1. Install Dependencies

```bash
npm install
```

### 2. Start Service

```bash
npm start
```

### 3. Docker (Recommended)

```bash
# From project root
docker compose up webhook
```

## 🔧 Configuration

### Environment Variables

- `PORT`: Service port (default: 3000)
- `API_CALLBACK_URL`: Main API callback URL (default: http://document-api:8001/api-callback)

### Docker Environment

```yaml
environment:
  - PORT=3000
  - API_CALLBACK_URL=http://document-api:8001/api-callback
```

## 📡 Endpoints

### POST /webhook
Receives webhook callbacks from datalab.to

**Input:**
```json
{
  "request_id": "si_K4OQB88Yxz8BKMsGvkg",
  "request_check_url": "https://www.datalab.to/api/v1/marker/si_K4OQB88Yxz8BKMsGvkg"
}
```

**Process:**
1. Detects datalab completion notification
2. Fetches content from `request_check_url`
3. Extracts markdown from JSON response
4. Sends callback to main API

### GET /health
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-17T15:30:00.000Z",
  "uptime": 3600
}
```

### GET /webhooks
List all received webhooks

**Response:**
```json
{
  "success": true,
  "count": 5,
  "webhooks": [...]
}
```

## 🔄 Processing Flow

1. **Receive Webhook**: datalab.to sends completion notification
2. **Extract Request ID**: Get datalab request ID from URL
3. **Fetch Content**: Call datalab API to get results
4. **Parse Markdown**: Extract markdown from JSON response
5. **Send Callback**: Forward results to main API
6. **Log Results**: Display extracted content in logs

## 🛠️ Development

### Local Development

```bash
# Install dependencies
npm install

# Start with auto-reload
npm run dev

# Start production
npm start
```

### Testing

```bash
# Test webhook endpoint
curl -X POST http://localhost:3000/webhook \
  -H "Content-Type: application/json" \
  -d '{"request_id":"test123","request_check_url":"https://example.com/test"}'

# Test health check
curl http://localhost:3000/health
```

## 📊 Logging

The service provides detailed logging:

- **📨 Webhook received**: Shows incoming webhook data
- **🔍 Content extraction**: Logs API calls to datalab
- **✅ Success**: Confirms successful processing
- **❌ Errors**: Shows any processing failures

### Log Levels

- **INFO**: Normal operations
- **WARN**: Non-critical issues
- **ERROR**: Processing failures

## 🔧 Troubleshooting

### Common Issues

1. **File class error**
   - Fixed with Node.js compatibility code
   - Check server.js for File class mock

2. **API callback failures**
   - Verify API_CALLBACK_URL is correct
   - Check main API service is running
   - Review network connectivity

3. **datalab API errors**
   - Check API key configuration
   - Verify request_check_url is valid
   - Review datalab.to service status

### Debug Mode

```bash
# Enable debug logging
DEBUG=* npm start
```

## 📦 Dependencies

- **express**: Web framework
- **axios**: HTTP client for API calls
- **cheerio**: HTML parsing (if needed)
- **cors**: Cross-origin requests
- **helmet**: Security headers

## 🚀 Deployment

### Docker

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 3000
CMD ["npm", "start"]
```

### Environment

```bash
# Production environment
NODE_ENV=production
PORT=3000
API_CALLBACK_URL=http://api-service:8001/api-callback
```

---

**Webhook Service - Bridging datalab.to and your API! 🌉**