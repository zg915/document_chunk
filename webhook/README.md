# Webhook Service

This webhook service receives notifications from datalab when document processing is complete and automatically extracts the markdown content.

## Features

- Receives webhook notifications from datalab
- Automatically extracts markdown content from webhook payloads
- Sends callback to API server with extracted content
- Handles datalab.to URLs with automatic API key injection
- Supports multiple content types (JSON, HTML, text, etc.)

## API Endpoints

### `POST /webhook`
Main webhook endpoint that receives notifications from datalab.

**Request Body:**
```json
{
  "status": "completed",
  "markdown": "# Document content...",
  "request_id": "uuid-here"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Webhook received successfully",
  "webhookId": "timestamp",
  "urlsFound": 0
}
```

### `GET /webhooks`
View all received webhooks and extracted data.

### `GET /`
Health check endpoint.

## Environment Variables

- `PORT`: Server port (default: 3000)
- `API_CALLBACK_URL`: URL to send callbacks to API server (default: http://document-api:8001/api-callback)

## Usage with Docker

The webhook service is automatically started with the main application using Docker Compose:

```bash
docker-compose up
```

This will start both the API server and webhook service.

## Webhook Flow

1. Client uploads document to API server
2. API server uploads file to datalab with webhook URL
3. Datalab processes document and sends webhook notification
4. Webhook service receives notification and extracts content
5. Webhook service sends callback to API server with content
6. API server returns content to client

## Configuration

The webhook URL should be configured in your datalab requests:

```python
webhook_url = "https://your-app.onrender.com:3000/webhook"
```

When deployed on Render, both services will be accessible:
- API Server: `https://your-app.onrender.com:8001`
- Webhook Service: `https://your-app.onrender.com:3000/webhook`