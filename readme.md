# Document Processing API

A FastAPI-based service for converting PDFs and images to markdown and storing them in Weaviate vector database.

## Quick Start

### 1. Setup Environment
```bash
# Copy environment file
cp env.example .env
# Edit .env with your API keys
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the API
```bash
python api_server.py
```

### 4. Access Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Docker

### Build and Run
```bash
# Build image
docker build -t document-processing-api .

# Run container
docker run -p 8000:8000 \
  -e WEAVIATE_URL=your_weaviate_url \
  -e WEAVIATE_API_KEY=your_api_key \
  -e MARKER_API_KEY=your_marker_key \
  document-processing-api

# Or use docker-compose
docker-compose up
```

## API Endpoints

- `POST /convert-to-markdown-upload` - Convert uploaded file to markdown
- `POST /process-document-upload` - Process uploaded file to Weaviate
- `GET /health` - Health check
- `GET /docs` - API documentation

## Environment Variables

- `WEAVIATE_URL` - Weaviate database URL
- `WEAVIATE_API_KEY` - Weaviate API key
- `MARKER_API_KEY` - Marker API key
- `USE_LOCAL_MARKER` - Use local Marker (default: false)
