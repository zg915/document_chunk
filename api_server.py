"""
FastAPI server for document processing and Weaviate integration.

This API provides endpoints for:
1. Converting documents (PDF/images) to markdown
2. Processing documents and storing them in Weaviate vector database
3. Managing documents in Weaviate (delete, retrieve)

Usage:
    uvicorn api_server:app --host 0.0.0.0 --port 8001 --reload
"""

import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional
import logging
import time
import asyncio
import threading

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import our document processing functions
# Add one more import 
from integration import (
    convert_to_markdown,
    process_document_to_weaviate,
    delete_document_from_weaviate,
    _get_weaviate_client,
    chunk_markdown,
    fast_convert_to_markdown,
    fast_doc_to_weaviate,
    convert_to_markdown_with_webhook
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store pending webhook requests
pending_webhook_requests = {}

# Initialize FastAPI app
app = FastAPI(
    title="Document Processing API",
    description="API for converting documents to markdown and storing in Weaviate",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response models
class ConvertToMarkdownResponse(BaseModel):
    success: bool
    markdown_content: Optional[str] = None
    file_path: Optional[str] = None
    message: str
    error: Optional[str] = None

class ProcessDocumentResponse(BaseModel):
    success: bool
    document_id: Optional[str] = None
    file_name: Optional[str] = None
    total_chunks: Optional[int] = None
    processing_time: Optional[float] = None
    message: str
    error: Optional[str] = None

class ChunkMarkdownResponse(BaseModel):
    success: bool
    chunks: Optional[List[Dict]] = None
    total_chunks: Optional[int] = None
    message: str
    error: Optional[str] = None

class DeleteDocumentResponse(BaseModel):
    success: bool
    document_id: Optional[str] = None
    chunks_deleted: Optional[int] = None
    message: str
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    weaviate_connected: bool

class WebhookCallbackRequest(BaseModel):
    request_id: str
    success: bool
    markdown_content: Optional[str] = None
    extracted_data: Optional[Dict] = None
    timestamp: str

class WebhookRequest(BaseModel):
    request_id: str
    file_path: str
    webhook_url: str
    future: asyncio.Future
    
    class Config:
        arbitrary_types_allowed = True

#create new response model
class FastConvertToMarkdownResponse(BaseModel):
    success:bool
    markdown_content:Optional[str]=None
    file_path: Optional[str]=None
    processing_time: Optional[float] = None
    message: str
    error: Optional[str] = None

# Supported file formats for convert-doc-to-markdown (PDF and images only)
SUPPORTED_FORMATS = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.tiff', '.webp', '.docx', '.doc', '.xlsx', '.xls']

def validate_file_format(filename: str, supported_formats: List[str]) -> None:
    """Validate uploaded file format."""
    file_ext = Path(filename).suffix.lower()
    if file_ext not in supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_ext}. Supported: {supported_formats}"
        )

async def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file to temp directory and return path."""
    temp_file_path = os.path.join(tempfile.gettempdir(), file.filename)
    with open(temp_file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    return temp_file_path

def cleanup_temp_file(file_path: str) -> None:
    """Clean up temporary file."""
    if file_path and os.path.exists(file_path):
        try:
            os.unlink(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Could not clean up temp file {file_path}: {e}")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and Weaviate connection."""
    try:
        client = _get_weaviate_client()
        client.close()
        weaviate_connected = True
    except Exception:
        weaviate_connected = False
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        weaviate_connected=weaviate_connected
    )

# Webhook callback endpoint
@app.post("/api-callback")
async def webhook_callback(callback_data: WebhookCallbackRequest):
    """Receive webhook callback from webhook service."""
    try:
        request_id = callback_data.request_id
        logger.info(f"Received webhook callback for request_id: {request_id}")
        
        if request_id in pending_webhook_requests:
            webhook_request = pending_webhook_requests[request_id]
            
            # Set the result in the future
            if callback_data.success and callback_data.markdown_content:
                webhook_request["future"].set_result({
                    "success": True,
                    "markdown_content": callback_data.markdown_content,
                    "extracted_data": callback_data.extracted_data
                })
            else:
                webhook_request["future"].set_result({
                    "success": False,
                    "error": "Webhook callback indicated failure"
                })
            
            # Clean up
            del pending_webhook_requests[request_id]
            logger.info(f"Processed webhook callback for request_id: {request_id}")
        else:
            logger.warning(f"Received callback for unknown request_id: {request_id}")
        
        return {"success": True, "message": "Callback processed"}
        
    except Exception as e:
        logger.error(f"Error processing webhook callback: {e}")
        return {"success": False, "error": str(e)}

# Convert PDF/image to markdown
@app.post("/convert-doc-to-markdown", response_model=ConvertToMarkdownResponse)
async def convert_doc_to_markdown(
    file: UploadFile = File(...),
    use_local: bool = Query(True),
    use_webhook: bool = Query(False),
    webhook_url: Optional[str] = Query(None)
):
    """Convert an uploaded PDF or image file to markdown format."""
    temp_file_path = None
    try:
        validate_file_format(file.filename, SUPPORTED_FORMATS)
        temp_file_path = await save_uploaded_file(file)
        
        # Use webhook-based conversion if requested
        if use_webhook and webhook_url:
            # Generate request ID and store pending request
            request_id = str(uuid.uuid4())
            future = asyncio.Future()
            
            # Store the pending request
            pending_webhook_requests[request_id] = {
                "request_id": request_id,
                "file_path": temp_file_path,
                "webhook_url": webhook_url,
                "future": future
            }
            
            # Upload file to datalab with webhook
            from integration import DocumentProcessor
            processor = DocumentProcessor()
            result = processor._upload_to_marker_with_webhook(temp_file_path, webhook_url, request_id)
            
            if not result:
                raise ValueError("Failed to upload file to datalab")
            
            # Extract the datalab request_id from the response
            datalab_request_id = result.get('request_id') or result.get('id')
            if not datalab_request_id:
                raise ValueError("No request_id returned from datalab")
            
            # Update the pending request to use datalab request_id
            if datalab_request_id != request_id:
                pending_webhook_requests[datalab_request_id] = pending_webhook_requests.pop(request_id)
                pending_webhook_requests[datalab_request_id]["request_id"] = datalab_request_id
                logger.info(f"Updated pending request to use datalab request_id: {datalab_request_id}")
            
            # Wait for webhook callback
            try:
                webhook_result = await asyncio.wait_for(future, timeout=300)  # 5 minute timeout
                markdown_content = webhook_result.get("markdown_content")
                
                if not markdown_content:
                    raise ValueError("Webhook returned empty content")
                    
            except asyncio.TimeoutError:
                # Clean up pending request
                if datalab_request_id in pending_webhook_requests:
                    del pending_webhook_requests[datalab_request_id]
                raise ValueError("Webhook timeout - no response received")
        else:
            # Use traditional conversion
            markdown_content = convert_to_markdown(
                file_path=temp_file_path,
                use_local=use_local
            )
        
        if not markdown_content:
            raise ValueError("Conversion failed")
        
        return ConvertToMarkdownResponse(
            success=True,
            markdown_content=markdown_content,
            message="Document converted successfully"
        )
        
    except Exception as e:
        logger.error(f"Conversion error: {e}")
        return ConvertToMarkdownResponse(
            success=False,
            message="Failed to convert document",
            error=str(e)
        )
    finally:
        cleanup_temp_file(temp_file_path)

# Webhook-based convert PDF/image to markdown
@app.post("/convert-doc-to-markdown-webhook", response_model=ConvertToMarkdownResponse)
async def convert_doc_to_markdown_webhook(
    file: UploadFile = File(...),
    webhook_url: str = Query(...)
):
    """Convert an uploaded PDF or image file to markdown using webhook."""
    temp_file_path = None
    try:
        validate_file_format(file.filename, SUPPORTED_FORMATS)
        temp_file_path = await save_uploaded_file(file)
        
        # Generate request ID and create future for webhook callback
        request_id = str(uuid.uuid4())
        future = asyncio.Future()
        
        # Store the pending request
        pending_webhook_requests[request_id] = {
            "request_id": request_id,
            "file_path": temp_file_path,
            "webhook_url": webhook_url,
            "future": future
        }
        
        # Upload file to datalab with webhook
        from integration import DocumentProcessor
        processor = DocumentProcessor()
        result = processor._upload_to_marker_with_webhook(temp_file_path, webhook_url, request_id)
        
        if not result:
            raise ValueError("Failed to upload file to datalab")
        
        # Extract the datalab request_id from the response
        datalab_request_id = result.get('request_id') or result.get('id')
        if not datalab_request_id:
            raise ValueError("No request_id returned from datalab")
        
        # Update the pending request to use datalab request_id
        if datalab_request_id != request_id:
            pending_webhook_requests[datalab_request_id] = pending_webhook_requests.pop(request_id)
            pending_webhook_requests[datalab_request_id]["request_id"] = datalab_request_id
            logger.info(f"Updated pending request to use datalab request_id: {datalab_request_id}")
        
        # Wait for webhook callback
        try:
            webhook_result = await asyncio.wait_for(future, timeout=300)  # 5 minute timeout
            markdown_content = webhook_result.get("markdown_content")
            
            if not markdown_content:
                raise ValueError("Webhook returned empty content")
                
        except asyncio.TimeoutError:
            # Clean up pending request
            if datalab_request_id in pending_webhook_requests:
                del pending_webhook_requests[datalab_request_id]
            raise ValueError("Webhook timeout - no response received")
        
        return ConvertToMarkdownResponse(
            success=True,
            markdown_content=markdown_content,
            message="Document converted successfully via webhook"
        )
        
    except Exception as e:
        logger.error(f"Webhook conversion error: {e}")
        return ConvertToMarkdownResponse(
            success=False,
            message="Failed to convert document via webhook",
            error=str(e)
        )
    finally:
        cleanup_temp_file(temp_file_path)

# fast convert files into markdown 
@app.post("/fast-convert-to-markdown", response_model=ConvertToMarkdownResponse)
async def fast_convert_to_mark_down(
    file: UploadFile = File(...)
):
    temp_file_path = None
    start_time = time.time()
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Convert to markdown
        markdown_content = fast_convert_to_markdown(
            file_path=temp_file_path
        )
        
        processing_time = time.time() - start_time
        
        if markdown_content is None:
            return ConvertToMarkdownResponse(
                success=False,
                processing_time=processing_time,
                message="Conversion failed"
            )
        
        return ConvertToMarkdownResponse(
            success=True,
            markdown_content=markdown_content,
            processing_time=processing_time,
            content_length=len(markdown_content),
            message="Conversion successful"
        )
        
    except Exception as e:
        return ConvertToMarkdownResponse(
            success=False,
            processing_time=time.time() - start_time,
            message="Conversion failed",
            error=str(e)
        )
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

# Process document and save to Weaviate
@app.post("/process-doc-to-weaviate", response_model=ProcessDocumentResponse)
async def upload_and_process_document(
    file: UploadFile = File(...),
    tenant_id: str = Query(...),
    document_id: Optional[str] = Query(None),
    use_local: bool = Query(True)
):
    """Upload and process document to Weaviate."""
    temp_file_path = None
    try:
        validate_file_format(file.filename, SUPPORTED_FORMATS)
        temp_file_path = await save_uploaded_file(file)
        
        doc_id = document_id or str(uuid.uuid4())
        
        result = process_document_to_weaviate(
            file_path=temp_file_path,
            document_id=doc_id,
            tenant_id=tenant_id,
            use_local=use_local
        )
        
        return ProcessDocumentResponse(
            success=True,
            document_id=result["document_id"],
            file_name=result["file_name"],
            total_chunks=result["total_chunks"],
            message=f"Document processed with {result['total_chunks']} chunks"
        )
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return ProcessDocumentResponse(
            success=False,
            message="Failed to process document",
            error=str(e)
        )
    finally:
        cleanup_temp_file(temp_file_path)


# Fast process document and save to Weaviate
@app.post("/fast-doc-to-weaviate", response_model=ProcessDocumentResponse)
async def upload_and_fast_process_document(
    file: UploadFile = File(...),
    tenant_id: str = Query(...),
    document_id: Optional[str] = Query(None)
):
    """Upload and fast process document to Weaviate with configurable quality settings."""
    temp_file_path = None
    try:
        validate_file_format(file.filename, SUPPORTED_FORMATS)
        temp_file_path = await save_uploaded_file(file)
        
        doc_id = document_id or str(uuid.uuid4())
        
        result = fast_doc_to_weaviate(
            file_path=temp_file_path,
            document_id=doc_id,
            tenant_id=tenant_id
        )
        
        return ProcessDocumentResponse(
            success=True,
            document_id=result["document_id"],
            file_name=result["file_name"],
            total_chunks=result["total_chunks"],
            message=f"Document fast processed with {result['total_chunks']} chunks)"
        )
        
    except Exception as e:
        logger.error(f"Fast processing error: {e}")
        return ProcessDocumentResponse(
            success=False,
            message="Failed to fast process document",
            error=str(e)
        )
    finally:
        cleanup_temp_file(temp_file_path)


# Chunk markdown from text content
@app.post("/chunk-markdown-text", response_model=ChunkMarkdownResponse)
async def chunk_markdown_text(
    markdown_content: str
):
    """Chunk markdown text content directly."""
    temp_file_path = None
    try:
        # Save text content to temporary file since chunk_markdown requires a file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
            f.write(markdown_content)
            temp_file_path = f.name
        
        chunks = chunk_markdown(
            markdown_input=Path(temp_file_path)
        )
        
        return ChunkMarkdownResponse(
            success=True,
            chunks=chunks,
            total_chunks=len(chunks),
            message=f"Markdown chunked into {len(chunks)} chunks"
        )
        
    except Exception as e:
        logger.error(f"Chunking error: {e}")
        return ChunkMarkdownResponse(
            success=False,
            message="Failed to chunk markdown",
            error=str(e)
        )
    finally:
        cleanup_temp_file(temp_file_path)

# Chunk markdown file (requires file upload)
@app.post("/chunk-markdown", response_model=ChunkMarkdownResponse)
async def chunk_markdown_file(
    file: UploadFile = File(...)
):
    """Upload a markdown file and return chunks."""
    temp_file_path = None
    try:
        # Handle file upload - now required
        validate_file_format(file.filename, ['.md'])
        temp_file_path = await save_uploaded_file(file)
        chunks = chunk_markdown(
            markdown_input=Path(temp_file_path)
        )
        
        return ChunkMarkdownResponse(
            success=True,
            chunks=chunks,
            total_chunks=len(chunks),
            message=f"Markdown chunked into {len(chunks)} chunks"
        )
        
    except Exception as e:
        logger.error(f"Chunking error: {e}")
        return ChunkMarkdownResponse(
            success=False,
            message="Failed to chunk markdown",
            error=str(e)
        )
    finally:
        cleanup_temp_file(temp_file_path)

# Delete document from Weaviate
@app.delete("/delete-document", response_model=DeleteDocumentResponse)
async def delete_document(
    document_id: str = Query(...),
    tenant_id: str = Query(...)
):
    """Delete a document and its chunks from Weaviate."""
    try:
        result = delete_document_from_weaviate(
            document_id=document_id,
            tenant_id=tenant_id
        )
        
        return DeleteDocumentResponse(
            success=True,
            document_id=result["document_id"],
            chunks_deleted=result["chunks_deleted"],
            message=result["status"]
        )
        
    except Exception as e:
        logger.error(f"Deletion error: {e}")
        return DeleteDocumentResponse(
            success=False,
            message="Failed to delete document",
            error=str(e)
        )

# List documents
@app.get("/documents")
async def list_documents(
    tenant_id: str = Query(...),
    limit: int = Query(10, ge=1, le=100)
):
    """List all documents for a tenant."""
    client = _get_weaviate_client()
    try:
        documents_collection = client.collections.get("Documents").with_tenant(tenant_id)
        results = documents_collection.query.fetch_objects(limit=limit)
        
        documents = [
            {
                "document_id": str(obj.uuid),
                "file_name": obj.properties.get("file_name"),
                "file_size": obj.properties.get("file_size"),
                "total_chunks": obj.properties.get("total_chunks"),
                "mime_type": obj.properties.get("mime_type")
            }
            for obj in results.objects
        ]
        
        return {"documents": documents, "total": len(documents)}
    finally:
        client.close()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=True)
