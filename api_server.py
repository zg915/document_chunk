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
import requests
from urllib.parse import urlparse
import asyncio
import threading

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import from new modular structure
from converter import (
    convert_to_markdown,
    fast_convert_to_markdown,
    get_webhook_manager
)
from storage import (
    chunk_markdown,
    fast_doc_to_weaviate,
    delete_document_from_weaviate,
    _get_weaviate_client
)
from utils import (
    acquire_file,
    cleanup_temp_file,
    save_uploaded_file,
    validate_file_format,
    ERROR_MESSAGES
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the webhook manager instance from integration module
webhook_manager = get_webhook_manager()

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
    processing_time: Optional[float] = None
    content_length: Optional[int] = None
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

class HealthResponse(BaseModel):
    status: str
    version: str
    weaviate_connected: bool

class DeleteDocumentResponse(BaseModel):
    success: bool
    document_id: Optional[str] = None
    chunks_deleted: Optional[int] = None
    message: str
    error: Optional[str] = None

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

        # Parse markdown content - extract only markdown field if it's a JSON response
        markdown_content = callback_data.markdown_content
        if markdown_content and isinstance(markdown_content, str):
            try:
                # Try to parse as JSON (datalab.to response)
                import json
                parsed_data = json.loads(markdown_content)
                if isinstance(parsed_data, dict) and 'markdown' in parsed_data:
                    # Extract only the markdown field from datalab.to response
                    markdown_content = parsed_data['markdown']
                    logger.info(f"Extracted markdown field from JSON response ({len(markdown_content)} chars)")
            except (json.JSONDecodeError, TypeError):
                # If not JSON, use the content as-is
                pass

        # Use the webhook manager to process the callback
        success = webhook_manager.process_callback(
            request_id=request_id,
            success=callback_data.success,
            markdown_content=markdown_content,
            extracted_data=callback_data.extracted_data
        )

        if success:
            logger.info(f"Processed webhook callback for request_id: {request_id}")
            return {"success": True, "message": "Callback processed"}
        else:
            logger.warning(f"Received callback for unknown request_id: {request_id}")
            return {"success": False, "error": "Unknown request_id"}

    except Exception as e:
        logger.error(f"Error processing webhook callback: {e}")
        return {"success": False, "error": str(e)}

# Convert PDF/image to markdown
@app.post("/convert-doc-to-markdown", response_model=ConvertToMarkdownResponse)
async def convert_doc_to_markdown(
    request: Request,
    file: UploadFile = File(default=None),
    url: str = Form(default=None)
):
    """
    Convert a PDF or image file to markdown format using Marker API.

    Accepts either:
    - An uploaded file (multipart/form-data)
    - A URL to download the file from
    """
    temp_file_path = None
    start_time = time.time()

    try:
        # Acquire file from either upload or URL
        temp_file_path, _ = await acquire_file(file, url)

        # Use the unified convert_to_markdown function with webhook support
        webhook_url = os.getenv('WEBHOOK_URL')
        markdown_content = await convert_to_markdown(
            file_path=temp_file_path,
            webhook_url=webhook_url
        )

        if not markdown_content:
            raise ValueError("Conversion failed")

        processing_time = time.time() - start_time

        return ConvertToMarkdownResponse(
            success=True,
            markdown_content=markdown_content,
            processing_time=processing_time,
            content_length=len(markdown_content),
            message="Document converted successfully"
        )

    except Exception as e:
        logger.error(f"Conversion error: {e}")
        return ConvertToMarkdownResponse(
            success=False,
            processing_time=time.time() - start_time,
            message="Failed to convert document",
            error=str(e)
        )
    finally:
        cleanup_temp_file(temp_file_path)


# fast convert files into markdown 
@app.post("/fast-convert-to-markdown", response_model=ConvertToMarkdownResponse)
async def fast_convert_to_mark_down(
    request: Request,
    file: UploadFile = File(default=None),
    url: str = Form(default=None)
):
    """
    Fast convert a document file to markdown format.

    Accepts either:
    - An uploaded file (multipart/form-data)
    - A URL to download the file from
    """
    temp_file_path = None
    start_time = time.time()

    try:
        temp_file_path, _ = await acquire_file(file, url)

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
        logger.error(f"Fast convert error - file: {file}, url: {url}")
        logger.error(f"Fast convert error - file.filename: {file.filename if file else 'None'}")
        logger.error(f"Fast convert error - temp_file_path: {temp_file_path}")
        logger.error(f"Fast convert error - exception: {str(e)}")
        logger.error(f"Fast convert error - exception type: {type(e).__name__}")
        return ConvertToMarkdownResponse(
            success=False,
            processing_time=time.time() - start_time,
            message="Conversion failed",
            error=str(e)
        )
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


# Fast process document and save to Weaviate
@app.post("/fast-doc-to-weaviate", response_model=ProcessDocumentResponse)
async def upload_and_fast_process_document(
    request: Request,
    file: UploadFile = File(default=None),
    url: str = Form(default=None),
    tenant_id: str = Query(..., description="Tenant ID for multi-tenancy"),
    document_id: Optional[str] = Query(None, description="Optional custom document ID (auto-generated if not provided)"),
    document_type: Optional[str] = Query(None, description="Optional document category (contract, manual, material_list, etc.)"),
    custom_file_path: Optional[str] = Query(None, description="Optional file path to store in DB (folder or full path)"),
    business_id: Optional[str] = Query(None, description="Optional business identifier associated with this document")
):
    """
    Upload and fast process document to Weaviate Personal_Documents/Personal_Chunks.

    Accepts either:
    - An uploaded file (multipart/form-data)
    - A URL to download the file from

    Optional parameters:
    - document_type: Category of the document (contract, manual, material_list, etc.)
    - custom_file_path: File path metadata to store in DB (e.g., 'ITS/Disney/' or 'ITS/Disney/file.pdf')
    - business_id: Business identifier associated with this document
    """
    temp_file_path = None
    start_time = time.time()

    try:
        temp_file_path, original_filename = await acquire_file(file, url)
        print(f"🔍 DEBUG: temp_file_path after acquire_file = {temp_file_path}")
        print(f"🔍 DEBUG: original_filename = {original_filename}")
        print(f"🔍 DEBUG: file exists = {os.path.exists(temp_file_path)}")
        print(f"🔍 DEBUG: file extension = {os.path.splitext(temp_file_path)[1]}")

        doc_id = document_id or str(uuid.uuid4())

        result = fast_doc_to_weaviate(
            file_path=temp_file_path,
            document_id=doc_id,
            tenant_id=tenant_id,
            document_type=document_type,
            custom_file_path=custom_file_path,
            business_id=business_id,
            original_filename=original_filename
        )

        processing_time = time.time() - start_time

        return ProcessDocumentResponse(
            success=True,
            document_id=result["document_id"],
            file_name=result["file_name"],
            total_chunks=result["total_chunks"],
            processing_time=processing_time,
            message=f"Document fast processed with {result['total_chunks']} chunks"
        )

    except Exception as e:
        logger.error(f"Fast processing error: {e}")
        return ProcessDocumentResponse(
            success=False,
            processing_time=time.time() - start_time,
            message="Failed to fast process document",
            error=str(e)
        )
    finally:
        cleanup_temp_file(temp_file_path)


# Chunk markdown content (accepts file upload or text)
@app.post("/chunk-markdown", response_model=ChunkMarkdownResponse)
async def chunk_markdown_content(
    request: Request,
    file: UploadFile = File(default=None),
    markdown_text: str = Form(default=None)
):
    """
    Chunk markdown content into smaller pieces.

    Accepts either:
    - An uploaded markdown file (multipart/form-data)
    - Raw markdown text content
    """
    if not file and not markdown_text:
        raise HTTPException(status_code=400, detail=ERROR_MESSAGES["FILE_OR_TEXT_REQUIRED"])

    if file and markdown_text:
        raise HTTPException(status_code=400, detail=ERROR_MESSAGES["FILE_OR_TEXT_NOT_BOTH"])

    temp_file_path = None
    try:
        if file:
            # Handle file upload
            validate_file_format(file.filename, ['.md'])
            temp_file_path, _ = await save_uploaded_file(file)
            chunks = chunk_markdown(
                markdown_input=Path(temp_file_path)
            )
        else:
            # Handle text input directly
            chunks = chunk_markdown(
                markdown_input=markdown_text
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


# Delete document and its chunks
@app.delete("/delete-document", response_model=DeleteDocumentResponse)
async def delete_document(
    document_id: str = Query(..., description="Document ID to delete"),
    tenant_id: str = Query(..., description="Tenant ID for multi-tenancy")
):
    """
    Delete a document and all its associated chunks from Weaviate Personal_Documents/Personal_Chunks.
    """
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

    except ValueError as e:
        # Document not found
        logger.warning(f"Document not found: {e}")
        return DeleteDocumentResponse(
            success=False,
            message="Document not found",
            error=str(e)
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
    tenant_id: str = Query(..., description="Tenant ID for multi-tenancy"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of documents to return")
):
    """List all documents for a tenant from Personal_Documents collection."""
    client = _get_weaviate_client()
    try:
        documents_collection = client.collections.get("Personal_Documents").with_tenant(tenant_id)
        results = documents_collection.query.fetch_objects(limit=limit)

        documents = [
            {
                "uuid": str(obj.uuid),
                "document_id": obj.properties.get("document_id"),
                "file_name": obj.properties.get("file_name"),
                "file_path": obj.properties.get("file_path"),
                "file_category": obj.properties.get("file_category"),
                "file_size": obj.properties.get("file_size"),
                "total_chunks": obj.properties.get("total_chunks"),
                "mime_type": obj.properties.get("mime_type"),
                "file_modified_at": obj.properties.get("file_modified_at")
            }
            for obj in results.objects
        ]

        return {"documents": documents, "total": len(documents)}
    finally:
        client.close()

# Custom OpenAPI schema to fix file upload display
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    from fastapi.openapi.utils import get_openapi
    openapi_schema = get_openapi(
        title="Document Processing API",
        version="1.0.0",
        description="API for converting documents to markdown and processing them with Weaviate",
        routes=app.routes,
    )

    # Fix file upload parameters in OpenAPI schema
    for path_item in openapi_schema["paths"].values():
        for operation in path_item.values():
            if isinstance(operation, dict) and "requestBody" in operation:
                content = operation["requestBody"].get("content", {})
                if "multipart/form-data" in content:
                    properties = content["multipart/form-data"]["schema"].get("properties", {})
                    for prop_name, prop_value in properties.items():
                        if prop_name == "file" and prop_value.get("type") == "string":
                            prop_value.update({
                                "type": "string",
                                "format": "binary"
                            })

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=True)
