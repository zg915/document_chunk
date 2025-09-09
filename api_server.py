"""
FastAPI server for document processing and Weaviate integration.

This API provides endpoints for:
1. Converting documents (PDF/images) to markdown
2. Processing documents and storing them in Weaviate vector database
3. Managing documents in Weaviate (delete, retrieve)

Usage:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import tempfile
import uuid
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import our document processing functions
from integration import (
    convert_to_markdown,
    process_document_to_weaviate,
    delete_document_from_weaviate,
    _get_weaviate_client,
    chunk_markdown
)

# Import Google Cloud integrations
from secrets_manager import get_weaviate_config, get_marker_config
from monitoring import logger, log_document_processed, log_api_request, log_error

# Initialize FastAPI app
app = FastAPI(
    title="Document Processing API",
    description="API for converting documents to markdown and storing in Weaviate",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure with specific domains in production
)

# Add CORS middleware with production-ready settings
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    log_api_request(
        endpoint=request.url.path,
        status_code=response.status_code,
        response_time=process_time
    )
    
    return response

# Pydantic models for request/response
class ConvertToMarkdownRequest(BaseModel):
    """Request model for convert to markdown endpoint."""
    file_path: str = Field(..., description="Path to the input file")
    save_to_file: bool = Field(False, description="Whether to save markdown to file")
    output_path: Optional[str] = Field(None, description="Custom output path for markdown file")
    use_local: bool = Field(True, description="Use local Marker (True) or API (False)")

class ConvertToMarkdownResponse(BaseModel):
    """Response model for convert to markdown endpoint."""
    success: bool
    markdown_content: Optional[str] = None
    file_path: Optional[str] = None
    message: str
    error: Optional[str] = None

class ProcessDocumentRequest(BaseModel):
    """Request model for process document endpoint."""
    file_path: str = Field(..., description="Path to the input file")
    document_id: Optional[str] = Field(None, description="Custom document ID (auto-generated if not provided)")
    tenant_id: str = Field(..., description="Tenant ID for multi-tenancy")
    save_markdown: bool = Field(False, description="Save markdown to file")
    save_chunks_json: bool = Field(False, description="Save chunks to JSON file")
    output_dir: Optional[str] = Field(None, description="Output directory for files")

class ProcessDocumentResponse(BaseModel):
    """Response model for process document endpoint."""
    success: bool
    document_id: Optional[str] = None
    file_name: Optional[str] = None
    total_chunks: Optional[int] = None
    message: str
    error: Optional[str] = None

class DeleteDocumentRequest(BaseModel):
    """Request model for delete document endpoint."""
    document_id: str = Field(..., description="Document ID to delete")
    tenant_id: str = Field(..., description="Tenant ID")

class DeleteDocumentResponse(BaseModel):
    """Response model for delete document endpoint."""
    success: bool
    document_id: Optional[str] = None
    chunks_deleted: Optional[int] = None
    message: str
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    weaviate_connected: bool

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and Weaviate connection."""
    try:
        # Test Weaviate connection
        client = _get_weaviate_client()
        client.close()
        weaviate_connected = True
    except Exception as e:
        logger.warning(f"Weaviate connection test failed: {e}")
        weaviate_connected = False
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        weaviate_connected=weaviate_connected
    )

# Convert to markdown endpoint (file path)
@app.post("/convert-to-markdown", response_model=ConvertToMarkdownResponse)
async def convert_document_to_markdown(request: ConvertToMarkdownRequest):
    """
    Convert a PDF or image file to markdown format using file path.
    
    This endpoint processes documents using the Marker API or local installation
    and returns the markdown content.
    """
    try:
        logger.info(f"Converting document to markdown: {request.file_path}")
        
        # Convert document to markdown
        markdown_content = convert_to_markdown(
            file_path=request.file_path,
            save_to_file=request.save_to_file,
            output_path=request.output_path,
            use_local=request.use_local
        )
        
        if markdown_content is None:
            return ConvertToMarkdownResponse(
                success=False,
                message="Failed to convert document to markdown",
                error="Conversion returned None"
            )
        
        # Determine output file path if saved
        output_file_path = None
        if request.save_to_file:
            if request.output_path:
                output_file_path = request.output_path
            else:
                input_path = Path(request.file_path)
                output_file_path = str(input_path.with_suffix('.md'))
        
        return ConvertToMarkdownResponse(
            success=True,
            markdown_content=markdown_content,
            file_path=output_file_path,
            message="Document converted to markdown successfully"
        )
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return ConvertToMarkdownResponse(
            success=False,
            message="File not found",
            error=str(e)
        )
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return ConvertToMarkdownResponse(
            success=False,
            message="Invalid input",
            error=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return ConvertToMarkdownResponse(
            success=False,
            message="Internal server error",
            error=str(e)
        )

# Convert to markdown endpoint (file upload)
@app.post("/convert-to-markdown-upload", response_model=ConvertToMarkdownResponse)
async def convert_uploaded_document_to_markdown(
    file: UploadFile = File(...),
    save_to_file: bool = Query(False, description="Save markdown to file"),
    output_path: Optional[str] = Query(None, description="Custom output path for markdown file"),
    use_local: bool = Query(True, description="Use local Marker (True) or API (False)")
):
    """
    Convert an uploaded PDF or image file to markdown format.
    
    This endpoint accepts file uploads and processes them using the Marker API
    or local installation, returning the markdown content.
    """
    temp_file_path = None
    try:
        # Validate file type
        file_ext = Path(file.filename).suffix.lower()
        supported_formats = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.tiff', '.webp']
        
        if file_ext not in supported_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_ext}. Supported formats: {supported_formats}"
            )
        
        # Save uploaded file to temporary location
        temp_file_path = os.path.join(tempfile.gettempdir(), f"upload_{uuid.uuid4()}{file_ext}")
        
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"Uploaded file saved to: {temp_file_path}")
        
        # Convert document to markdown
        markdown_content = convert_to_markdown(
            file_path=temp_file_path,
            save_to_file=save_to_file,
            output_path=output_path,
            use_local=use_local
        )
        
        if markdown_content is None:
            return ConvertToMarkdownResponse(
                success=False,
                message="Failed to convert document to markdown",
                error="Conversion returned None"
            )
        
        # Determine output file path if saved
        output_file_path = None
        if save_to_file:
            if output_path:
                output_file_path = output_path
            else:
                input_name = Path(file.filename).stem
                output_file_path = f"{input_name}.md"
        
        return ConvertToMarkdownResponse(
            success=True,
            markdown_content=markdown_content,
            file_path=output_file_path,
            message="Document converted to markdown successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        return ConvertToMarkdownResponse(
            success=False,
            message="Failed to process uploaded file",
            error=str(e)
        )
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Could not clean up temp file {temp_file_path}: {e}")

# Process document to Weaviate endpoint (file path)
@app.post("/process-document", response_model=ProcessDocumentResponse)
async def process_document(request: ProcessDocumentRequest):
    """
    Process a document and store it in Weaviate vector database using file path.
    
    This endpoint:
    1. Converts the document to markdown
    2. Chunks the markdown content
    3. Stores document metadata in Weaviate
    4. Stores chunks in Weaviate with references
    """
    try:
        logger.info(f"Processing document: {request.file_path}")
        
        # Generate document ID if not provided
        document_id = request.document_id or str(uuid.uuid4())
        
        # Process document to Weaviate
        result = process_document_to_weaviate(
            file_path=request.file_path,
            document_id=document_id,
            tenant_id=request.tenant_id,
            save_markdown=request.save_markdown,
            save_chunks_json=request.save_chunks_json,
            output_dir=request.output_dir
        )
        
        return ProcessDocumentResponse(
            success=True,
            document_id=result["document_id"],
            file_name=result["file_name"],
            total_chunks=result["total_chunks"],
            message=f"Document processed successfully with {result['total_chunks']} chunks"
        )
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return ProcessDocumentResponse(
            success=False,
            message="File not found",
            error=str(e)
        )
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return ProcessDocumentResponse(
            success=False,
            message="Invalid input",
            error=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return ProcessDocumentResponse(
            success=False,
            message="Internal server error",
            error=str(e)
        )

# Process document to Weaviate endpoint (file upload)
@app.post("/process-document-upload", response_model=ProcessDocumentResponse)
async def process_uploaded_document(
    file: UploadFile = File(...),
    tenant_id: str = Query(..., description="Tenant ID for multi-tenancy"),
    document_id: Optional[str] = Query(None, description="Custom document ID"),
    save_markdown: bool = Query(False, description="Save markdown to file"),
    save_chunks_json: bool = Query(False, description="Save chunks to JSON file"),
    output_dir: Optional[str] = Query(None, description="Output directory for files")
):
    """
    Process an uploaded document and store it in Weaviate vector database.
    
    This endpoint:
    1. Accepts file upload
    2. Converts the document to markdown
    3. Chunks the markdown content
    4. Stores document metadata in Weaviate
    5. Stores chunks in Weaviate with references
    """
    temp_file_path = None
    try:
        # Validate file type
        file_ext = Path(file.filename).suffix.lower()
        supported_formats = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.tiff', '.webp']
        
        if file_ext not in supported_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_ext}. Supported formats: {supported_formats}"
            )
        
        # Save uploaded file to temporary location
        temp_file_path = os.path.join(tempfile.gettempdir(), f"upload_{uuid.uuid4()}{file_ext}")
        
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"Uploaded file saved to: {temp_file_path}")
        
        # Generate document ID if not provided
        doc_id = document_id or str(uuid.uuid4())
        
        # Process document to Weaviate
        result = process_document_to_weaviate(
            file_path=temp_file_path,
            document_id=doc_id,
            tenant_id=tenant_id,
            save_markdown=save_markdown,
            save_chunks_json=save_chunks_json,
            output_dir=output_dir
        )
        
        return ProcessDocumentResponse(
            success=True,
            document_id=result["document_id"],
            file_name=result["file_name"],
            total_chunks=result["total_chunks"],
            message=f"Document processed successfully with {result['total_chunks']} chunks"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        return ProcessDocumentResponse(
            success=False,
            message="Failed to process uploaded file",
            error=str(e)
        )
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Could not clean up temp file {temp_file_path}: {e}")

# Delete document from Weaviate endpoint
@app.delete("/delete-document", response_model=DeleteDocumentResponse)
async def delete_document(request: DeleteDocumentRequest):
    """
    Delete a document and all its associated chunks from Weaviate.
    
    This endpoint removes both the document metadata and all its chunks
    from the Weaviate vector database.
    """
    try:
        logger.info(f"Deleting document: {request.document_id}")
        
        # Delete document from Weaviate
        result = delete_document_from_weaviate(
            document_id=request.document_id,
            tenant_id=request.tenant_id
        )
        
        return DeleteDocumentResponse(
            success=True,
            document_id=result["document_id"],
            chunks_deleted=result["chunks_deleted"],
            message=result["status"]
        )
        
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        return DeleteDocumentResponse(
            success=False,
            message="Failed to delete document",
            error=str(e)
        )

# File upload endpoint for direct file processing
@app.post("/upload-and-process", response_model=ProcessDocumentResponse)
async def upload_and_process_document(
    file: UploadFile = File(...),
    tenant_id: str = Query(..., description="Tenant ID for multi-tenancy"),
    document_id: Optional[str] = Query(None, description="Custom document ID"),
    save_markdown: bool = Query(False, description="Save markdown to file"),
    save_chunks_json: bool = Query(False, description="Save chunks to JSON file")
):
    """
    Upload a file and process it directly to Weaviate.
    
    This endpoint accepts file uploads and processes them without requiring
    the file to be saved to disk first.
    """
    temp_file_path = None
    try:
        # Validate file type
        file_ext = Path(file.filename).suffix.lower()
        supported_formats = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.tiff', '.webp']
        
        if file_ext not in supported_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_ext}. Supported formats: {supported_formats}"
            )
        
        # Save uploaded file to temporary location
        temp_file_path = os.path.join(tempfile.gettempdir(), f"upload_{uuid.uuid4()}{file_ext}")
        
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"Uploaded file saved to: {temp_file_path}")
        
        # Generate document ID if not provided
        doc_id = document_id or str(uuid.uuid4())
        
        # Process the uploaded file
        result = process_document_to_weaviate(
            file_path=temp_file_path,
            document_id=doc_id,
            tenant_id=tenant_id,
            save_markdown=save_markdown,
            save_chunks_json=save_chunks_json
        )
        
        return ProcessDocumentResponse(
            success=True,
            document_id=result["document_id"],
            file_name=result["file_name"],
            total_chunks=result["total_chunks"],
            message=f"File uploaded and processed successfully with {result['total_chunks']} chunks"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        return ProcessDocumentResponse(
            success=False,
            message="Failed to process uploaded file",
            error=str(e)
        )
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Could not clean up temp file {temp_file_path}: {e}")

# Get document info endpoint
@app.get("/document/{document_id}")
async def get_document_info(
    document_id: str,
    tenant_id: str = Query(..., description="Tenant ID")
):
    """
    Get information about a document stored in Weaviate.
    """
    try:
        client = _get_weaviate_client()
        
        # Get document from Weaviate
        documents_collection = client.collections.get("Documents").with_tenant(tenant_id)
        document = documents_collection.query.fetch_object_by_id(document_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get chunks count
        chunks_collection = client.collections.get("Chunks").with_tenant(tenant_id)
        from weaviate.classes.query import Filter
        chunk_results = chunks_collection.query.fetch_objects(
            where=Filter.by_property("document_id").equal(document_id),
            limit=1
        )
        
        client.close()
        
        return {
            "document_id": document_id,
            "file_name": document.properties.get("file_name"),
            "file_size": document.properties.get("file_size"),
            "total_chunks": document.properties.get("total_chunks"),
            "mime_type": document.properties.get("mime_type"),
            "actual_chunks_in_db": len(chunk_results.objects)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# List documents endpoint
@app.get("/documents")
async def list_documents(
    tenant_id: str = Query(..., description="Tenant ID"),
    limit: int = Query(10, description="Maximum number of documents to return")
):
    """
    List all documents for a tenant.
    """
    try:
        client = _get_weaviate_client()
        
        # Get documents from Weaviate
        documents_collection = client.collections.get("Documents").with_tenant(tenant_id)
        results = documents_collection.query.fetch_objects(limit=limit)
        
        documents = []
        for obj in results.objects:
            documents.append({
                "document_id": str(obj.uuid),
                "file_name": obj.properties.get("file_name"),
                "file_size": obj.properties.get("file_size"),
                "total_chunks": obj.properties.get("total_chunks"),
                "mime_type": obj.properties.get("mime_type")
            })
        
        client.close()
        
        return {
            "documents": documents,
            "total": len(documents)
        }
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
