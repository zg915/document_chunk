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

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import our document processing functions
from integration import (
    convert_to_markdown,
    process_document_to_weaviate,
    delete_document_from_weaviate,
    _get_weaviate_client,
    chunk_markdown
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Supported file formats
SUPPORTED_FORMATS = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.tiff', '.webp']

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

# Convert PDF/image to markdown
@app.post("/convert-doc-to-markdown", response_model=ConvertToMarkdownResponse)
async def convert_doc_to_markdown(
    file: UploadFile = File(...),
    save_to_file: bool = Query(False),
    output_path: Optional[str] = Query(None),
    use_local: bool = Query(True)
):
    """Convert an uploaded PDF or image file to markdown format."""
    temp_file_path = None
    try:
        validate_file_format(file.filename, SUPPORTED_FORMATS)
        temp_file_path = await save_uploaded_file(file)
        
        markdown_content = convert_to_markdown(
            file_path=temp_file_path,
            save_to_file=save_to_file,
            output_path=output_path,
            use_local=use_local
        )
        
        if not markdown_content:
            raise ValueError("Conversion failed")
        
        output_file_path = output_path if save_to_file else None
        if save_to_file and not output_path:
            output_file_path = f"{Path(file.filename).stem}.md"
        
        return ConvertToMarkdownResponse(
            success=True,
            markdown_content=markdown_content,
            file_path=output_file_path,
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

# Chunk markdown file
@app.post("/chunk-markdown", response_model=ChunkMarkdownResponse)
async def chunk_markdown_file(
    file: UploadFile = File(...),
    save_chunks_json: bool = Query(False),
    output_path: Optional[str] = Query(None)
):
    """Upload a markdown file and return its chunks."""
    temp_file_path = None
    try:
        validate_file_format(file.filename, ['.md'])
        temp_file_path = await save_uploaded_file(file)
        
        chunks = chunk_markdown(
            markdown_input=Path(temp_file_path),
            save_to_file=save_chunks_json,
            output_path=output_path
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
    uvicorn.run("api_server:app", host="0.0.0.0", port=8001, reload=True)