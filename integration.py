"""
Document processing module for converting documents to markdown and chunking.

This module provides functions to:
1. Convert PDF/image files to markdown using the Marker API
2. Chunk markdown content into smaller, manageable pieces for vector storage
3. Save documents and chunks to Weaviate vector database
"""

import json
import logging
import mimetypes
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

# GPU optimizations - set before importing torch-based libraries
os.environ["OMP_NUM_THREADS"] = "1"
try:
    import torch
    if torch.cuda.is_available():
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"GPU optimizations enabled: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"GPU optimizations failed, continuing without: {e}")
    torch = None

import requests
import tiktoken
from dotenv import load_dotenv
from PIL import Image
from unstructured.partition.md import partition_md
from unstructured.chunking.title import chunk_by_title
from unstructured.chunking.basic import chunk_elements
from weaviate.util import generate_uuid5
from weaviate import WeaviateClient
from weaviate.auth import AuthApiKey
from weaviate.connect import ConnectionParams, ProtocolParams

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv('.env')

# Configuration
class Config:
    """Configuration settings for the document processor."""
    API_KEY = os.getenv("MARKER_API_KEY")
    API_URL = os.getenv("MARKER_API_URL", "https://www.datalab.to/api/v1/marker")
    
    MARKER_PARAMS = {
        'output_format': 'markdown',
        'use_llm': True,
        'disable_image_extraction': True
    }
    
    SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.gif', '.tiff', '.webp']
    SUPPORTED_DOCUMENT_FORMATS = ['.pdf'] + SUPPORTED_IMAGE_FORMATS
    
    # Chunking parameters
    MAX_CHUNK_SIZE = 2500
    NEW_CHUNK_AFTER = 2000
    MIN_CHUNK_SIZE = 500
    CHUNK_OVERLAP = 200
    
    # API settings
    MAX_RETRIES = 30
    RETRY_DELAY = 10
    
    # Local Marker settings - using marker instead of marker_single for batch processing
    LOCAL_MARKER_COMMAND = "marker"  # Use marker for consistent batch processing
    LOCAL_MARKER_OUTPUT_DIR = os.getenv("LOCAL_MARKER_OUTPUT_DIR", "./marker_output")
    
    # GPU optimization settings
    NUM_WORKERS = int(os.getenv("MARKER_NUM_WORKERS", "4"))  # 4-7 range recommended
    PAGE_BATCH_SIZE = int(os.getenv("MARKER_PAGE_BATCH", "6"))  # 4-8 range recommended
    
    # Weaviate settings
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
    WEAVIATE_URL = os.getenv("WEAVIATE_URL")
    WEAVIATE_HTTP_PORT = 8080
    WEAVIATE_GRPC_PORT = 50051


class DocumentProcessor:
    """Handles document conversion and processing operations."""
    
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        """Initialize the document processor."""
        self.api_key = api_key or Config.API_KEY
        self.api_url = api_url or Config.API_URL
        self.headers = {"X-API-Key": self.api_key}
    
    def _convert_image_to_pdf(self, image_path: str) -> str:
        """
        Convert an image file to PDF format.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Path to the temporary PDF file
        """
        try:
            with Image.open(image_path) as img:
                temp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
                temp_pdf_path = temp_pdf.name
                temp_pdf.close()
                
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img.save(temp_pdf_path, 'PDF', resolution=150, quality=95)
                logger.info(f"Converted image to PDF: {temp_pdf_path}")
                return temp_pdf_path
        except Exception as e:
            logger.error(f"Failed to convert image to PDF: {e}")
            raise
    
    def _upload_to_marker(self, file_path: str) -> Optional[Dict]:
        """
        Upload a file to the Marker API for processing.
        
        Args:
            file_path: Path to the file to upload
            
        Returns:
            API response or None if failed
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        temp_file_created = False
        
        try:
            if file_ext == '.pdf':
                mime_type = 'application/pdf'
                process_path = file_path
            elif file_ext in Config.SUPPORTED_IMAGE_FORMATS:
                mime_type = 'application/pdf'
                process_path = self._convert_image_to_pdf(file_path)
                temp_file_created = True
            else:
                logger.error(f"Unsupported file format: {file_ext}")
                return None
            
            with open(process_path, 'rb') as f:
                files = {'file': (os.path.basename(process_path), f, mime_type)}
                response = requests.post(
                    self.api_url, 
                    headers=self.headers, 
                    files=files, 
                    data=Config.MARKER_PARAMS
                )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API request failed with status {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return None
            
        finally:
            if temp_file_created and os.path.exists(process_path):
                os.unlink(process_path)
                logger.debug(f"Cleaned up temporary file: {process_path}")
    
    def _wait_for_processing(self, check_url: str) -> Optional[str]:
        """
        Wait for asynchronous processing to complete.
        
        Args:
            check_url: URL to check processing status
            
        Returns:
            Markdown content or None if failed
        """
        for attempt in range(Config.MAX_RETRIES):
            try:
                response = requests.get(check_url, headers=self.headers)
                if response.status_code == 200:
                    result = response.json()
                    status = result.get('status', 'unknown')
                    
                    if status in ['completed', 'complete']:
                        logger.info("Processing completed successfully")
                        return result.get('markdown')
                    elif status == 'failed':
                        logger.error("Processing failed on server")
                        return None
                    else:
                        logger.debug(f"Processing status: {status} (attempt {attempt + 1}/{Config.MAX_RETRIES})")
                        
            except Exception as e:
                logger.error(f"Error checking status: {e}")
            
            time.sleep(Config.RETRY_DELAY)
        
        logger.error("Processing timed out")
        return None
    
    def _process_with_local_marker(self, file_path: str) -> Optional[str]:
        """
        Process a file using local Marker installation.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Markdown content or None if failed
        """
        try:
            logger.info("Using marker Python API for GPU-optimized processing")
            
            # Import correct marker modules according to PyPI documentation
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.output import text_from_rendered
            
            # Use preloaded models if available
            try:
                from preload_models import get_preloaded_models
                models = get_preloaded_models()
                if models:
                    logger.info(f"✅ Using {len(models)} preloaded models")
                else:
                    logger.info("Loading models...")
                    models = create_model_dict()
            except ImportError:
                logger.info("Loading models...")
                models = create_model_dict()
            
            # Create optimized configuration for faster processing
            from marker.config.parser import ConfigParser
            
            # Speed optimizations
            config = {
                # GPU optimization for Tesla T4 (16GB VRAM)
                "batch_multiplier": 8,  # Increase to 8x for Tesla T4
                "ocr_batch_size": 32,  # Larger OCR batch size
                "layout_batch_size": 8,  # Larger layout batch
                "table_rec_batch_size": 8,  # Larger table batch
                
                # OCR optimizations
                "ocr_all_pages": False,  # Only OCR when needed
                "disable_ocr": False,  # Keep OCR but optimize
                "ocr_error_detection": False,  # Skip OCR error detection for speed
                "detect_language": False,  # Skip language detection
                
                # Processing optimizations
                "paginate_output": False,  # Faster without pagination
                "disable_image_extraction": True,  # Skip images
                "skip_table_detection": False,  # Keep tables but optimize
                
                # Parallel processing
                "workers": 6,  # More parallel workers
                "ray_workers": 6,  # Ray parallel processing
                
                # Other optimizations
                "use_llm": True,  # No LLM for speed
                "force_gpu": True,  # Force GPU usage
                "langs": ["en"],  # Skip language detection
            }
            
            config_parser = ConfigParser(config)
            
            # Create converter with GPU optimizations and config
            converter = PdfConverter(
                artifact_dict=models,
                config=config_parser.generate_config_dict()
            )
            
            # Convert PDF to markdown
            logger.info(f"Converting {file_path} with GPU acceleration...")
            logger.info(f"Settings: batch_multiplier=4, ocr_all_pages=False, workers=4")
            start_time = time.perf_counter()
            
            # Run conversion
            rendered = converter(file_path)
            text, _, _ = text_from_rendered(rendered)
            
            processing_time = time.perf_counter() - start_time
            logger.info(f"✅ Conversion completed in {processing_time:.2f}s")
            
            return text
            
        except ImportError as e:
            logger.error(f"Marker Python API not available: {e}")
            logger.error("Please ensure marker-pdf is installed: pip install marker-pdf[gpu]")
            return None
        except Exception as e:
            logger.error(f"Error processing with local Marker: {str(e)}")
            return None


def convert_to_markdown(
    file_path: str,
    save_to_file: Optional[bool] = False,
    output_path: Optional[str] = None,
    use_local: Optional[bool] = None
) -> Optional[str]:
    """
    Convert a PDF or image file to markdown format.
    
    Args:
        file_path: Path to the input file (PDF or supported image format)
        save_to_file: Whether to save the markdown to a file
        output_path: Optional custom output path (defaults to input_name.md)
        use_local: Whether to use local Marker (True) or API (False). Defaults to True.
        
    Returns:
        Markdown content as string, or None if conversion failed
        
    Raises:
        FileNotFoundError: If the input file doesn't exist
        ValueError: If the file format is not supported
    """
    # Validate input
    file_path = os.path.expanduser(file_path.strip().strip('"').strip("'"))
    file_path = os.path.abspath(file_path)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext not in Config.SUPPORTED_DOCUMENT_FORMATS:
        raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: {Config.SUPPORTED_DOCUMENT_FORMATS}")
    
    # Process file
    processor = DocumentProcessor()
    logger.info(f"Processing file: {file_path}")
    
    # Default to local processing if use_local is not specified
    if use_local is None:
        use_local = True
    
    # Choose between local and API processing
    if use_local:
        logger.info("Using local Marker for processing")
        markdown_content = processor._process_with_local_marker(file_path)
        
        # If local processing fails, raise error instead of fallback
        if markdown_content is None:
            raise RuntimeError("Local Marker processing failed. Please check Marker installation and configuration.")
    else:
        logger.info("Using Marker API for processing")
        # Upload and get initial response
        result = processor._upload_to_marker(file_path)
        if not result:
            return None
        
        # Check if processing is complete or async
        if result.get('success'):
            if 'markdown' in result and result['markdown']:
                markdown_content = result['markdown']
            elif 'request_check_url' in result:
                markdown_content = processor._wait_for_processing(result['request_check_url'])
            else:
                logger.error("Unexpected API response format")
                return None
        else:
            logger.error(f"API returned error: {result.get('error', 'Unknown error')}")
            return None
    
    if not markdown_content:
        return None
    
    # Save to file if requested
    if save_to_file:
        if output_path is None:
            output_path = Path(file_path).with_suffix('.md')
        else:
            output_path = Path(output_path)
            
        output_path.write_text(markdown_content, encoding='utf-8')
        logger.info(f"Saved markdown to: {output_path}")
    
    return markdown_content


def chunk_markdown(
    markdown_input: Union[str, Path],
    save_to_file: Optional[bool] = False,
    output_path: Optional[str] = None,
) -> List[Dict]:
    """
    Chunk markdown content into smaller pieces for vector storage.
    
    Args:
        markdown_input: Either markdown text string or path to markdown file
        max_chunk_size: Maximum characters per chunk (default: 2500)
        new_chunk_after: Start new chunk after this many characters (default: 2000)
        min_chunk_size: Minimum chunk size to avoid tiny chunks (default: 500)
        chunk_overlap: Character overlap between chunks (default: 200)
        include_metadata: Whether to include metadata in chunk output
        
    Returns:
        List of chunk dictionaries containing content and metadata
        
    Raises:
        FileNotFoundError: If markdown_input is a path that doesn't exist
        ValueError: If the input is empty
    """
    # Handle input - either text or file path
    markdown_content = markdown_input
    if isinstance(markdown_input, Path):
        with open(markdown_input, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        logger.info(f"Loaded markdown from file: {markdown_input}")
    
    if not markdown_content or not markdown_content.strip():
        raise ValueError("Empty markdown content")
    
    # Set chunking parameters
    max_chunk_size = Config.MAX_CHUNK_SIZE
    new_chunk_after = Config.NEW_CHUNK_AFTER
    min_chunk_size = Config.MIN_CHUNK_SIZE
    chunk_overlap = Config.CHUNK_OVERLAP
    
    # Parse markdown into elements
    try:
        elements = partition_md(text=markdown_content)
    except Exception as e:
        logger.error(f"Failed to partition markdown: {e}")
        raise
    
    # Extract titles for context
    title_by_id = {}
    for el in elements:
        if getattr(el, "category", None) == "Title" and hasattr(el, "element_id"):
            title_by_id[getattr(el, "element_id")] = str(el).strip()
    
    # Chunk the elements
    try:
        if chunk_by_title is None:
            raise ImportError("chunk_by_title not available")
        chunks = chunk_by_title(
            elements,
            max_characters=max_chunk_size,
            new_after_n_chars=new_chunk_after,
            combine_text_under_n_chars=min_chunk_size
        )
        logger.info(f"Used title-based chunking")
    except Exception as e:
        logger.warning(f"Title-based chunking failed, falling back to basic: {e}")
        chunks = chunk_elements(
            elements,
            max_characters=max_chunk_size,
            new_after_n_chars=new_chunk_after,
            overlap=chunk_overlap
        )
    
    # Prepare final chunks with metadata
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    final_chunks = []
    
    for i, chunk in enumerate(chunks):
        text = str(chunk).strip()
        if not text:
            continue
        
        # Add header context if available
        header = None
        
        # Use Unstructured's parent_id to retrieve title directly
        parent_id = getattr(getattr(chunk, "metadata", None), "parent_id", None)
        if parent_id and parent_id in title_by_id:
            header = title_by_id[parent_id][:200]
        
        # If no parent relationship found, check if chunk starts with a title-like line
        if not header and hasattr(chunk, 'category') and chunk.category == 'CompositeElement':
            first_line = text.split('\n')[0].strip()
            # Simple check: if first line is short and looks like a title
            if len(first_line) < 100 and len(first_line) > 5:
                # Check if it's mostly uppercase or title case
                if (first_line.isupper() or 
                    first_line.istitle() or 
                    first_line.count(':') > 0):  # Often titles end with colons
                    header = first_line
        
        # For TableChunk, use a descriptive identifier
        if hasattr(chunk, 'category') and chunk.category == 'TableChunk':
            header = "Table Data"
        
        chunk_data = {
            "content": text,
            "chunk_index": i,
            "header": header,
            "char_count": len(text),
            "token_count": len(tokenizer.encode(text)),
            "chunk_type": chunk.category if hasattr(chunk, 'category') else "text"
        }
        
        final_chunks.append(chunk_data)
    
    logger.info(f"Created {len(final_chunks)} chunks from markdown content")
    
    # Save to file if requested
    if save_to_file and output_path:
        _save_chunks_to_file(final_chunks, output_path)
    
    return final_chunks

def _save_chunks_to_file(chunks: List[Dict], output_path: Union[str, Path]) -> None:
    """
    Save chunks to a JSON file.
    
    Args:
        chunks: List of chunk dictionaries
        output_path: Path to save the JSON file
    """
    output_path = Path(output_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(chunks)} chunks to: {output_path}")

def save_document_to_weaviate(
    client,
    file_path: str,
    chunks: List[Dict],
    document_id: str,
    tenant_id: str
) -> str:
    """
    Save document metadata to Weaviate Documents collection.
    
    Args:
        client: Weaviate client instance
        file_path: Path to the original document file
        chunks: List of chunks (used to get total_chunks count)
        document_id: Unique identifier for the document
        tenant_id: Tenant ID for multi-tenancy
        
    Returns:
        The document UUID used in Weaviate
    """
    # Get file information
    file_stats = os.stat(file_path)
    file_path_obj = Path(file_path)
    
    file_name = file_path_obj.name
    file_size = file_stats.st_size
    mime_type = mimetypes.guess_type(file_path)[0]
    total_chunks = len(chunks)
    
    # Prepare document data
    document_data = {
        "file_name": file_name,
        "file_size": file_size,
        "total_chunks": total_chunks,
        "mime_type": mime_type or "application/octet-stream",
    }
    
    # Save to Weaviate Documents collection
    documents_collection = client.collections.get("Documents").with_tenant(tenant_id)
    documents_collection.data.insert(
        properties=document_data,
        uuid=document_id
    )
    
    logger.info(f"Saved document '{file_name}' to Weaviate with UUID: {document_id}")
    return document_id


def save_chunks_to_weaviate(
    client,
    chunks: List[Dict],
    document_id: str,
    tenant_id: str
) -> None:
    """
    Save chunks to Weaviate Chunks collection with reference to document.
    
    Args:
        client: Weaviate client instance
        chunks: List of chunk dictionaries from chunk_markdown
        document_id: Document UUID to reference
        tenant_id: Tenant ID for multi-tenancy
    """
    chunks_collection = client.collections.get("Chunks").with_tenant(tenant_id)
    
    with chunks_collection.batch.fixed_size(batch_size=50) as batch:
        for i, chunk in enumerate(chunks):
            chunk_uuid = generate_uuid5(f"{document_id}_chunk_{i}")
            
            chunk_data = {
                "content": chunk["content"],
                "chunk_index": i,
                "header": chunk.get("header", ""),
                "document_id": document_id,
                "char_count": chunk.get("char_count", len(chunk["content"])),
                "token_count": chunk.get("token_count", 0),
                "chunk_type": chunk.get("chunk_type", "text")
            }
            
            batch.add_object(
                properties=chunk_data,
                uuid=chunk_uuid,
                references={"document_object": document_id}
            )
    
    logger.info(f"Saved {len(chunks)} chunks to Weaviate for document {document_id}")

def delete_document_from_weaviate(
    document_id: str,
    tenant_id: str,
    client: Optional[object] = None
) -> Dict[str, Union[str, int]]:
    """
    Delete a document and all its associated chunks from Weaviate.
    
    Args:
        document_id: UUID of the document to delete
        tenant_id: Tenant ID for multi-tenancy
        client: Weaviate client instance (optional, will create if not provided)
        
    Returns:
        Dictionary containing:
        - document_id: The document UUID that was deleted
        - chunks_deleted: Number of chunks deleted
        - status: Success message
        
    Raises:
        Exception: If deletion fails
    """
    # Create client if not provided
    client_created = False
    if client is None:
        client = _get_weaviate_client()
        client_created = True
    
    try:
        # Track deletion results
        chunks_deleted = 0
        
        # Step 1: Delete all chunks associated with this document
        chunks_collection = client.collections.get("Chunks").with_tenant(tenant_id)
        
        # Use where filter to find and delete all chunks with this document_id
        logger.info(f"Deleting chunks for document {document_id}")
        
        # First, query to find chunks with this document_id
        try:
            # Use the correct query syntax for newer Weaviate
            from weaviate.classes.query import Filter
            
            chunk_results = chunks_collection.query.fetch_objects(
                where=Filter.by_property("document_id").equal(document_id),
                limit=1000  # Adjust if documents have more chunks
            )
            
            # Delete each chunk by UUID
            for chunk in chunk_results.objects:
                chunks_collection.data.delete_by_id(chunk.uuid)
                chunks_deleted += 1
                
            logger.info(f"Deleted {chunks_deleted} chunks")
        except Exception as e:
            logger.warning(f"Could not query/delete chunks: {e}")
            # Try alternative deletion method
            try:
                # Delete by batch using where condition
                deleted = chunks_collection.data.delete_many(
                    where=Filter.by_property("document_id").equal(document_id)
                )
                chunks_deleted = deleted.successful
                logger.info(f"Batch deleted {chunks_deleted} chunks")
            except Exception as e2:
                logger.warning(f"Batch deletion also failed: {e2}")
                # Continue to delete document even if chunks deletion has issues
        
        # Step 2: Delete the document itself
        documents_collection = client.collections.get("Documents").with_tenant(tenant_id)
        
        try:
            documents_collection.data.delete_by_id(document_id)
            logger.info(f"Deleted document {document_id}")
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            raise
        
        result = {
            "document_id": document_id,
            "chunks_deleted": chunks_deleted,
            "status": f"Successfully deleted document and {chunks_deleted} chunks"
        }
        
        logger.info(f"Successfully deleted document {document_id} and {chunks_deleted} associated chunks")
        return result
        
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise
    finally:
        # Close client if we created it
        if client_created:
            try:
                client.close()
                logger.info("Closed Weaviate connection")
            except Exception as e:
                logger.warning(f"Error closing Weaviate connection: {e}")

def _get_weaviate_client(
    api_key: Optional[str] = None,
    url: Optional[str] = None,
    http_port: Optional[int] = None,
    grpc_port: Optional[int] = None
):
    """
    Create and connect to a Weaviate client instance.
    
    Args:
        api_key: Weaviate API key (defaults to env variable)
        url: Weaviate URL (defaults to env variable)
        http_port: HTTP port (defaults to 8080)
        grpc_port: GRPC port (defaults to 50051)
        
    Returns:
        Connected WeaviateClient instance
        
    Raises:
        Exception: If connection fails
    """
    if WeaviateClient is None:
        logger.warning("WeaviateClient is not available. Please install weaviate-client package.")
        return None
    
    api_key = api_key or Config.WEAVIATE_API_KEY
    url = url or Config.WEAVIATE_URL
    http_port = http_port or Config.WEAVIATE_HTTP_PORT
    grpc_port = grpc_port or Config.WEAVIATE_GRPC_PORT
    
    try:
        # Try newer weaviate-client API
        client = WeaviateClient(
            connection_params=ConnectionParams(
                http=ProtocolParams(host=url, port=http_port, secure=False),
                grpc=ProtocolParams(host=url, port=grpc_port, secure=False)
            ),
            auth_client_secret=AuthApiKey(api_key),
        )
        
        client.connect()
        logger.info(f"Connected to Weaviate at {url}")
        return client
    except Exception:
        # Fallback to older weaviate-client API
        try:
            import weaviate
            client = weaviate.Client(
                url=f"http://{url}:{http_port}",
                auth_client_secret=weaviate.AuthApiKey(api_key)
            )
            logger.info(f"Connected to Weaviate at {url} (legacy client)")
            return client
        except Exception as e2:
            logger.error(f"Failed to connect to Weaviate: {e2}")
            raise

def process_document_to_weaviate(
    file_path: str,
    document_id: str,
    tenant_id: str,
    use_local: Optional[bool] = True,
    client: Optional[object] = None,
    save_markdown: bool = False,
    save_chunks_json: bool = False,
    output_dir: Optional[str] = None
) -> Dict[str, Union[str, int, List[Dict]]]:
    """
    Complete pipeline to process a document and save to Weaviate.
    
    This function:
    1. Converts the document to markdown
    2. Chunks the markdown content
    3. Saves document metadata to Weaviate
    4. Saves chunks to Weaviate with references
    
    Args:
        file_path: Path to the input document (PDF or image)
        document_id: Unique identifier for the document
        tenant_id: Tenant ID for multi-tenancy
        client: Weaviate client instance (optional, will create if not provided)
        save_markdown: Whether to save markdown to file (optional)
        save_chunks_json: Whether to save chunks to JSON file (optional)
        output_dir: Directory for output files if saving (optional)
        
    Returns:
        Dictionary containing:
        - document_id: The document UUID used
        - file_name: Original file name
        - total_chunks: Number of chunks created
        - chunks: List of chunk dictionaries (if needed for further processing)
        
    Raises:
        FileNotFoundError: If the input file doesn't exist
        ValueError: If conversion or chunking fails
        Exception: If Weaviate operations fail
    """
    # Create client if not provided
    client_created = False
    if client is None:
        client = _get_weaviate_client()
        client_created = True
    
    try:
        # Step 1: Convert document to markdown
        logger.info(f"Processing document: {file_path}")
        
        markdown_path = None
        if save_markdown and output_dir:
            markdown_path = os.path.join(output_dir, f"{Path(file_path).stem}.md")
        
        markdown_content = convert_to_markdown(
            file_path=file_path,
            save_to_file=save_markdown,
            output_path=markdown_path,
            use_local=use_local
        )
        
        if not markdown_content:
            raise ValueError(f"Failed to convert document to markdown: {file_path}")
        
        # Step 2: Chunk the markdown content
        chunks_json_path = None
        if save_chunks_json and output_dir:
            chunks_json_path = os.path.join(output_dir, f"{Path(file_path).stem}_chunks.json")
        
        chunks = chunk_markdown(
            markdown_input=markdown_content,
            save_to_file=save_chunks_json,
            output_path=chunks_json_path
        )
        
        if not chunks:
            raise ValueError(f"No chunks created from document: {file_path}")
        
        # Step 3: Save document metadata to Weaviate
        saved_doc_id = save_document_to_weaviate(
            client=client,
            file_path=file_path,
            chunks=chunks,
            document_id=document_id,
            tenant_id=tenant_id
        )
        
        # Step 4: Save chunks to Weaviate
        save_chunks_to_weaviate(
            client=client,
            chunks=chunks,
            document_id=saved_doc_id,
            tenant_id=tenant_id
        )
        
        # Return summary information
        result = {
            "document_id": saved_doc_id,
            "file_name": Path(file_path).name,
            "total_chunks": len(chunks),
            "chunks": chunks  # Include chunks in case caller needs them
        }
        
        logger.info(f"Successfully processed document '{Path(file_path).name}' with {len(chunks)} chunks")
        return result
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"Processing error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing document: {e}")
        raise
    finally:
        # Close client if we created it
        if client_created:
            try:
                client.close()
                logger.info("Closed Weaviate connection")
            except Exception as e:
                logger.warning(f"Error closing Weaviate connection: {e}")