"""
Storage module for chunking markdown and saving to Weaviate.

This module provides functions to:
1. Chunk markdown content into smaller pieces
2. Save documents and chunks to Weaviate Personal_Documents/Personal_Chunks collections
3. Manage Weaviate client connections
"""
import logging
import mimetypes
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import tiktoken
from dotenv import load_dotenv
from weaviate import WeaviateClient
from weaviate.auth import AuthApiKey
from weaviate.connect import ConnectionParams, ProtocolParams

# Suppress warnings before importing unstructured
warnings.filterwarnings("ignore", message=".*short text.*")
warnings.filterwarnings("ignore", message=".*Need to load profiles.*")

# Suppress langdetect and unstructured logging warnings
logging.getLogger('langdetect').setLevel(logging.ERROR)
logging.getLogger('unstructured').setLevel(logging.ERROR)
logging.getLogger('unstructured.partition').setLevel(logging.ERROR)

# Unstructured imports
from unstructured.chunking.basic import chunk_elements
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.md import partition_md

# Import from converter module
from converter import fast_convert_to_markdown

# Load environment variables
load_dotenv('.env')

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Configuration settings for chunking and storage."""
    # Chunking parameters
    MAX_CHUNK_SIZE = 2500
    NEW_CHUNK_AFTER = 2000
    MIN_CHUNK_SIZE = 500
    CHUNK_OVERLAP = 200

    # Weaviate settings
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
    WEAVIATE_URL = os.getenv("WEAVIATE_URL")
    WEAVIATE_HTTP_PORT = 8080
    WEAVIATE_GRPC_PORT = 50051


# ============================================================================
# WEAVIATE CLIENT
# ============================================================================
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


# ============================================================================
# CHUNKING
# ============================================================================
def chunk_markdown(
    markdown_input: Union[str, Path],
) -> List[Dict]:
    """
    Chunk markdown content into smaller pieces for vector storage.

    Args:
        markdown_input: Either markdown text string or path to markdown file

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

    return final_chunks


# ============================================================================
# WEAVIATE STORAGE
# ============================================================================
def save_document_to_weaviate(
    client,
    file_path: str,
    chunks: List[Dict],
    document_id: str,
    tenant_id: str,
    document_type: Optional[str] = None,
    custom_file_path: Optional[str] = None
) -> str:
    """
    Save document metadata to Weaviate Personal_Documents collection with multi-tenancy.

    Args:
        client: Weaviate client instance
        file_path: Path to the original document file (for extracting file info)
        chunks: List of chunks (used to get total_chunks count)
        document_id: Unique identifier for the document
        tenant_id: Tenant ID for multi-tenancy
        document_type: Optional document category (contract, manual, material_list, etc.)
        custom_file_path: Optional custom file path to store (e.g., "ITS/Disney/file.pdf")

    Returns:
        The document UUID used in Weaviate
    """
    from datetime import datetime
    import pytz

    # Get file information
    file_stats = os.stat(file_path)
    file_path_obj = Path(file_path)

    file_name = file_path_obj.name
    file_size = file_stats.st_size
    mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    total_chunks = len(chunks)
    file_modified_at = datetime.fromtimestamp(file_stats.st_mtime, tz=pytz.UTC)

    # Prepare document data matching Personal_Documents schema
    document_data = {
        "document_id": document_id,
        "file_name": file_name,
        "file_path": custom_file_path if custom_file_path is not None else "",  # Store custom path as-is
        "mime_type": mime_type,
        "file_modified_at": file_modified_at,
        "file_category": document_type or "",  # Use provided type or "NA"
        "file_size": file_size,
        "total_chunks": total_chunks,
    }

    # Save to Weaviate Personal_Documents collection with tenant
    documents_collection = client.collections.get("Personal_Documents").with_tenant(tenant_id)
    document_uuid = documents_collection.data.insert(
        properties=document_data
    )

    logger.info(f"Saved document '{file_name}' to Personal_Documents (tenant: {tenant_id}) with UUID: {document_uuid}")
    return document_uuid


def save_chunks_to_weaviate(
    client,
    chunks: List[Dict],
    document_uuid: str,
    document_id: str,
    tenant_id: str,
    source_file: str
) -> None:
    """
    Save chunks to Weaviate Personal_Chunks collection with cross-reference to document.

    Args:
        client: Weaviate client instance
        chunks: List of chunk dictionaries from chunk_markdown
        document_uuid: Document UUID (Weaviate internal ID) to reference
        document_id: Document ID (custom identifier)
        tenant_id: Tenant ID for multi-tenancy
        source_file: Source file name (original document name)
    """
    chunks_collection = client.collections.get("Personal_Chunks").with_tenant(tenant_id)

    # Insert chunks with cross-reference to the document
    for i, chunk in enumerate(chunks):
        chunk_data = {
            "document_id": document_id,
            "chunk_index": chunk.get("chunk_index", i),
            "content": chunk["content"],
            "header": chunk.get("header") or "",
            "source_file": source_file,
            "chunk_type": chunk.get("chunk_type", "text"),
            "char_count": chunk.get("char_count", len(chunk["content"])),
            "token_count": chunk.get("token_count", 0),
            "extraction_method": "unstructured_v2"
        }

        # Insert with cross-reference to parent document
        chunks_collection.data.insert(
            properties=chunk_data,
            references={
                "from_document": document_uuid  # Cross-reference to the parent document
            }
        )

    logger.info(f"Saved {len(chunks)} chunks to Personal_Chunks (tenant: {tenant_id}) for document {document_id}")


# ============================================================================
# COMPLETE PIPELINE
# ============================================================================
def fast_doc_to_weaviate(
    file_path: str,
    document_id: str,
    tenant_id: str,
    client: Optional[object] = None,
    include_metadata: Optional[bool] = True,
    document_type: Optional[str] = None,
    custom_file_path: Optional[str] = None
) -> Dict[str, Union[str, int, List[Dict]]]:
    """
    Fast pipeline to process a document and save to Weaviate using fast_convert_to_markdown.

    This function:
    1. Converts the document to markdown using fast_convert_to_markdown
    2. Chunks the markdown content
    3. Saves document metadata to Personal_Documents (with tenant)
    4. Saves chunks to Personal_Chunks with cross-references (with tenant)

    Args:
        file_path: Path to the input document (PDF or image) - used for file system operations
        document_id: Unique identifier for the document
        tenant_id: Tenant ID for multi-tenancy
        client: Weaviate client instance (optional, will create if not provided)
        include_metadata: Whether to include processing metadata in output
        document_type: Optional document category (contract, manual, material_list, etc.)
        custom_file_path: Optional custom path to store in DB (can be folder or full path)

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
        # Step 1: Convert document to markdown using fast conversion
        logger.info(f"Fast processing document: {file_path}")

        markdown_content = fast_convert_to_markdown(
            file_path=file_path,
            include_metadata=include_metadata
        )

        if not markdown_content:
            raise ValueError(f"Failed to convert document to markdown: {file_path}")

        # Step 2: Chunk the markdown content
        chunks = chunk_markdown(
            markdown_input=markdown_content
        )

        if not chunks:
            raise ValueError(f"No chunks created from document: {file_path}")

        # Step 3: Save document metadata to Weaviate Personal_Documents
        document_uuid = save_document_to_weaviate(
            client=client,
            file_path=file_path,
            chunks=chunks,
            document_id=document_id,
            tenant_id=tenant_id,
            document_type=document_type,
            custom_file_path=custom_file_path
        )

        # Get source file name for chunks
        source_file_name = Path(file_path).name

        # Step 4: Save chunks to Weaviate Personal_Chunks with cross-reference
        save_chunks_to_weaviate(
            client=client,
            chunks=chunks,
            document_uuid=document_uuid,
            document_id=document_id,
            tenant_id=tenant_id,
            source_file=source_file_name
        )

        # Return summary information
        result = {
            "document_id": document_id,
            "file_name": Path(file_path).name,
            "total_chunks": len(chunks),
            "chunks": chunks  # Include chunks in case caller needs them
        }

        logger.info(f"Successfully fast processed document '{Path(file_path).name}' with {len(chunks)} chunks (tenant: {tenant_id})")
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
