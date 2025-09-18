"""
Document processing module for converting documents to markdown and chunking.

This module provides functions to:
1. Convert PDF/image files to markdown using the Marker API
2. Chunk markdown content into smaller, manageable pieces for vector storage
3. Save documents and chunks to Weaviate vector database
"""
#!/usr/bin/env python3
# Standard library imports
import json
import logging
import mimetypes
import os
import subprocess
import tempfile
import time
import uuid
import warnings
import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Third-party imports
import requests
import tiktoken
import weaviate
from dotenv import load_dotenv
from PIL import Image
from weaviate import WeaviateClient
from weaviate.auth import AuthApiKey
from weaviate.connect import ConnectionParams, ProtocolParams
from weaviate.util import generate_uuid5

# Unstructured imports
from unstructured.chunking.basic import chunk_elements
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.md import partition_md

# LangChain imports
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    UnstructuredHTMLLoader,
    UnstructuredCSVLoader,
    UnstructuredEPubLoader,
    UnstructuredODTLoader,
    UnstructuredRTFLoader,
    PyMuPDFLoader
)

# Configuration
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv('.env')


class WebhookManager:
    """Manages webhook-based document processing requests and callbacks."""

    def __init__(self):
        """Initialize the webhook manager."""
        self.pending_requests = {}
        self.logger = logging.getLogger(f"{__name__}.WebhookManager")

    def create_request(self, request_id: str, webhook_url: str) -> asyncio.Future:
        """
        Create a new webhook request and return a future for the result.

        Args:
            request_id: Unique identifier for the request
            webhook_url: URL for webhook callbacks

        Returns:
            asyncio.Future object that will be resolved when webhook callback is received
        """
        future = asyncio.Future()

        self.pending_requests[request_id] = {
            "request_id": request_id,
            "webhook_url": webhook_url,
            "future": future,
            "created_at": time.time()
        }

        self.logger.info(f"Created webhook request: {request_id}")
        return future

    def process_callback(self, request_id: str, success: bool,
                         markdown_content: Optional[str] = None,
                         extracted_data: Optional[Dict] = None,
                         error_message: Optional[str] = None) -> bool:
        """
        Process an incoming webhook callback.

        Args:
            request_id: The request ID from the callback
            success: Whether the processing was successful
            markdown_content: The converted markdown content (if successful)
            extracted_data: Additional extracted data (if any)
            error_message: Error message (if failed)

        Returns:
            True if callback was processed, False if request_id not found
        """
        if request_id not in self.pending_requests:
            self.logger.warning(f"Received callback for unknown request_id: {request_id}")
            return False

        webhook_request = self.pending_requests[request_id]
        future = webhook_request["future"]

        try:
            if success and markdown_content:
                result = {
                    "success": True,
                    "markdown_content": markdown_content,
                    "extracted_data": extracted_data or {}
                }
                future.set_result(result)
                self.logger.info(f"Successfully processed webhook callback for request_id: {request_id}")
            else:
                result = {
                    "success": False,
                    "error": error_message or "Webhook callback indicated failure"
                }
                future.set_result(result)
                self.logger.warning(f"Webhook callback failed for request_id: {request_id}")

            # Clean up the pending request
            del self.pending_requests[request_id]
            return True

        except Exception as e:
            self.logger.error(f"Error processing webhook callback for {request_id}: {e}")
            if not future.done():
                future.set_exception(e)
            return False

    def cleanup_timeout_requests(self, timeout_seconds: int = 300):
        """
        Clean up requests that have timed out.

        Args:
            timeout_seconds: Maximum age of requests before cleanup (default: 5 minutes)
        """
        current_time = time.time()
        timeout_requests = []

        for request_id, request_data in self.pending_requests.items():
            if current_time - request_data["created_at"] > timeout_seconds:
                timeout_requests.append(request_id)

        for request_id in timeout_requests:
            request_data = self.pending_requests[request_id]
            future = request_data["future"]

            if not future.done():
                future.set_exception(asyncio.TimeoutError(f"Webhook request {request_id} timed out"))

            del self.pending_requests[request_id]
            self.logger.warning(f"Cleaned up timed out request: {request_id}")

        if timeout_requests:
            self.logger.info(f"Cleaned up {len(timeout_requests)} timed out webhook requests")

    def get_pending_count(self) -> int:
        """Get the number of pending webhook requests."""
        return len(self.pending_requests)


# Global webhook manager instance
_webhook_manager = WebhookManager()


def get_webhook_manager() -> WebhookManager:
    """Get the global webhook manager instance."""
    return _webhook_manager


# Configuration
class Config:
    """Configuration settings for the document processor."""
    API_KEY = os.getenv("MARKER_API_KEY")
    API_URL = os.getenv("MARKER_API_URL")
    
    MARKER_PARAMS = {
        'output_format': 'markdown',
        'use_llm': True,
        'disable_image_extraction': True
    }
    
    SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.gif', '.tiff', '.webp']
    SUPPORTED_WORD_FORMATS = ['.docx', '.doc']
    SUPPORTED_EXCEL_FORMATS = ['.xlsx', '.xls']
    SUPPORTED_DOCUMENT_FORMATS = ['.pdf'] + SUPPORTED_IMAGE_FORMATS + SUPPORTED_WORD_FORMATS + SUPPORTED_EXCEL_FORMATS
    
    # Chunking parameters
    MAX_CHUNK_SIZE = 2500
    NEW_CHUNK_AFTER = 2000
    MIN_CHUNK_SIZE = 500
    CHUNK_OVERLAP = 200
    
    # API settings
    MAX_RETRIES = 30
    RETRY_DELAY = 10
    
    # Local Marker settings
    LOCAL_MARKER_COMMAND = "marker_single"  # Will try full path if not found
    LOCAL_MARKER_OUTPUT_DIR = os.getenv("LOCAL_MARKER_OUTPUT_DIR", "./marker_output")
    
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
    
    async def _upload_to_marker(self, file_path: str, webhook_url: str) -> Optional[str]:
        """
        Upload a file to the Marker API for processing using webhook.

        Args:
            file_path: Path to the file to upload
            webhook_url: URL for webhook callback

        Returns:
            Markdown content or None if failed
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
            elif file_ext in Config.SUPPORTED_WORD_FORMATS:
                if file_ext == '.docx':
                    mime_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                else:  # .doc
                    mime_type = 'application/msword'
                process_path = file_path
            elif file_ext in Config.SUPPORTED_EXCEL_FORMATS:
                if file_ext == '.xlsx':
                    mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                else:  # .xls
                    mime_type = 'application/vnd.ms-excel'
                process_path = file_path
            else:
                logger.error(f"Unsupported file format: {file_ext}")
                return None

            with open(process_path, 'rb') as f:
                files = {'file': (os.path.basename(process_path), f, mime_type)}

                # Add webhook parameters to the request
                data = {**Config.MARKER_PARAMS}

                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    files=files,
                    data=data
                )

            if response.status_code != 200:
                logger.error(f"API request failed with status {response.status_code}: {response.text}")
                return None

            # Get the request_id from the response
            result = response.json()
            request_id = result.get('request_id')
            if not request_id:
                logger.error("No request_id returned from Marker API")
                return None

            logger.info(f"File uploaded successfully with request_id: {request_id}")

            # Get webhook manager and create request with the actual request_id
            webhook_mgr = get_webhook_manager()
            future = webhook_mgr.create_request(request_id, webhook_url)

            # Wait for webhook callback
            try:
                webhook_result = await asyncio.wait_for(future, timeout=300)  # 5 minute timeout
                markdown_content = webhook_result.get("markdown_content")

                if not markdown_content:
                    logger.error("Webhook returned empty content")
                    return None

                return markdown_content

            except asyncio.TimeoutError:
                logger.error("Webhook processing timed out")
                return None

        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return None

        finally:
            if temp_file_created and os.path.exists(process_path):
                os.unlink(process_path)
                logger.debug(f"Cleaned up temporary file: {process_path}")
    
    
    def _process_with_local_marker(self, file_path: str) -> Optional[str]:
        """
        Process a file using local Marker installation.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Markdown content or None if failed
        """
        try:
            # Create unique output directory for this job
            job_id = str(uuid.uuid4())[:8]
            output_dir = os.path.join(Config.LOCAL_MARKER_OUTPUT_DIR, job_id)
            os.makedirs(output_dir, exist_ok=True)
            
            # Find marker_single command
            marker_cmd = "marker_single"
            possible_paths = [
                "marker_single",
                "/usr/local/bin/marker_single",
                "/usr/bin/marker_single"
            ]
            
            for path in possible_paths:
                try:
                    result = subprocess.run([path, "--help"], capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        marker_cmd = path
                        break
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    continue
            
            # Build the command with resource-friendly options
            cmd = [
                marker_cmd,
                file_path,
                "--use_llm",
                "--disable_image_extraction",
                "--equation_batch_size", "1",  # Reduce batch size to use less memory
                "--layout_batch_size", "1",
                "--table_rec_batch_size", "1",
                "--output_dir", output_dir
            ]
            
            logger.info(f"Running local Marker: {' '.join(cmd)}")
            
            # Run the command with increased timeout and better memory handling
            logger.info(f"Executing marker command...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1200,  # 20 minute timeout for larger documents
                env={**os.environ, 'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512'}
            )
            
            logger.info(f"Marker command completed with return code: {result.returncode}")
            
            if result.returncode != 0:
                if result.returncode == -9:
                    logger.error(f"Marker process was killed (return code -9), likely due to memory constraints or timeout")
                    logger.error(f"Consider: 1) Reducing document size, 2) Increasing Docker memory limit, 3) Using smaller batch size")
                else:
                    logger.error(f"Marker command failed with return code {result.returncode}")
                
                logger.error(f"STDERR output:\n{result.stderr if result.stderr else 'No stderr output'}")
                logger.error(f"STDOUT output:\n{result.stdout if result.stdout else 'No stdout output'}")
                logger.error(f"Command was: {' '.join(cmd)}")
                logger.error(f"Working directory: {os.getcwd()}")
                logger.error(f"File exists: {os.path.exists(file_path)}")
                logger.error(f"Output dir exists: {os.path.exists(output_dir)}")
                return None
            
            # Find the generated markdown file
            # Marker creates: filename/filename.md in the output directory
            base_name = Path(file_path).stem
            markdown_file = os.path.join(output_dir, base_name, f"{base_name}.md")
            
            logger.info(f"Looking for markdown file at: {markdown_file}")
            
            if not os.path.exists(markdown_file):
                logger.warning(f"Expected markdown file not found at: {markdown_file}")
                # Try to find any .md file recursively in the output directory
                md_files = list(Path(output_dir).glob("**/*.md"))
                if md_files:
                    markdown_file = str(md_files[0])
                    logger.info(f"Found markdown file at: {markdown_file}")
                else:
                    logger.error(f"No markdown file found in {output_dir}")
                    logger.error(f"Directory contents: {os.listdir(output_dir) if os.path.exists(output_dir) else 'Directory does not exist'}")
                    return None
            
            # Read the markdown content
            with open(markdown_file, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            # Clean up the temporary output directory
            try:
                import shutil
                shutil.rmtree(output_dir)
            except Exception as e:
                logger.warning(f"Could not clean up temp directory {output_dir}: {e}")
            
            logger.info(f"Successfully processed with local Marker")
            return markdown_content
            
        except subprocess.TimeoutExpired:
            logger.error("Local Marker processing timed out after 600 seconds (10 minutes)")
            return None
        except Exception as e:
            logger.error(f"Error processing with local Marker: {str(e)}")
            return None


async def convert_to_markdown(
    file_path: str,
    save_to_file: Optional[bool] = False,
    output_path: Optional[str] = None,
    use_local: Optional[bool] = None,
    webhook_url: Optional[str] = None
) -> Optional[str]:
    """
    Convert a PDF or image file to markdown format.

    Args:
        file_path: Path to the input file (PDF or supported image format)
        save_to_file: Whether to save the markdown to a file
        output_path: Optional custom output path (defaults to input_name.md)
        use_local: Whether to use local Marker (True) or API (False). Defaults to True.
        webhook_url: URL for webhook callback (required if use_local is False)

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
        if not webhook_url:
            raise ValueError("webhook_url is required when use_local is False")

        logger.info("Using Marker API with webhook for processing")
        # Upload and wait for webhook response
        markdown_content = await processor._upload_to_marker(file_path, webhook_url)
        if not markdown_content:
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


# new class and functions for fast conversion 
#create another class called Fast_DocumentProcessor
class FastFileExtractor:
    """
    Fast file extraction class for converting various document types to markdown.
    
    Supports PDFs, Word documents, Excel files, PowerPoint presentations, 
    HTML files, CSV files, and other document formats.
    """
    
    FILE_LOADERS = {
        '.pdf': 'pdf', '.doc': 'word', '.docx': 'word', '.xls': 'excel', '.xlsx': 'excel',
        '.ppt': 'powerpoint', '.pptx': 'powerpoint', '.jpg': 'image', '.jpeg': 'image',
        '.png': 'image', '.gif': 'image', '.bmp': 'image', '.tiff': 'image',
        '.html': 'html', '.htm': 'html', '.md': 'markdown', '.csv': 'csv',
        '.epub': 'epub', '.odt': 'odt', '.rtf': 'rtf', '.txt': 'text'
    }
    
    def __init__(self, use_fast_mode: bool = True, include_metadata: bool = True):
        """
        Initialize the FastFileExtractor.
        
        Args:
            use_fast_mode: Whether to use fast extraction mode (PyMuPDF for PDFs)
            include_metadata: Whether to include metadata in results
        """
        self.use_fast_mode = use_fast_mode
        self.include_metadata = include_metadata
    
    def extract(self, file_path: str) -> Dict[str, any]:
        """
        Extract content from a file and convert to markdown.
        
        Args:
            file_path: Path to the file to extract
            
        Returns:
            Dictionary containing:
            - markdown: Extracted content as markdown string
            - metadata: File metadata and extraction info
            - processing_time: Time taken for extraction
            - success: Boolean indicating if extraction was successful
            - error: Error message if extraction failed
        """
        start_time = time.time()
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = Path(file_path).suffix.lower()
        file_type = self.FILE_LOADERS.get(file_ext, 'unknown')
        
        if file_type == 'unknown':
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        try:
            if file_type == 'pdf':
                markdown, metadata = self._extract_pdf(file_path)
            elif file_type == 'word':
                markdown, metadata = self._extract_word(file_path)
            elif file_type == 'excel':
                markdown, metadata = self._extract_excel(file_path)
            elif file_type == 'powerpoint':
                markdown, metadata = self._extract_powerpoint(file_path)
            elif file_type == 'image':
                raise ValueError("Image files not supported. Convert to PDF first.")
            elif file_type in ['html', 'csv', 'markdown', 'text']:
                markdown, metadata = self._extract_simple(file_path, file_type)
            else:
                markdown, metadata = self._extract_unstructured(file_path, file_type)
            
            processing_time = time.time() - start_time
            
            if self.include_metadata:
                metadata.update({
                    'processing_time': f"{processing_time:.2f}s",
                    'file_path': file_path,
                    'file_type': file_type,
                    'file_size': f"{os.path.getsize(file_path) / 1024:.1f} KB"
                })
            
            return {
                'markdown': markdown,
                'metadata': metadata,
                'processing_time': processing_time,
                'success': True
            }
            
        except Exception as e:
            return {
                'markdown': '',
                'metadata': {'error': str(e)},
                'processing_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def _extract_pdf(self, file_path: str) -> Tuple[str, Dict]:
        """Extract content from PDF file."""
        if self.use_fast_mode:
            try:
                loader = PyMuPDFLoader(file_path)
                docs = loader.load()
                
                markdown = ""
                for i, doc in enumerate(docs):
                    markdown += f"\n{{page_{i}}}\n---\n\n{doc.page_content}\n\n"
                
                return markdown, {'total_pages': len(docs), 'loader': 'PyMuPDF'}
            except Exception:
                pass
        
        loader = UnstructuredPDFLoader(file_path, mode="single", strategy="fast")
        docs = loader.load()
        markdown = self._format_documents(docs)
        return markdown, {'total_elements': len(docs), 'loader': 'Unstructured'}
    
    def _extract_word(self, file_path: str) -> Tuple[str, Dict]:
        """Extract content from Word document."""
        loader = UnstructuredWordDocumentLoader(file_path, mode="single")
        docs = loader.load()
        return self._format_documents(docs), {'total_elements': len(docs)}
    
    def _extract_excel(self, file_path: str) -> Tuple[str, Dict]:
        """Extract content from Excel file."""
        loader = UnstructuredExcelLoader(file_path, mode="single")
        docs = loader.load()
        
        markdown = "# Excel Document\n\n"
        for doc in docs:
            content = doc.page_content
            if '\t' in content:
                markdown += self._format_table(content)
            else:
                markdown += content + "\n\n"
        
        return markdown, {'total_sheets': len(docs)}
    
    def _extract_powerpoint(self, file_path: str) -> Tuple[str, Dict]:
        """Extract content from PowerPoint presentation."""
        loader = UnstructuredPowerPointLoader(file_path, mode="single")
        docs = loader.load()
        
        markdown = "# PowerPoint Presentation\n\n"
        for doc in docs:
            markdown += doc.page_content + "\n\n"
        
        return markdown, {'total_slides': len(docs)}
    
    def _extract_simple(self, file_path: str, file_type: str) -> Tuple[str, Dict]:
        """Extract content from simple file types (text, markdown, HTML, CSV)."""
        if file_type == 'markdown':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read(), {'loader': 'Direct'}
        
        elif file_type == 'text':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return f"# Text Document\n\n{content}", {'loader': 'Direct'}
        
        elif file_type == 'html':
            loader = UnstructuredHTMLLoader(file_path, mode="single")
            docs = loader.load()
            return self._format_documents(docs), {'loader': 'HTML'}
        
        elif file_type == 'csv':
            loader = UnstructuredCSVLoader(file_path, mode="single")
            docs = loader.load()
            markdown = "# CSV Data\n\n"
            for doc in docs:
                markdown += self._format_table(doc.page_content)
            return markdown, {'loader': 'CSV'}
    
    def _extract_unstructured(self, file_path: str, file_type: str) -> Tuple[str, Dict]:
        """Extract content using unstructured loaders for specialized formats."""
        loaders = {
            'epub': UnstructuredEPubLoader,
            'odt': UnstructuredODTLoader,
            'rtf': UnstructuredRTFLoader
        }
        
        loader_class = loaders.get(file_type)
        if not loader_class:
            raise ValueError(f"No loader available for {file_type}")
        
        loader = loader_class(file_path, mode="single")
        docs = loader.load()
        return self._format_documents(docs), {'loader': file_type.upper()}
    
    def _format_documents(self, docs: List) -> str:
        """Format document list into markdown string."""
        return "\n\n".join(doc.page_content for doc in docs) + "\n"
    
    def _format_table(self, content: str) -> str:
        """Format tabular content as markdown table."""
        lines = content.strip().split('\n')
        if not lines:
            return content
        
        markdown = "\n"
        for i, line in enumerate(lines):
            if '\t' in line:
                cells = line.split('\t')
                markdown += '| ' + ' | '.join(cells) + ' |\n'
                if i == 0:
                    markdown += '|' + '---|' * len(cells) + '\n'
            else:
                markdown += line + '\n'
        
        return markdown + "\n"
    
    def extract_batch(self, file_paths: List[str]) -> List[Dict]:
        """
        Extract content from multiple files in batch.
        
        Args:
            file_paths: List of file paths to extract
            
        Returns:
            List of extraction results for each file
        """
        results = []
        logger.info(f"Starting batch extraction of {len(file_paths)} files")
        
        for i, file_path in enumerate(file_paths):
            try:
                logger.info(f"Processing file {i+1}/{len(file_paths)}: {Path(file_path).name}")
                result = self.extract(file_path)
                results.append(result)
                
            except Exception as e:
                error_result = {
                    'markdown': '',
                    'metadata': {'error': str(e), 'file_path': file_path},
                    'processing_time': 0,
                    'success': False,
                    'error': str(e)
                }
                results.append(error_result)
                logger.error(f"Failed to extract {file_path}: {e}")
        
        successful = sum(1 for r in results if r['success'])
        logger.info(f"Batch extraction completed: {successful}/{len(file_paths)} files successful")
        return results

def fast_convert_to_markdown(
    file_path: str,
    save_to_file: Optional[bool] = False,
    output_path: Optional[str] = None,
    quality: Optional[str] = "fast",
    include_metadata: Optional[bool] = True
) -> Optional[str]:
    """
    Fast convert a document file to markdown format using FastFileExtractor.
    
    Args:
        file_path: Path to the input file (supports PDF, Word, Excel, PowerPoint, images, etc.)
        save_to_file: Whether to save the markdown to a file
        output_path: Optional custom output path (defaults to input_name.md)
        quality: Conversion quality mode - "fast" or "bnced"
        include_metadata: Whether to include processing metadata in output
        
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
    if file_ext not in FastFileExtractor.FILE_LOADERS:
        supported_formats = list(FastFileExtractor.FILE_LOADERS.keys())
        raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: {supported_formats}")
    
    # Configure fast extractor based on quality setting
    use_fast_mode = quality == "fast"

    # Initialize fast extractor
    extractor = FastFileExtractor(
        use_fast_mode=use_fast_mode,
        include_metadata=include_metadata
    )
    
    logger.info(f"Fast processing file: {file_path} (quality: {quality})")
    
    # Extract content
    try:
        result = extractor.extract(file_path)
        
        if not result['success']:
            logger.error(f"Fast conversion failed: {result.get('error', 'Unknown error')}")
            return None
        
        markdown_content = result['markdown']
        processing_time = result['processing_time']
        
        if not markdown_content:
            logger.error("Fast conversion returned empty content")
            return None
        
        logger.info(f"Fast conversion completed in {processing_time:.2f}s")
        
        # Optionally append metadata to markdown
        if include_metadata and result.get('metadata'):
            metadata = result['metadata']
            markdown_content += "\n\n---\n\n"
            markdown_content += "## Processing Metadata\n\n"
            for key, value in metadata.items():
                markdown_content += f"- **{key.replace('_', ' ').title()}**: {value}\n"
        
    except Exception as e:
        logger.error(f"Fast conversion error: {e}")
        return None
    
    # Save to file if requested
    if save_to_file:
        if output_path is None:
            output_path = Path(file_path).with_suffix('.md')
        else:
            output_path = Path(output_path)
        
        try:
            output_path.write_text(markdown_content, encoding='utf-8')
            logger.info(f"Fast converted markdown saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save markdown file: {e}")
            return None
    
    return markdown_content

def chunk_markdown(
    markdown_input: Union[str, Path],
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

async def process_document_to_weaviate(
    file_path: str,
    document_id: str,
    tenant_id: str,
    use_local: Optional[bool] = True,
    webhook_url: Optional[str] = None,
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
        
        markdown_content = await convert_to_markdown(
            file_path=file_path,
            save_to_file=save_markdown,
            output_path=markdown_path,
            use_local=use_local,
            webhook_url=webhook_url
        )
        
        if not markdown_content:
            raise ValueError(f"Failed to convert document to markdown: {file_path}")
        
        # Step 2: Chunk the markdown content
        chunks_json_path = None
        if save_chunks_json and output_dir:
            chunks_json_path = os.path.join(output_dir, f"{Path(file_path).stem}_chunks.json")
        
        chunks = chunk_markdown(
            markdown_input=markdown_content
        )
        
        # Save chunks to file if requested
        if save_chunks_json and chunks_json_path:
            _save_chunks_to_file(chunks, chunks_json_path)
        
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


def fast_doc_to_weaviate(
    file_path: str,
    document_id: str,
    tenant_id: str,
    client: Optional[object] = None,
    save_markdown: bool = False,
    save_chunks_json: bool = False,
    output_dir: Optional[str] = None,
    quality: Optional[str] = "fast",
    include_metadata: Optional[bool] = True
) -> Dict[str, Union[str, int, List[Dict]]]:
    """
    Fast pipeline to process a document and save to Weaviate using fast_convert_to_markdown.
    
    This function:
    1. Converts the document to markdown using fast_convert_to_markdown
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
        quality: Conversion quality mode - "fast" or "balanced"
        include_metadata: Whether to include processing metadata in output
        
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
        
        markdown_path = None
        if save_markdown and output_dir:
            markdown_path = os.path.join(output_dir, f"{Path(file_path).stem}.md")
        
        markdown_content = fast_convert_to_markdown(
            file_path=file_path,
            save_to_file=save_markdown,
            output_path=markdown_path,
            quality=quality,
            include_metadata=include_metadata
        )
        
        if not markdown_content:
            raise ValueError(f"Failed to convert document to markdown: {file_path}")
        
        # Step 2: Chunk the markdown content
        chunks_json_path = None
        if save_chunks_json and output_dir:
            chunks_json_path = os.path.join(output_dir, f"{Path(file_path).stem}_chunks.json")
        
        chunks = chunk_markdown(
            markdown_input=markdown_content
        )
        
        # Save chunks to file if requested
        if save_chunks_json and chunks_json_path:
            _save_chunks_to_file(chunks, chunks_json_path)
        
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
        
        logger.info(f"Successfully fast processed document '{Path(file_path).name}' with {len(chunks)} chunks")
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



