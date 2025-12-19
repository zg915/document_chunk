"""
Utility functions for file handling, validation, and format conversion.
"""
import os
import re
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import List
from urllib.parse import urlparse

import requests
from fastapi import HTTPException, UploadFile

# ============================================================================
# FILE FORMAT CONSTANTS
# ============================================================================
# Formats that can be processed directly
DIRECTLY_SUPPORTED_FORMATS = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.tiff', '.webp',
                               '.docx', '.xlsx', '.pptx', '.html', '.htm', '.txt', '.md']

# Formats that need conversion before processing
CONVERTIBLE_FORMATS = {
    '.csv': '.xlsx',    # CSV to Excel
    '.xls': '.xlsx',    # Old Excel to new Excel
    '.ppt': '.pptx',    # Old PowerPoint to new PowerPoint
    '.pps': '.pptx',    # PowerPoint Slideshow to PowerPoint Presentation
    '.doc': '.docx',    # Old Word to new Word
    '.svg': '.png',     # SVG to PNG
    '.dot': '.docx',    # Word template to Word document
    '.rtf': '.docx',    # Rich Text Format to Word
    '.odt': '.docx'     # OpenDocument to Word
}

# All supported formats (both direct and convertible)
SUPPORTED_FORMATS = DIRECTLY_SUPPORTED_FORMATS + list(CONVERTIBLE_FORMATS.keys())

# Common error messages
ERROR_MESSAGES = {
    "FILE_OR_URL_REQUIRED": "Either file or URL must be provided",
    "FILE_OR_URL_NOT_BOTH": "Provide either file or URL, not both",
    "FILE_OR_TEXT_REQUIRED": "Either file or markdown_text must be provided",
    "FILE_OR_TEXT_NOT_BOTH": "Provide either file or markdown_text, not both",
    "DOWNLOAD_FAILED": "Failed to download file from URL",
    "PROCESSING_FAILED": "Error processing URL"
}


# ============================================================================
# FILE VALIDATION
# ============================================================================
def validate_file_format(filename: str, supported_formats: List[str]) -> None:
    """Validate uploaded file format."""
    file_ext = Path(filename).suffix.lower()
    if file_ext not in supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_ext}. Supported: {supported_formats}"
        )


# ============================================================================
# FILE HANDLING
# ============================================================================
async def save_uploaded_file(file: UploadFile) -> tuple[str, str]:
    """
    Save uploaded file to temp directory and return path with original filename.

    Returns:
        Tuple of (temp_file_path, original_filename)
    """
    # Keep the original filename for metadata storage in Weaviate/DB
    original_filename = file.filename

    # Get file extension
    file_ext = Path(file.filename).suffix.lower()

    # Use a short UUID-based temp filename to avoid "File name too long" errors
    # The original filename is preserved and returned separately for Weaviate storage
    unique_id = str(uuid.uuid4())
    safe_filename = f"tmp_{unique_id}{file_ext}"

    temp_file_path = os.path.join(tempfile.gettempdir(), safe_filename)

    with open(temp_file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    return temp_file_path, original_filename


def cleanup_temp_file(file_path: str) -> None:
    """Clean up temporary file."""
    if file_path and os.path.exists(file_path):
        try:
            os.unlink(file_path)
        except Exception:
            pass  # Silently fail cleanup


# ============================================================================
# FILE FORMAT CONVERSION
# ============================================================================
async def convert_format(file_path: str, target_ext: str) -> str:
    """
    Convert a file from one format to another.

    Args:
        file_path: Path to the source file
        target_ext: Target extension (e.g., '.xlsx', '.docx')

    Returns:
        Path to the converted file

    Raises:
        Exception: If conversion fails
    """
    source_ext = Path(file_path).suffix.lower()
    base_name = Path(file_path).stem
    output_dir = os.path.dirname(file_path)
    output_path = os.path.join(output_dir, f"{base_name}{target_ext}")

    try:
        # CSV to XLSX conversion using pandas
        if source_ext == '.csv' and target_ext == '.xlsx':
            import pandas as pd
            try:
                df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin-1', on_bad_lines='skip')
            except Exception:
                df = pd.read_csv(file_path, encoding='utf-8', encoding_errors='ignore', on_bad_lines='skip')

            df.to_excel(output_path, index=False, engine='openpyxl')
            return output_path

        # XLS to XLSX conversion using pandas
        elif source_ext == '.xls' and target_ext == '.xlsx':
            import pandas as pd
            try:
                df = pd.read_excel(file_path, engine='xlrd')
            except Exception:
                df = pd.read_excel(file_path, engine='openpyxl')

            df.to_excel(output_path, index=False, engine='openpyxl')
            return output_path

        # DOC to DOCX, PPT to PPTX, etc. using LibreOffice
        elif source_ext in ['.doc', '.ppt', '.pps', '.rtf', '.odt', '.dot'] and target_ext in ['.docx', '.pptx']:
            target_format = target_ext[1:]
            cmd = [
                'libreoffice',
                '--headless',
                '--convert-to', target_format,
                '--outdir', output_dir,
                file_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                raise Exception(f"LibreOffice conversion failed: {result.stderr}")

            if not os.path.exists(output_path):
                possible_output = os.path.join(output_dir, f"{base_name}.{target_format}")
                if os.path.exists(possible_output):
                    output_path = possible_output
                else:
                    raise FileNotFoundError(f"Converted file not found: {output_path}")

            return output_path

        # SVG to PNG conversion
        elif source_ext == '.svg' and target_ext == '.png':
            try:
                import cairosvg
                cairosvg.svg2png(url=file_path, write_to=output_path)
                return output_path
            except ImportError:
                cmd = ['inkscape', file_path, '--export-png', output_path]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

                if result.returncode != 0:
                    raise Exception(f"SVG conversion failed: {result.stderr}")

                return output_path

        else:
            raise ValueError(f"Unsupported conversion: {source_ext} to {target_ext}")

    except Exception as e:
        raise


async def convert_if_needed(file_path: str) -> str:
    """
    Convert a file to a supported format if needed.

    Args:
        file_path: Path to the source file

    Returns:
        Path to the file (original or converted)
    """
    file_ext = Path(file_path).suffix.lower()
    print(f"🔍 DEBUG convert_if_needed: file_path = {file_path}, file_ext = {file_ext}")

    # Check if conversion is needed
    if file_ext in CONVERTIBLE_FORMATS:
        target_ext = CONVERTIBLE_FORMATS[file_ext]
        print(f"🔍 DEBUG convert_if_needed: conversion needed {file_ext} -> {target_ext}")

        try:
            converted_path = await convert_format(file_path, target_ext)
            print(f"🔍 DEBUG convert_if_needed: conversion SUCCESS, converted_path = {converted_path}")
            cleanup_temp_file(file_path)
            return converted_path

        except Exception as e:
            print(f"🔍 DEBUG convert_if_needed: conversion FAILED with error: {e}")
            return file_path

    print(f"🔍 DEBUG convert_if_needed: no conversion needed, returning original")
    return file_path


# ============================================================================
# FILE ACQUISITION
# ============================================================================
async def download_file_from_url(url: str) -> tuple[str, str]:
    """
    Download a file from URL and save to temp directory.

    Args:
        url: URL of the file to download

    Returns:
        Tuple of (temp_file_path, original_filename)

    Raises:
        HTTPException: If download fails or URL is invalid
    """
    try:
        parsed_url = urlparse(url)
        original_filename = os.path.basename(parsed_url.path)
        if not original_filename:
            original_filename = "downloaded_file"

        # Determine file extension
        file_ext = Path(original_filename).suffix.lower() if '.' in original_filename else ''

        # Add extension if missing
        if not file_ext:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.head(url, headers=headers, timeout=10)
            content_type = response.headers.get('content-type', '')
            if 'pdf' in content_type:
                file_ext = '.pdf'
            elif 'image' in content_type:
                if 'jpeg' in content_type or 'jpg' in content_type:
                    file_ext = '.jpg'
                elif 'png' in content_type:
                    file_ext = '.png'
                elif 'gif' in content_type:
                    file_ext = '.gif'
                elif 'webp' in content_type:
                    file_ext = '.webp'
                else:
                    file_ext = '.jpg'
            else:
                file_ext = '.pdf'
            original_filename += file_ext

        # Download file
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()

        # Use short UUID-based temp filename to avoid "File name too long" errors
        unique_id = str(uuid.uuid4())
        safe_filename = f"tmp_{unique_id}{file_ext}"
        temp_file_path = os.path.join(tempfile.gettempdir(), safe_filename)

        with open(temp_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return temp_file_path, original_filename

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"{ERROR_MESSAGES['DOWNLOAD_FAILED']}: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{ERROR_MESSAGES['PROCESSING_FAILED']}: {str(e)}")


async def acquire_file(file: UploadFile = None, url: str = None) -> tuple[str, str]:
    """
    Acquire a file either from upload or URL, and convert if needed.

    Args:
        file: Optional uploaded file
        url: Optional URL to download from

    Returns:
        Tuple of (temp_file_path, original_filename)
        - temp_file_path: Path to the temporary file (possibly converted)
        - original_filename: Original filename for storage in Weaviate/DB

    Raises:
        HTTPException: If neither or both are provided
    """
    has_file = file and file.filename
    has_url = url and url.strip()

    if not has_file and not has_url:
        raise HTTPException(status_code=400, detail=ERROR_MESSAGES["FILE_OR_URL_REQUIRED"])

    if has_file and has_url:
        raise HTTPException(status_code=400, detail=ERROR_MESSAGES["FILE_OR_URL_NOT_BOTH"])

    # Get the file
    if has_file:
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_ext}. Supported: {SUPPORTED_FORMATS}"
            )
        temp_file_path, original_filename = await save_uploaded_file(file)
    else:
        temp_file_path, original_filename = await download_file_from_url(url)
        file_ext = Path(original_filename).suffix.lower()
        if file_ext not in SUPPORTED_FORMATS:
            cleanup_temp_file(temp_file_path)
            raise HTTPException(
                status_code=400,
                detail=f"Downloaded file has unsupported format: {file_ext}"
            )

    # Convert if needed
    final_file_path = await convert_if_needed(temp_file_path)

    # Validate final format
    final_ext = Path(final_file_path).suffix.lower()
    if final_ext not in DIRECTLY_SUPPORTED_FORMATS:
        cleanup_temp_file(final_file_path)
        raise HTTPException(
            status_code=500,
            detail=f"File conversion resulted in unsupported format: {final_ext}"
        )

    # If conversion happened, update the original filename extension
    if final_ext != Path(original_filename).suffix.lower():
        original_filename = Path(original_filename).stem + final_ext

    return final_file_path, original_filename
