
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
- Swagger UI: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc

## Docker

### Build and Run
```bash
# Build image
docker build -t document-processing-api .

# Run container
docker run -p 8001:8001 \
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
- `USE_LOCAL_MARKER` - Use local Marker (default: true)


---

## Weaviate Document Management Schema

Hierarchical document storage and semantic search system for multi-tenant environments  
_Last updated: 2025-09-06_

### Embedding Configuration
- **Model**: VoyageAI `voyage-3.5-lite`
- **Dimensions**: 512  # voyage-3.5-lite has 512 dimensions
- **Vectorizer**: `text2vec-voyageai`

### Schema Overview

#### Documents Collection

| Property (name)     | Type<sup>†</sup> | Vectorised | Token-ization | Notes / Rules                                           |
|---------------------|------------------|------------|---------------|---------------------------------------------------------|
| `file_name`         | `text`           | **yes**    | `field`       | Original filename for semantic search across titles.   |
| `file_size`         | `number`         | no         | n/a           | File size in bytes. Skipped from vectorization.        |
| `total_chunks`      | `number`         | no         | n/a           | Total number of chunks created from this document.     |
| `mime_type`         | `text`           | no         | `field`       | MIME type of original file. Skipped from vectorization.|

#### Chunks Collection

| Property (name)     | Type<sup>†</sup> | Vectorised | Token-ization | Notes / Rules                                           |
|---------------------|------------------|------------|---------------|---------------------------------------------------------|
| `content`           | `text`           | **yes**    | `word`        | Primary text content for semantic search.              |
| `chunk_index`       | `number`         | no         | n/a           | Sequential position within source document (0-based).  |
| `header`            | `text`           | **yes**    | `field`       | Section title or header text associated with chunk.    |
| `document_id`       | `text`           | no         | `field`       | UUID reference to parent document. Skipped from vectorization. |
| `char_count`        | `number`         | no         | n/a           | Character count for the chunk. Skipped from vectorization. |
| `token_count`       | `number`         | no         | n/a           | Estimated token count for LLM processing. Skipped from vectorization. |
| `chunk_type`        | `text`           | no         | `field`       | Content classification from § 2.1. Skipped from vectorization. |

#### References

| Reference (name)    | Target Collection | Description                                             |
|---------------------|-------------------|---------------------------------------------------------|
| `document_object`   | `Documents`       | Hierarchical link from chunk to parent document.       |

<sup>†</sup>All data types from `weaviate.classes.config.DataType`.

### Controlled Vocabularies

#### `chunk_type`

| Code                 | Definition                                              | Examples                           |
|----------------------|---------------------------------------------------------|------------------------------------|
| `CompositeElement`   | Standard text content with mixed formatting.           | Paragraphs, mixed content blocks   |
| `Table`              | Tabular data extracted from documents.                 | CSV-like data, structured tables   |
| `Image`              | Image content with extracted text/captions.            | Charts, diagrams, photos           |
| `Header`             | Document headers and section titles.                   | H1-H6 elements, chapter titles     |
| `List`               | Bulleted or numbered list content.                     | Bullet points, enumerated items    |
| `Code`               | Source code or technical snippets.                     | Programming code, configuration    |
| `Quote`              | Quoted or citation content.                            | Block quotes, referenced material  |
| `Footer`             | Page footers and metadata.                             | Page numbers, legal disclaimers    |

### Multi-Tenancy Configuration

Both collections implement multi-tenancy with the following settings:

```python
multi_tenancy_config=Configure.multi_tenancy(
    enabled=True,
    auto_tenant_creation=True,
    auto_tenant_activation=None
)
```

#### Tenant Naming Convention

* Use consistent tenant identifiers: `user_{user_id}`, `org_{organization_id}`, or `project_{project_id}`
* Tenant names must be URL-safe alphanumeric strings
* Maximum length: 64 characters

### Data Rules

* **chunk_index** must be sequential starting from 0 within each document
* **document_id** must be a valid UUID format (e.g., `550e8400-e29b-41d4-a716-446655440000`)
* **document_id** must reference an existing UUID from the Documents collection
* **file_name** should preserve original filename for semantic searchability
* **content** chunks should target 200-500 tokens for optimal semantic coherence
* **header** can be null if no section heading is associated with the chunk
* All metadata fields use `module_config={"text2vec-voyageai": {"skip": True}}` to exclude from embeddings
* Properties that should be vectorized do not specify skip configuration

### Ingestion Pipeline

1. **Document Upload** → create entry in Documents collection with metadata
2. **Content Extraction** → parse file content using appropriate libraries
3. **Chunking Strategy** → split content maintaining semantic boundaries
4. **Header Detection** → identify and associate section headers with chunks
5. **Type Classification** → assign `chunk_type` based on content analysis
6. **UUID Generation** → use `generate_uuid5()` for consistent document IDs # Create unique IDs ( Weaviate auto-generates these)
7. **Batch Insertion** → insert chunks with references to parent document
8. **Validation** → verify reference integrity and chunk completeness

## Python Integration API

Our `integration.py` module provides a complete pipeline for document processing and Weaviate storage:

### Core Functions

#### Document Processing Pipeline
```python
from integration import process_document_to_weaviate

# Complete pipeline: PDF/image → markdown → chunks → Weaviate
result = process_document_to_weaviate(
    file_path="/path/to/document.pdf",
    document_id="doc-123",
    tenant_id="user-456"
)
print(f"Processed {result['total_chunks']} chunks")
```

#### Individual Functions
```python
from integration import convert_to_markdown, chunk_markdown, save_document_to_weaviate, save_chunks_to_weaviate

# Convert document to markdown
markdown = convert_to_markdown("/path/to/document.pdf")

# Chunk the markdown content
chunks = chunk_markdown(markdown)

# Save to Weaviate (requires Weaviate client)
client = get_weaviate_client()
save_document_to_weaviate(client, "/path/to/document.pdf", chunks, "doc-123", "user-456")
save_chunks_to_weaviate(client, chunks, "doc-123", "user-456")
```

#### Document Deletion
```python
from integration import delete_document_from_weaviate

# Delete document and all associated chunks
result = delete_document_from_weaviate("doc-123", "user-456")
print(f"Deleted {result['chunks_deleted']} chunks")
```

### Supported File Formats

- **PDF files** (`.pdf`)
- **Image files** (`.jpg`, `.jpeg`, `.png`, `.gif`, `.tiff`, `.webp`)

All files are processed through the Marker API for markdown conversion.

### Header Detection

The chunking process automatically detects headers using:
1. **Parent-child relationships** from Unstructured parsing
2. **Title-like first lines** in CompositeElements (uppercase, title case, contains colons)
3. **Special handling** for TableChunk elements ("Table Data" header)

## Retrieval Patterns

### Basic semantic search across chunks
```jsonc
response = client.collections.get("Chunks").query.near_text(
    query="machine learning algorithms",
    limit=10,
    return_properties=["content", "header", "chunk_index", "documentId"],
    return_references=["document_object"]
).with_tenant(tenant_id)
```

## Example Objects

### Document Object

```jsonc
{
  "file_name": "Machine_Learning_Research_2024.pdf",
  "file_size": 2048576,
  "total_chunks": 47,
  "mime_type": "application/pdf"
}
```

### Chunk Object

```jsonc
{
  "content": "Deep learning architectures have revolutionized natural language processing through transformer models. These attention-based mechanisms enable parallel processing of sequential data, leading to significant improvements in language understanding tasks.",
  "chunk_index": 12,
  "header": "Transformer Architectures in NLP",
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "char_count": 234,
  "token_count": 47,
  "chunk_type": "CompositeElement"
}
```

## Change Log

| Version | Date       | Changes |
|---------|------------|---------|
| **1.3.1** | 2025-09-08 | - Restructured README for improved clarity and organization<br>- Separated API documentation from Weaviate schema documentation<br>- Improved document hierarchy and removed redundant numbering<br>- Consolidated dual README content into single cohesive document |
| **1.3.0** | 2025-09-06 | - **BREAKING**: Switched from OpenAI to VoyageAI `voyage-3.5-lite` embedding model<br>- Added complete Python `integration.py` API documentation<br>- Implemented enhanced header detection with fallback logic<br>- Added document deletion functionality<br>- Updated schema to use `module_config` for vectorization control<br>- Added comprehensive file format support (PDF + images via Marker API) |
| **1.2.0** | 2025-09-05 | - Added contextual retrieval patterns and performance guidelines<br>- Expanded error handling section with retry logic<br>- Updated chunk type vocabulary with 8 standardized categories<br>- Added batch operation examples and optimization recommendations |
| **1.1.0** | 2025-08-20 | - Introduced multi-tenancy configuration documentation<br>- Added reference relationship between Chunks and Documents<br>- Defined controlled vocabulary for chunk_type classification<br>- Enhanced retrieval pattern examples with filtering |
| **1.0.0** | 2025-08-01 | - Initial schema definition for Documents and Chunks collections<br>- Established OpenAI text-embedding-3-small as standard vectorizer<br>- Defined core properties and vectorization strategy<br>- Created basic ingestion and retrieval guidelines |
