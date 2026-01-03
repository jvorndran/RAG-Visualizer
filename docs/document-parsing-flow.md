# Document Parsing Flow

This document explains how documents are parsed, stored, and rendered in the RAG-Visualizer application.

## Single Document Model

The application stores **only one document at a time**. When a new document is uploaded, it replaces any existing document. This simplifies the workflow and ensures the UI always operates on "the" current document.

## Overview

The parsing flow follows this path:

```
Upload New Document
    ↓
save_document()  →  Clears existing, saves new to ~/.rag-visualizer/documents/
    ↓
get_current_document()  →  Loads the single document
    ↓
parse_document()  →  Parsed Text + Images
    ↓
Session State + Persistent Cache
    ↓
Chunking  →  Embeddings  →  Query/Retrieval
    ↓
UI Rendering
```

## 1. Entry Point: parse_document()

**Location:** `rag_visualizer/utils/parsers.py:657-715`

```python
def parse_document(filename: str, content: bytes, params: Optional[dict] = None)
    -> tuple[str, str, List[ExtractedImage]]
```

**Returns:**
- `parsed_text` (str): Extracted and processed content
- `file_format` (str): Document type (PDF, DOCX, Markdown, Text)
- `images` (List[ExtractedImage]): Extracted images with metadata

**Processing Steps:**
1. Determines file format from extension
2. Routes to format-specific parser:
   - **PDF**: `parse_pdf()` supports pypdf, docling, or LlamaParse engines
   - **DOCX**: `parse_docx()` extracts paragraphs and tables as markdown
   - **Markdown**: `parse_markdown()` preserves structure
   - **Text**: `parse_text()` decodes UTF-8
3. Applies output format conversion (markdown, plain_text, original)
4. Post-processing: whitespace normalization, special character removal
5. Inserts image placeholders for RAG indexing
6. Applies character limit cap if configured

## 2. Document Storage (Single Document)

### Storage Functions

**Location:** `rag_visualizer/services/storage.py`

```python
def save_document(filename: str, content: bytes) -> Path:
    """Save document, replacing any existing document."""
    clear_documents()  # Remove all existing documents first
    # Save new document
    ...

def get_current_document() -> tuple[str, bytes] | None:
    """Get the current (only) stored document."""
    docs = list_documents()
    if not docs:
        return None
    return (docs[0], load_document(docs[0]))

def clear_documents() -> None:
    """Clear all documents from storage."""
    ...
```

### Storage Structure

```
~/.rag-visualizer/
├── documents/          # Single raw document (only one file at a time)
├── parsed_texts/       # Cached parsed text by parameters
├── chunks/             # Processed chunk data (JSON)
├── embeddings/         # Cached embeddings
├── indices/            # FAISS vector indices
└── session/            # Session state persistence
```

### Session State Keys

| Key | Contents |
|-----|----------|
| `document_metadata` | Single document info: name, format, size, char count, preview, path |
| `doc_name` | Current document filename (kept in sync with storage) |
| `parsed_text_{hash}` | Cached parsed text for specific params |
| `chunks` | List of Chunk objects |
| `last_embeddings_result` | Vector store + embeddings + metadata |

## 3. Data Flow Through Steps

### Upload Step (`ui/steps/upload.py`)

```python
# Single file uploader (not multiple)
uploaded_file = st.file_uploader(..., accept_multiple_files=False)

# On upload: replace existing document
doc_path = save_document(uploaded_file.name, content)  # Clears old, saves new

# Store metadata (single document, not dict of documents)
st.session_state.document_metadata = {
    "name": uploaded_file.name,
    "format": file_format,
    "size_bytes": len(content),
    "char_count": char_count,
    "preview": preview,
    "path": str(doc_path),
}

# Automatically select uploaded document
st.session_state.doc_name = uploaded_file.name
```

### Chunks Step (`ui/steps/chunks.py`)

```python
# Auto-load the current document from storage
current_doc = get_current_document()
if not current_doc:
    st.info("No document uploaded...")
    return

selected_doc, doc_content = current_doc
st.session_state.doc_name = selected_doc  # Keep in sync

# Parse with current params
parsed_text, _, _ = parse_document(selected_doc, doc_content, parsing_params)

# Cache in session state
st.session_state[parsed_text_key] = parsed_text

# Generate chunks
chunks = get_chunks(provider, splitter, parsed_text, **splitter_params)
st.session_state["chunks"] = chunks
```

### Embeddings Step (`ui/steps/embeddings.py`)

```python
chunks = st.session_state.get("chunks", [])
texts = [chunk.text for chunk in chunks]
embeddings = generate_embeddings(texts, model_name)

# Store results
st.session_state["last_embeddings_result"] = {
    "vector_store": vector_store,
    "reduced_embeddings": reduced_embeddings,  # UMAP 2D projection
    "chunks": chunks,
    "model": selected_model,
}
```

## 4. UI Rendering

### Upload Step: Single Document Card

Displays the current document (or "No document" message):

```python
metadata = st.session_state.document_metadata
if metadata:
    # Show single document card with name, format, size, preview
    # Delete button clears the document
else:
    st.info("No document uploaded yet...")
```

### Chunks Step: Visualization

Custom HTML rendering with:
- Color-coded chunk backgrounds
- Overlap visualization (dashed boxes)
- Character count badges
- Markdown rendering (if configured)

### Embeddings Step: Scatter Plot

UMAP-reduced embeddings displayed via Plotly:
- Blue dots = document chunks
- Pink star = query point (if search performed)
- Dashed lines = nearest neighbors

### Query Step: Retrieved Chunks

Cards displaying:
- Chunk text content
- Similarity score (color-coded)
- Source chunk index

## 5. Configuration Parameters

Parsing parameters are set in the sidebar (`ui/sidebar.py`):

```python
st.session_state.parsing_params = {
    "pdf_parser": "pypdf" | "docling" | "llamaparse",
    "output_format": "markdown" | "original" | "plain_text",
    "normalize_whitespace": bool,
    "remove_special_chars": bool,
    "docling_enable_ocr": bool,
    "docling_table_structure": bool,
    "docling_threads": int,
    "docling_device": str,
    "docling_extract_images": bool,
    "max_characters": int,
}
```

## 6. Invalidation Cascades

When parameters change, downstream state is invalidated:

```
New document uploaded
    ↓ invalidates
Everything: chunks, embeddings, search results

Parsing params change
    ↓ invalidates
Parsed text cache, chunks, embeddings, search results

Chunking params change
    ↓ invalidates
Chunks, embeddings, search results

Embedding model change
    ↓ invalidates
Embeddings, search results
```

## 7. Image Handling

Images extracted during parsing are NOT persisted to disk. Instead:

1. **Extraction**: Docling extracts images as `ExtractedImage` objects
2. **Placeholder insertion**: `insert_image_placeholders()` appends text markers:
   ```
   ## Document Images

   [Image 1: Architecture Diagram]
   [Image 2]
   ```
3. **RAG indexing**: Placeholders included in chunked text for retrieval

## Summary

| Stage | Input | Output | Storage |
|-------|-------|--------|---------|
| Upload | Raw bytes | Document metadata | `documents/` (single file) |
| Parse | Bytes + params | Parsed text | Session state + `parsed_texts/` |
| Chunk | Parsed text | Chunk objects | Session state |
| Embed | Chunk texts | Vectors + FAISS index | Session state + `indices/` |
| Query | User query | Retrieved chunks | Session state |

## Key Design Decisions

1. **Single document at a time**: Simplifies UI and state management. New uploads replace existing documents.

2. **Auto-loading**: The Chunks step automatically loads the current document from storage via `get_current_document()` rather than requiring explicit selection.

3. **Session state sync**: `st.session_state.doc_name` is kept in sync with storage but storage is the source of truth.

4. **Invalidation on upload**: When a new document is uploaded, all derived state (chunks, embeddings, search results) is invalidated to ensure consistency.
