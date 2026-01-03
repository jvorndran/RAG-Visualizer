# 🔍 RAG Visualizer

A visual sandbox for experimenting with RAG (Retrieval-Augmented Generation) configurations.

## Overview

When building RAG systems, developers struggle to understand:
- How their documents are being chunked
- How embeddings cluster in vector space
- Which chunks get retrieved for different queries

RAG Visualizer provides an interactive tool to experiment with RAG configurations locally.

## Installation

```bash
pip install rag-visualizer
```

Or install from source:

```bash
git clone https://github.com/rag-visualizer/rag-visualizer.git
cd rag-visualizer
pip install -e .
```

### GPU Acceleration (Optional)

To significantly speed up document parsing (especially OCR/layout analysis) and embedding generation, we recommend installing PyTorch with CUDA support if you have a compatible GPU.

Visit [pytorch.org](https://pytorch.org/get-started/locally/) to find the installation command for your specific system configuration.

## Usage

Simply run the CLI command to launch the Streamlit app:

```bash
rag-visualizer
```

This opens the app in your browser at `http://localhost:8501`.

### CLI Options

```bash
rag-visualizer --help

Options:
  -p, --port INTEGER  Port to run the Streamlit app on (default: 8501)
  -h, --host TEXT     Host to bind the Streamlit app to (default: localhost)
  --version           Show the version and exit.
  --help              Show this message and exit.
```

### Running as a Python Module

```bash
python -m rag_visualizer
```

## Features

### 📄 Document Upload
- Drag-and-drop file uploader
- Support for PDF, TXT, MD, DOCX formats
- Document preview and management

### ✂️ Chunk Visualization
- Multiple chunking strategies
- Configurable parameters (chunk size, overlap)
- Visual chunk boundaries
- Chunk statistics and size distribution

### 🎯 Embedding Explorer
- Multiple embedding model options
- 2D UMAP visualization
- Color by document or cluster
- Interactive exploration

### 🔎 Query Testing
- Test retrieval queries
- Adjustable Top-K results
- Similarity scores
- Query visualization on embedding plot

## Storage

RAG Visualizer stores data locally in `~/.rag-visualizer/`:

```
~/.rag-visualizer/
├── documents/     # Uploaded raw documents
├── chunks/        # Processed chunk data
├── embeddings/    # Cached embeddings
└── indices/       # FAISS vector indices
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/rag-visualizer/rag-visualizer.git
cd rag-visualizer

# Install in development mode
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black rag_visualizer
ruff check rag_visualizer
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| UI | Streamlit |
| Visualization | Plotly |
| Document Parsing | pypdf, python-docx, markdown |
| Chunking | Custom implementations |
| Embeddings | sentence-transformers |
| Vector Search | FAISS |
| Dimensionality Reduction | UMAP |
| CLI | Click |

## License

MIT License - see LICENSE file for details.

