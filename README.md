![Unravel Demo](docs/unravel_logo.png)

# Unravel: Visualize RAG Pipelines in Real-Time

[![PyPI version](https://img.shields.io/pypi/v/unravel-rag.svg)](https://pypi.org/project/unravel-rag/)
[![Python Version](https://img.shields.io/pypi/pyversions/unravel-rag.svg)](https://pypi.org/project/unravel-rag/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://img.shields.io/pypi/dm/unravel-rag.svg)](https://pypi.org/project/unravel-rag/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Unravel** is an interactive tool to **visualize RAG (Retrieval-Augmented Generation)** pipelines. It helps developers understand, debug, and optimize their RAG systems through real-time visual inspection of document parsing, chunking, embeddings, and retrieval steps.

## Overview

Unravel is designed for developers who want to **visualize RAG** workflows and experiment with different configurations locally. Whether you are testing new chunking strategies, comparing embedding models, or fine-tuning retrieval parameters, Unravel provides the visual clarity needed to build better RAG applications.

## Demo

[**Live Demo**](https://unravel-swdurxfrvrtv96vretcssx.streamlit.app/?step=chunks)

![Unravel Demo](docs/demo.gif)

## Installation

### From PyPI

```bash
# Using pip
pip install unravel

# Or using uv (faster)
uv pip install unravel
```

### From Source

```bash
git clone https://github.com/unravel/unravel.git
cd unravel

# Create virtual environment
uv venv

# Activate virtual environment
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install in editable mode
uv sync
```

### Development Dependencies

For contributing to the project, install development tools:

```bash
uv sync --all-extras
```

> **Note**: This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management. Install uv: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"` (Windows) or see [uv installation docs](https://github.com/astral-sh/uv#installation).

## Usage

Simply run the CLI command to launch the Streamlit app:

```bash
unravel
```

This opens the app in your browser at `http://localhost:8501`.

In **Upload**, choose **Paste text** to add a source without creating a file first. Paste plain text and click **Add text** to replace the current document with a UTF-8 TXT source. Blank submissions leave the current document unchanged. Whitespace and Unicode are preserved; changed text receives a new document identity so parsing caches cannot reuse an earlier paste. Then continue to **Chunks** to parse and split it as usual. Pasted text is available in local mode.

### LLM Setup

Unravel can use **hosted LLM providers** (API key) or a **local/self-hosted LLM** (OpenAI-compatible server).

Unravel only loads keys from `~/.unravel/.env` (Windows: `%USERPROFILE%\.unravel\.env`) and **never** stores them in session/config files.

#### Option A: Hosted providers (API keys)

**Recommended:** Start with **Groq** (`GROQ_API_KEY`) — it’s usually the fastest path to “it works”, and does **not** require adding a credit card to get started. Sign up at [groq.com](https://console.groq.com/keys).

1. Run `unravel` once to create `~/.unravel/`
2. Edit `~/.unravel/.env` and add one or more keys:

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Google Gemini
GEMINI_API_KEY=...

# Groq
GROQ_API_KEY=...

# OpenRouter (access many models via one API)
OPENROUTER_API_KEY=...
```

3. Refresh the app and open **Sidebar → LLM Config** to choose:
   - **Provider**: OpenAI / Anthropic / Gemini / Groq / OpenRouter
   - **Model**: select from the list (or enter a model name for OpenRouter)

#### Option B: Local LLM (Ollama / LM Studio / any OpenAI-compatible server)

In **Sidebar → LLM Config**, choose **Provider: OpenAI-Compatible**. This routes requests to any server that implements the OpenAI Chat Completions API.

- **Ollama**
  - Install Ollama and pull a model:

```bash
ollama pull llama3.1
```

  - Base URL: `http://localhost:11434/v1`
  - Model name: `llama3.1` (or whatever you pulled)

- **LM Studio**
  - Start the local server in LM Studio
  - Base URL (common default): `http://localhost:1234/v1`
  - Model name: the model identifier LM Studio exposes (shown in its server UI)

**Note:** Local servers typically don’t require a real API key. Unravel will use a placeholder value when needed for compatibility.

### Running as a Python Module

```bash
uv run python -m unravel
```

## Features

The Text splitting step includes an **Explore chunk text** panel. Search generated raw text with a case-insensitive literal substring and narrow matches with inclusive **Minimum characters** and **Maximum characters** bounds. Counts use raw Unicode characters, including whitespace, rather than tokens; leave the maximum empty for no upper bound or set both bounds to zero to inspect empty chunks. Enable **Repeated text only** to inspect every occurrence of text appearing in multiple generated chunks, including the first copy. Repeats require exact raw-text equality, including case, whitespace, and empty text; this filter composes with text search and size bounds. Inspect the full text, original chunk number, source offsets, and metadata of each match. Choose **Prepare JSON download**, then **Download matching chunks** to export matching full text, original numbers, offsets, and applied filters (including `repeated_text_only`) as versioned Unicode JSON. An empty search with default filters exports all chunks. Preparation runs only on request and is reused until the filters or generated chunks change. The download excludes arbitrary metadata and provider settings. These controls use existing chunks locally and leave the chunking, embedding, and retrieval pipeline unchanged.

Unravel guides you through a structured 5-step pipeline for building and testing RAG systems:

### Step 1: Document Upload 📄
Advanced multi-format document ingestion with file upload and URL scraping:
- 📁 **File Upload**: PDF, DOCX, PPTX, XLSX, HTML, Markdown, TXT, and Images (PNG, JPG, BMP, TIFF)
- 🌐 **URL Scraping**: Extract content from web pages with JavaScript rendering support
  - **Crawl Modes**: Single page scraping or multi-page crawling
  - **Discovery Methods**: Crawler (follows internal links), Sitemap, or Feeds (RSS/Atom)
  - **Extraction Modes**: Balanced, Favor Precision, or Favor Recall
  - **Output Formats**: Markdown, TXT, CSV, JSON, HTML, XML, XML-TEI
  - **Metadata Extraction**: Author, publication date, description, tags/categories, site name
  - **Advanced Options**: robots.txt compliance, language filtering, content options (links, images, tables, formatting)
- 🔍 **Advanced Parsing**: OCR support for scanned documents and images
- 🏗️ **Structure Preservation**: Intelligent table structure extraction and hierarchical layout understanding
- 🎯 **Content Filtering**: Selective extraction by element type (headers, footers, code blocks, etc.)
- ⚙️ **Configurable Processing**: Thread settings and OCR options for optimal performance


### Step 2: Chunk Visualization ✂️
Flexible text splitting with transparent, visual chunk inspection:
- 🔀 **Two Chunking Strategies**:
  - **Hierarchical Chunker**: One chunk per document element, preserving semantic structure
  - **Hybrid Chunker**: Token-aware splitting with configurable limits and overlap for consistency
- 🎚️ **Token-Aware Configuration**: Set maximum chunk size (default: 512 tokens) and overlap percentage
- 🏷️ **Rich Metadata**: Attach configurable metadata to each chunk (section hierarchy, element type, token count, heading text, page numbers)
- 🎴 **Visual Chunk Cards**: Display full text with section breadcrumbs, metadata badges, and overlap highlighting
- 👁️ **Context Preview**: Expandable previews showing how chunks overlap with neighbors for seamless retrieval

### Step 3: Embedding Explorer 🧭
Visualize and analyze your document embeddings:
- 🤖 **10+ Embedding Models**: Choose from sentence-transformers models (all-MiniLM, all-mpnet, paraphrase variants), multilingual, QA-optimized, and BGE embeddings
- ⚡ **Fast Startup**: Lazy model loading ensures snappy UI responsiveness
- 🚀 **GPU Acceleration**: Automatic CUDA detection and acceleration when available
- 📊 **3D UMAP Visualization**: Interactive 3D scatter plot showing embedding space with cluster analysis
- 🎨 **Color Coding Options**: Visualize by KMeans clustering to identify semantic groupings
- 🔴 **Outlier Detection**: Identify and analyze outlier chunks in the embedding space
- 🔬 **Detailed Inspection**: Hover over points to preview chunks, click to view full details
- 🔎 **Semantic Search**: Test similarity search within embeddings with visual query point projection

### Step 4: Query Testing 🔍
End-to-end RAG testing with multiple retrieval strategies:
- 🎯 **Three Retrieval Methods**:
  - **Dense (Qdrant)**: Vector similarity search using embeddings for semantic matching
  - **Sparse (BM25)**: Keyword-based search for exact term matching
  - **Hybrid**: Combines dense and sparse with configurable fusion methods (weighted sum or reciprocal rank fusion)
- 🔄 **Query Expansion**: Generate multiple query variations with LLM to improve retrieval coverage (Reciprocal Rank Fusion for result merging)
- ⚙️ **Retrieval Configuration**: Adjust Top K results and minimum similarity score thresholds (strategy-specific defaults)
- 📈 **Reranking (Optional)**: Cross-encoder reranking to re-score and improve retrieval relevance
- 🤖 **LLM Answer Generation**: Integrate OpenAI, Anthropic, or local models (Ollama, LM Studio) to generate answers from retrieved chunks
- 🎛️ **Flexible Configuration**: Customize temperature, max tokens, system prompts, and API keys
- 📋 **Detailed Results**: View ranked chunks with similarity scores, source locations, and generated answers with full transparency
- 🌐 **API Endpoint**: Expose the entire RAG pipeline as a REST API with streaming responses for programmatic access

### Step 5: Code Export 💾
Generate production-ready Python code capturing your exact configuration:
- 📝 **Complete Implementation Code**: Exports working Python snippets for parsing, chunking, embedding, retrieval, and reranking
- 🎯 **Exact Configuration Preservation**: Every parameter and choice is captured in the generated code
- 📦 **Dependencies Management**: Complete requirements.txt with all necessary libraries and pinned versions
- ✂️ **Copy-Paste Ready**: Integration-ready code for immediate deployment to your application
- ✨ **Supports All Features**: Includes code for parsing options, chunking strategies, embedding models, retrieval methods, and LLM configuration

## Storage

All data is stored locally in `~/.unravel/`:

```
~/.unravel/
├── documents/          # Uploaded raw documents
├── chunks/             # Processed chunk data
├── embeddings/         # Cached embeddings
├── indices/            # Vector indices (Qdrant storage)
├── session_state.json  # UI state persistence
├── llm_config.json     # LLM configuration
└── rag_config.json     # RAG pipeline settings
```

No data is transmitted to external servers except when using LLM APIs for query generation.

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/unravel/unravel.git
cd unravel

# Create virtual environment
uv venv

# Activate virtual environment (Windows)
.venv\Scripts\activate
# Or macOS/Linux: source .venv/bin/activate

# Install in development mode with all dependencies
uv sync --all-extras
```

### Running Tests

```bash
uv run pytest
```

### Code Formatting

```bash
uv run black unravel
uv run ruff check unravel
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines

See [CLAUDE.md](CLAUDE.md) for coding standards and development philosophy.

## License

MIT License - see LICENSE file for details.


## Compare retrieved context

Use **Find in retrieved context** in the Query step to search this run's chunk text with case-insensitive substring matching. The count shows how many cards match, and each card retains its original retrieval rank and scores. Clear the search to restore every card. This inspection control runs locally; the generated answer and pinned comparison retain all retrieved chunks.

After asking a query, open **Compare retrieval runs** in the Query step and select **Pin current run as baseline**. Change the query, retrieval strategy, top k, minimum score, or reranking settings, then click **Ask** again. The table shows retained, added, and removed chunks with both ranks and scores; a positive rank change means the chunk moved up. Scores remain on their original scales and should not be compared numerically across retrievers or rerankers.

The pinned snapshot stays unchanged until replaced or cleared and lasts for the browser session. Comparison uses already retrieved results and makes no extra embedding, retrieval, or model calls. Chunk identity combines its index and full text, so replacing document content cannot silently match a different chunk with the same index.

Use **Show changes** to focus on added, removed, or retained chunks, and **Find in comparison text** to search their full text with case-insensitive literal matching. The controls compose, preserve original ranks, and include removed baseline text beyond the table excerpts. The visible count reflects these filters; overall change counts, pinned snapshots, generated answers, and experiment downloads remain complete. Clear the search and select **All changes** to restore the full comparison.

Select **Download retrieval experiment** in the Query step to save a JSON snapshot of the executed query, retrieval settings, full chunk text, ranks and scores. If a baseline is pinned, the download also contains its full snapshot and the comparison table. Empty-result runs can be exported. The versioned document uses JSON `null` for non-finite measurements and excludes provider configuration and arbitrary metadata; it includes the retrieved document text. Export uses only captured results and makes no additional provider calls.

Use **Inspect full chunk text** below the comparison table to read beyond an excerpt. Added and retained chunks show the current run's complete text; removed chunks show the pinned baseline's text, even when the current run returns no chunks. The selector includes each chunk's change type and original ranks.
