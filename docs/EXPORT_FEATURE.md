# Pipeline Export Feature

The Export Code feature allows users to export their configured RAG pipeline as Python code snippets that can be copied and used in their own applications.

## Overview

When a user configures their RAG pipeline in the UI (selecting a parser, chunking strategy, embedding model, etc.), they can navigate to the **Export Code** tab to generate ready-to-use Python code that replicates their exact configuration.

## Architecture

```
rag_visualizer/
├── services/
│   └── export/
│       ├── __init__.py          # Public API
│       ├── templates.py         # Code template strings
│       └── generator.py         # Template selection & parameter substitution
└── ui/
    └── steps/
        └── export.py            # Export step UI
```

## How Code Generation Works

### 1. Configuration Source

The export feature pulls configuration from Streamlit's session state. These values are set by the user through the sidebar UI:

| Session State Key | Source | Description |
|-------------------|--------|-------------|
| `applied_parsing_params` | Sidebar > RAG Config | PDF parser, output format, whitespace normalization |
| `applied_chunking_params` | Sidebar > RAG Config | Splitter type, chunk size, overlap, splitter-specific params |
| `embedding_model_name` | Sidebar > RAG Config | Selected embedding model name |
| `doc_name` | Upload step / Sidebar | Current document (used to detect file format) |

### 2. Template System

Code generation uses a template-based approach with placeholder substitution. Templates are defined in `services/export/templates.py`.

**Example template:**

```python
CHUNKING_RECURSIVE = '''"""Text Chunking with RecursiveCharacterTextSplitter

Configuration:
- Chunk Size: {chunk_size} characters
- Chunk Overlap: {chunk_overlap} characters
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size={chunk_size},
        chunk_overlap={chunk_overlap},
        length_function=len,
        add_start_index=True,
    )
    documents = splitter.create_documents([text])
    return [doc.page_content for doc in documents]
'''
```

### 3. Generator Logic

The generator (`services/export/generator.py`) performs three steps:

1. **Template Selection** - Chooses the appropriate template based on configuration
2. **Parameter Extraction** - Pulls relevant values from the config
3. **String Formatting** - Substitutes placeholders with actual values

```python
def generate_chunking_code(config: ExportConfig) -> str:
    params = config.chunking_params
    splitter = params.get("splitter", "RecursiveCharacterTextSplitter")

    # 1. Select template
    template = CHUNKING_TEMPLATES.get(splitter)

    # 2. Extract parameters
    format_kwargs = {
        "chunk_size": params.get("chunk_size", 500),
        "chunk_overlap": params.get("chunk_overlap", 50),
    }

    # 3. Format template
    return template.format(**format_kwargs)
```

### 4. Post-Processing Code

For parsing templates, additional post-processing code is dynamically generated based on user settings:

```python
def _build_post_processing(params: Dict[str, Any]) -> str:
    lines = []

    if params.get("normalize_whitespace", False):
        lines.append("    text = re.sub(r' +', ' ', text)")
        lines.append("    text = re.sub(r'\\n\\n+', '\\n\\n', text)")
        # ...

    if params.get("remove_special_chars", False):
        lines.append("    text = re.sub(r'[^\\w\\s.,-!?:;\\n]', '', text)")

    return "\n".join(lines)
```

## Template Registry

### Parsing Templates

| Template Key | When Used | Source File Reference |
|--------------|-----------|----------------------|
| `pypdf` | Default PDF parser | `utils/parsers.py:parse_pdf_pypdf()` |
| `docling` | Advanced PDF with OCR | `utils/parsers.py:parse_pdf_docling()` |
| `llamaparse` | Cloud-based parsing | `utils/parsers.py:parse_pdf_llamaparse()` |
| `PARSING_DOCX` | DOCX files | `utils/parsers.py:parse_docx()` |
| `PARSING_TEXT` | TXT/MD files | `utils/parsers.py:parse_text()` |

### Chunking Templates

| Template Key | LangChain Class | Source Reference |
|--------------|-----------------|------------------|
| `RecursiveCharacterTextSplitter` | `RecursiveCharacterTextSplitter` | `services/chunking/providers/langchain_provider.py` |
| `CharacterTextSplitter` | `CharacterTextSplitter` | Same |
| `TokenTextSplitter` | `TokenTextSplitter` | Same |
| `MarkdownTextSplitter` | `MarkdownTextSplitter` | Same |
| `LatexTextSplitter` | `LatexTextSplitter` | Same |
| `PythonCodeTextSplitter` | `PythonCodeTextSplitter` | Same |
| `HTMLHeaderTextSplitter` | `HTMLHeaderTextSplitter` | Same |
| `RecursiveJsonSplitter` | `RecursiveJsonSplitter` | Same |
| `SentenceTransformersTokenTextSplitter` | `SentenceTransformersTokenTextSplitter` | Same |
| `NLTKTextSplitter` | `NLTKTextSplitter` | Same |
| `SpacyTextSplitter` | `SpacyTextSplitter` | Same |

### Embedding Template

| Template | Models Supported | Source Reference |
|----------|------------------|------------------|
| `EMBEDDING_TEMPLATE` | all-MiniLM-L6-v2, all-mpnet-base-v2, paraphrase-MiniLM-L3-v2, multi-qa-MiniLM-L6-cos-v1 | `services/embedders.py:EMBEDDING_MODELS` |

## Dependency Mapping

The generator automatically determines required pip packages based on the configuration:

```python
PARSER_DEPENDENCIES = {
    "pypdf": ["pypdf"],
    "docling": ["docling"],
    "llamaparse": ["llama-parse"],
}

SPLITTER_DEPENDENCIES = {
    "RecursiveCharacterTextSplitter": ["langchain-text-splitters"],
    "TokenTextSplitter": ["langchain-text-splitters", "tiktoken"],
    "NLTKTextSplitter": ["langchain-text-splitters", "nltk"],
    # ...
}

EMBEDDING_DEPENDENCIES = ["sentence-transformers", "numpy"]
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Configuration                           │
│  (Sidebar: parser, splitter, chunk_size, embedding model, etc.) │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    st.session_state                              │
│  - applied_parsing_params                                        │
│  - applied_chunking_params                                       │
│  - embedding_model_name                                          │
│  - doc_name                                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ExportConfig                                │
│  Dataclass that bundles all config for code generation          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Generator Functions                           │
│  - generate_parsing_code(config)                                 │
│  - generate_chunking_code(config)                                │
│  - generate_embedding_code(config)                               │
│  - generate_installation_command(config)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Template Selection                            │
│  Based on parser type, splitter type, file format               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Parameter Substitution                          │
│  template.format(chunk_size=500, chunk_overlap=50, ...)         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Generated Code                               │
│  Ready-to-copy Python code displayed in st.code() blocks        │
└─────────────────────────────────────────────────────────────────┘
```

## UI Components

The export step (`ui/steps/export.py`) renders:

1. **Configuration Summary Card** - Shows current settings at a glance
2. **Installation Section** - `pip install` command via `st.code(..., language="bash")`
3. **Document Parsing Section** - Python code via `st.code(..., language="python")`
4. **Text Chunking Section** - Python code via `st.code(..., language="python")`
5. **Embedding Generation Section** - Python code via `st.code(..., language="python")`
6. **Full Pipeline Script** - Expandable section with combined script

Streamlit's `st.code()` component automatically provides a copy button for each code block.

## Adding New Templates

To add support for a new splitter or parser:

1. **Add the template** in `services/export/templates.py`:
   ```python
   CHUNKING_NEW_SPLITTER = '''"""Description...

   Configuration:
   - Param: {param_value}
   """

   from langchain_text_splitters import NewSplitter

   def chunk_text(text: str) -> list[str]:
       splitter = NewSplitter(param={param_value})
       # ...
   '''
   ```

2. **Register in the template registry**:
   ```python
   CHUNKING_TEMPLATES = {
       # ...
       "NewSplitter": CHUNKING_NEW_SPLITTER,
   }
   ```

3. **Add parameter extraction** in `generator.py`:
   ```python
   elif splitter == "NewSplitter":
       format_kwargs = {
           "param_value": params.get("param", default_value),
       }
   ```

4. **Add dependencies** if needed:
   ```python
   SPLITTER_DEPENDENCIES = {
       # ...
       "NewSplitter": ["langchain-text-splitters", "new-dependency"],
   }
   ```
