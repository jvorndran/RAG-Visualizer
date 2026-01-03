# Text Splitting

## Architecture

```
rag_visualizer/services/chunking/
├── __init__.py                    # Public API
├── core.py                        # Chunk dataclass, provider registry, get_chunks()
└── providers/
    ├── __init__.py
    ├── base.py                    # ChunkingProvider ABC, SplitterInfo, ParameterInfo
    └── langchain_provider.py      # LangChain implementation
```

## Public API

```python
from rag_visualizer.services.chunking import (
    Chunk,                    # Dataclass: text, start_index, end_index, metadata
    get_chunks,               # Main function to chunk text
    get_available_providers,  # List registered provider names
    get_provider_splitters,   # Get splitter metadata for a provider
    get_provider,             # Get provider instance by name
)
```

## Usage

```python
chunks = get_chunks(
    provider="LangChain",
    splitter="RecursiveCharacterTextSplitter",
    text="Your document text...",
    chunk_size=500,
    chunk_overlap=50,
)

for chunk in chunks:
    print(chunk.text, chunk.start_index, chunk.end_index)
```

## Adding a New Provider

1. Create `rag_visualizer/services/chunking/providers/llamaindex_provider.py`:

```python
from typing import List
from ..core import Chunk
from .base import ChunkingProvider, SplitterInfo, ParameterInfo

class LlamaIndexProvider(ChunkingProvider):
    @property
    def name(self) -> str:
        return "LlamaIndex"

    @property
    def display_name(self) -> str:
        return "LlamaIndex"

    @property
    def attribution(self) -> str:
        return "Powered by LlamaIndex"

    def get_available_splitters(self) -> List[SplitterInfo]:
        return [
            SplitterInfo(
                name="SentenceSplitter",
                display_name="Sentence",
                description="Splits by sentences",
                category="NLP",
                parameters=[
                    ParameterInfo("chunk_size", "int", 500, "Chunk size", 50, 10000),
                    ParameterInfo("chunk_overlap", "int", 50, "Overlap", 0, 1000),
                ],
            ),
        ]

    def chunk(self, splitter_name: str, text: str, **params) -> List[Chunk]:
        # Implement chunking logic, return List[Chunk]
        pass
```

2. Register in `core.py`:

```python
def _init_providers() -> None:
    from .providers.langchain_provider import LangChainProvider
    from .providers.llamaindex_provider import LlamaIndexProvider

    register_provider(LangChainProvider())
    register_provider(LlamaIndexProvider())
```

3. Add dependency to `pyproject.toml`.

## Session State

Chunking params stored in `st.session_state.chunking_params`:

```python
{
    "provider": "LangChain",
    "splitter": "RecursiveCharacterTextSplitter",
    "chunk_size": 500,
    "chunk_overlap": 50,
    # ... other splitter-specific params
}
```

## Key Files

| File | Purpose |
|------|---------|
| `services/chunking/core.py` | Provider registry, `get_chunks()` with `@st.cache_data` |
| `services/chunking/providers/base.py` | `ChunkingProvider` ABC, `SplitterInfo`, `ParameterInfo` |
| `services/chunking/providers/langchain_provider.py` | LangChain splitter registry and adapter |
| `ui/sidebar.py:316-420` | Dynamic UI rendering from `SplitterInfo.parameters` |
| `ui/steps/chunks.py` | Chunk visualization |
| `ui/steps/embeddings.py:80-100` | Fallback chunk generation |

## Dependencies

```toml
# pyproject.toml
"langchain-text-splitters>=0.3"
"tiktoken>=0.5"

# Optional (for NLP splitters)
[project.optional-dependencies]
nlp = ["nltk>=3.8", "spacy>=3.7"]
```
