"""LangChain text splitters provider."""

import json
from typing import Any

from langchain_text_splitters import (
    CharacterTextSplitter,
    HTMLHeaderTextSplitter,
    LatexTextSplitter,
    MarkdownHeaderTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
    TokenTextSplitter,
)

from ..core import Chunk
from .base import ChunkingProvider, ParameterInfo, SplitterInfo

# Optional imports
try:
    from langchain_text_splitters import SentenceTransformersTokenTextSplitter

    HAS_SENTENCE_TRANSFORMERS_SPLITTER = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS_SPLITTER = False

try:
    from langchain_text_splitters import NLTKTextSplitter

    HAS_NLTK_SPLITTER = True
except ImportError:
    HAS_NLTK_SPLITTER = False

try:
    from langchain_text_splitters import SpacyTextSplitter

    HAS_SPACY_SPLITTER = True
except ImportError:
    HAS_SPACY_SPLITTER = False


# Splitter registry with metadata
LANGCHAIN_SPLITTERS: dict[str, dict[str, Any]] = {
    "RecursiveCharacterTextSplitter": {
        "class": RecursiveCharacterTextSplitter,
        "display_name": "Recursive Character",
        "description": "Best for most documents. Splits by paragraphs, then sentences, then words to maintain context.",
        "category": "Character-Based",
        "parameters": [
            ParameterInfo(
                "chunk_size", "int", 500, "Target chunk size in characters", 50, 10000
            ),
            ParameterInfo(
                "chunk_overlap", "int", 50, "Overlap between chunks", 0, 1000
            ),
        ],
    },
    # TODO: Add support for other character-based seperators
    "CharacterTextSplitter": {
        "class": CharacterTextSplitter,
        "display_name": "Character",
        "description": "Simple split on a separator. Use for well-structured text with consistent delimiters.",
        "category": "Character-Based",
        "parameters": [
            ParameterInfo(
                "chunk_size", "int", 500, "Target chunk size in characters", 50, 10000
            ),
            ParameterInfo(
                "chunk_overlap", "int", 50, "Overlap between chunks", 0, 1000
            ),
            ParameterInfo("separator", "str", "\n\n", "Separator to split on"),
        ],
    },
    "TokenTextSplitter": {
        "class": TokenTextSplitter,
        "display_name": "Token",
        "description": "Splits by tokens (not characters). Use when you need precise token counts for LLM context windows.",
        "category": "Token-Based",
        "parameters": [
            ParameterInfo(
                "chunk_size", "int", 500, "Target chunk size in tokens", 50, 8000
            ),
            ParameterInfo(
                "chunk_overlap", "int", 50, "Overlap in tokens", 0, 1000
            ),
            ParameterInfo(
                "encoding_name",
                "str",
                "cl100k_base",
                "Tiktoken encoding",
                options=["cl100k_base", "p50k_base", "r50k_base", "o200k_base"],
            ),
        ],
    },
    "MarkdownTextSplitter": {
        "class": MarkdownTextSplitter,
        "display_name": "Markdown",
        "description": "Use for Markdown files. Keeps headings with their content for better semantic chunks.",
        "category": "Format-Specific",
        "parameters": [
            ParameterInfo(
                "chunk_size", "int", 500, "Target chunk size in characters", 50, 10000
            ),
            ParameterInfo(
                "chunk_overlap", "int", 50, "Overlap between chunks", 0, 1000
            ),
        ],
    },
    "LatexTextSplitter": {
        "class": LatexTextSplitter,
        "display_name": "LaTeX",
        "description": "Use for LaTeX/academic documents. Respects sections, equations, and environments.",
        "category": "Format-Specific",
        "parameters": [
            ParameterInfo(
                "chunk_size", "int", 500, "Target chunk size in characters", 50, 10000
            ),
            ParameterInfo(
                "chunk_overlap", "int", 50, "Overlap between chunks", 0, 1000
            ),
        ],
    },
    "PythonCodeTextSplitter": {
        "class": PythonCodeTextSplitter,
        "display_name": "Python Code",
        "description": "Use for Python code. Splits at function/class boundaries to keep code units intact.",
        "category": "Code",
        "parameters": [
            ParameterInfo(
                "chunk_size", "int", 500, "Target chunk size in characters", 50, 10000
            ),
            ParameterInfo(
                "chunk_overlap", "int", 50, "Overlap between chunks", 0, 1000
            ),
        ],
    },
    "HTMLHeaderTextSplitter": {
        "class": HTMLHeaderTextSplitter,
        "display_name": "HTML Header",
        "description": "Use for HTML pages. Creates chunks based on heading hierarchy (h1, h2, h3).",
        "category": "Format-Specific",
        "parameters": [],
        "special_handling": "html_header",
    },
    "MarkdownHeaderTextSplitter": {
        "class": MarkdownHeaderTextSplitter,
        "display_name": "Markdown Header",
        "description": "Use for Markdown files. Creates chunks based on heading hierarchy (#, ##, ###).",
        "category": "Format-Specific",
        "parameters": [],
        "special_handling": "markdown_header",
    },
    "RecursiveJsonSplitter": {
        "class": RecursiveJsonSplitter,
        "display_name": "JSON",
        "description": "Use for JSON data. Keeps nested objects together while respecting size limits.",
        "category": "Format-Specific",
        "parameters": [
            ParameterInfo(
                "max_chunk_size", "int", 500, "Maximum chunk size", 50, 10000
            ),
            ParameterInfo(
                "min_chunk_size", "int", 100, "Minimum chunk size", 0, 5000
            ),
        ],
        "special_handling": "json",
    },
}

# Conditionally add optional splitters
if HAS_SENTENCE_TRANSFORMERS_SPLITTER:
    LANGCHAIN_SPLITTERS["SentenceTransformersTokenTextSplitter"] = {
        "class": SentenceTransformersTokenTextSplitter,
        "display_name": "Sentence Transformers Token",
        "description": "Aligns with embedding model tokenization. Use for optimal embedding boundaries.",
        "category": "Token-Based",
        "parameters": [
            ParameterInfo(
                "tokens_per_chunk", "int", 256, "Tokens per chunk", 50, 2048
            ),
            ParameterInfo("chunk_overlap", "int", 50, "Overlap in tokens", 0, 500),
            ParameterInfo(
                "model_name",
                "str",
                "sentence-transformers/all-MiniLM-L6-v2",
                "Model for tokenization",
            ),
        ],
    }

if HAS_NLTK_SPLITTER:
    LANGCHAIN_SPLITTERS["NLTKTextSplitter"] = {
        "class": NLTKTextSplitter,
        "display_name": "NLTK Sentence",
        "description": "Uses NLTK for sentence detection. Good for natural language with proper punctuation.",
        "category": "NLP-Based",
        "parameters": [
            ParameterInfo(
                "chunk_size", "int", 500, "Target chunk size", 50, 10000
            ),
            ParameterInfo("chunk_overlap", "int", 50, "Overlap between chunks", 0, 1000),
        ],
    }

if HAS_SPACY_SPLITTER:
    LANGCHAIN_SPLITTERS["SpacyTextSplitter"] = {
        "class": SpacyTextSplitter,
        "display_name": "spaCy",
        "description": "Uses spaCy NLP for intelligent text segmentation. Best for complex natural language.",
        "category": "NLP-Based",
        "parameters": [
            ParameterInfo(
                "chunk_size", "int", 500, "Target chunk size", 50, 10000
            ),
            ParameterInfo("chunk_overlap", "int", 50, "Overlap between chunks", 0, 1000),
            ParameterInfo(
                "pipeline", "str", "en_core_web_sm", "spaCy pipeline name"
            ),
        ],
    }


class LangChainProvider(ChunkingProvider):
    """LangChain text splitters provider."""

    @property
    def name(self) -> str:
        return "LangChain"

    @property
    def display_name(self) -> str:
        return "LangChain"

    @property
    def attribution(self) -> str:
        return "Powered by LangChain"

    def get_available_splitters(self) -> list[SplitterInfo]:
        """Return all available LangChain splitters."""
        splitters = []
        for name, config in LANGCHAIN_SPLITTERS.items():
            splitters.append(
                SplitterInfo(
                    name=name,
                    display_name=config["display_name"],
                    description=config["description"],
                    parameters=config["parameters"],
                    category=config["category"],
                )
            )
        return splitters

    def chunk(
        self, splitter_name: str, text: str, **params: Any  # noqa: ANN401
    ) -> list[Chunk]:
        """Split text using specified LangChain splitter."""
        if not text:
            return []

        if splitter_name not in LANGCHAIN_SPLITTERS:
            raise ValueError(f"Unknown splitter: {splitter_name}")

        config = LANGCHAIN_SPLITTERS[splitter_name]
        special = config.get("special_handling")

        if special == "json":
            return self._chunk_json(text, **params)
        elif special == "html_header":
            return self._chunk_html_header(text, **params)
        elif special == "markdown_header":
            return self._chunk_markdown_header(text, **params)

        return self._chunk_standard(splitter_name, text, **params)

    def _chunk_standard(
        self, splitter_name: str, text: str, **params: Any  # noqa: ANN401
    ) -> list[Chunk]:
        """Handle standard text splitters."""
        config = LANGCHAIN_SPLITTERS[splitter_name]
        splitter_class = config["class"]

        # Always enable start_index tracking
        params["add_start_index"] = True

        # Create splitter instance
        splitter = splitter_class(**params)

        # Use create_documents to get Document objects with metadata
        documents = splitter.create_documents([text])

        # Convert to Chunk objects
        chunks = []
        for i, doc in enumerate(documents):
            start_idx = doc.metadata.get("start_index", 0)
            end_idx = start_idx + len(doc.page_content)

            chunks.append(
                Chunk(
                    text=doc.page_content,
                    start_index=start_idx,
                    end_index=end_idx,
                    metadata={
                        "strategy": config["display_name"],
                        "provider": "LangChain",
                        "splitter": splitter_name,
                        "size": len(doc.page_content),
                        "chunk_index": i,
                    },
                )
            )

        return chunks

    def _chunk_json(self, text: str, **params: Any) -> list[Chunk]:  # noqa: ANN401
        """Handle JSON splitting."""
        max_chunk_size = params.get("max_chunk_size", 500)
        min_chunk_size = params.get("min_chunk_size", 100)

        splitter = RecursiveJsonSplitter(
            max_chunk_size=max_chunk_size, min_chunk_size=min_chunk_size
        )

        try:
            json_data = json.loads(text)
        except json.JSONDecodeError:
            # Fall back to recursive character splitting
            return self._chunk_standard(
                "RecursiveCharacterTextSplitter",
                text,
                chunk_size=max_chunk_size,
                chunk_overlap=0,
            )

        json_chunks = splitter.split_json(json_data)

        chunks = []
        current_pos = 0
        for i, chunk_data in enumerate(json_chunks):
            chunk_text = json.dumps(chunk_data, indent=2)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    start_index=current_pos,
                    end_index=current_pos + len(chunk_text),
                    metadata={
                        "strategy": "JSON",
                        "provider": "LangChain",
                        "splitter": "RecursiveJsonSplitter",
                        "size": len(chunk_text),
                        "chunk_index": i,
                    },
                )
            )
            current_pos += len(chunk_text)

        return chunks

    def _chunk_html_header(self, text: str, **params: Any) -> list[Chunk]:  # noqa: ANN401
        """Handle HTML header splitting."""
        headers_to_split_on = [
            ("h1", "Header 1"),
            ("h2", "Header 2"),
            ("h3", "Header 3"),
        ]

        splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        try:
            documents = splitter.split_text(text)
        except Exception:
            # Fall back to recursive character splitting
            return self._chunk_standard(
                "RecursiveCharacterTextSplitter",
                text,
                chunk_size=500,
                chunk_overlap=50,
            )

        chunks = []
        current_pos = 0
        for i, doc in enumerate(documents):
            chunk_text = doc.page_content
            # Try to find actual position in text
            found_pos = text.find(chunk_text, current_pos)
            if found_pos >= 0:
                start_idx = found_pos
            else:
                start_idx = current_pos

            end_idx = start_idx + len(chunk_text)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    start_index=start_idx,
                    end_index=end_idx,
                    metadata={
                        "strategy": "HTML Header",
                        "provider": "LangChain",
                        "splitter": "HTMLHeaderTextSplitter",
                        "size": len(chunk_text),
                        "chunk_index": i,
                        **{k: v for k, v in doc.metadata.items()},
                    },
                )
            )
            current_pos = end_idx

        return chunks

    def _chunk_markdown_header(
        self, text: str, **params: Any  # noqa: ANN401
    ) -> list[Chunk]:
        """Handle Markdown header splitting."""
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        try:
            documents = splitter.split_text(text)
        except Exception:
            # Fall back to recursive character splitting
            return self._chunk_standard(
                "RecursiveCharacterTextSplitter",
                text,
                chunk_size=500,
                chunk_overlap=50,
            )

        chunks = []
        current_pos = 0
        for i, doc in enumerate(documents):
            chunk_text = doc.page_content
            # Try to find actual position in text
            found_pos = text.find(chunk_text, current_pos)
            if found_pos >= 0:
                start_idx = found_pos
            else:
                start_idx = current_pos

            end_idx = start_idx + len(chunk_text)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    start_index=start_idx,
                    end_index=end_idx,
                    metadata={
                        "strategy": "Markdown Header",
                        "provider": "LangChain",
                        "splitter": "MarkdownHeaderTextSplitter",
                        "size": len(chunk_text),
                        "chunk_index": i,
                        **{k: v for k, v in doc.metadata.items()},
                    },
                )
            )
            current_pos = end_idx

        return chunks
