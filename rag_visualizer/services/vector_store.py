"""
FAISS vector store for RAG-Visualizer.

Provides a wrapper around FAISS for storing and searching embeddings.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Lazy import to avoid slow startup
_faiss = None


def _get_faiss():  # type: ignore[no-any-return]  # noqa: ANN202
    """Lazy load FAISS to speed up imports."""
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss


@dataclass
class SearchResult:
    """Result from a vector similarity search."""
    index: int
    score: float
    text: str
    metadata: dict


class VectorStore:
    """
    FAISS-based vector store for embedding storage and retrieval.
    """

    def __init__(self, dimension: int, metric: str = "cosine") -> None:
        """Initialize the vector store.

        Args:
            dimension: Embedding dimension
            metric: Distance metric ('cosine', 'l2', or 'ip' for inner product)
        """
        self.dimension = dimension
        self.metric = metric
        self._index = None
        self._texts: list[str] = []
        self._metadata: list[dict] = []

    def _create_index(self) -> None:
        """Create the FAISS index based on the metric."""
        faiss = _get_faiss()

        if self.metric == "cosine":
            # For cosine similarity, we use inner product with normalized vectors
            self._index = faiss.IndexFlatIP(self.dimension)
        elif self.metric == "l2":
            self._index = faiss.IndexFlatL2(self.dimension)
        elif self.metric == "ip":
            self._index = faiss.IndexFlatIP(self.dimension)
        else:
            raise ValueError(
                f"Unknown metric: {self.metric}. Use 'cosine', 'l2', or 'ip'."
            )

    @property
    def index(self):  # type: ignore[no-any-return]  # noqa: ANN201
        """Get or create the FAISS index."""
        if self._index is None:
            self._create_index()
        return self._index

    @property
    def size(self) -> int:
        """Return the number of vectors in the store."""
        if self._index is None:
            return 0
        return self._index.ntotal

    def add(
        self,
        embeddings: np.ndarray,
        texts: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        """Add embeddings to the vector store.

        Args:
            embeddings: numpy array of shape (n, dimension)
            texts: List of corresponding text strings
            metadata: Optional list of metadata dicts for each embedding
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} doesn't match "
                f"store dimension {self.dimension}"
            )

        if len(texts) != embeddings.shape[0]:
            raise ValueError(
                f"Number of texts ({len(texts)}) doesn't match "
                f"number of embeddings ({embeddings.shape[0]})"
            )

        # Ensure embeddings are float32 (required by FAISS)
        embeddings = embeddings.astype(np.float32)

        # Add to FAISS index
        self.index.add(embeddings)

        # Store texts and metadata
        self._texts.extend(texts)
        if metadata:
            self._metadata.extend(metadata)
        else:
            self._metadata.extend([{} for _ in texts])

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> list[SearchResult]:
        """Search for similar vectors.

        Args:
            query_embedding: Query vector of shape (dimension,) or (1, dimension)
            k: Number of results to return

        Returns:
            List of SearchResult objects sorted by similarity (highest first)
        """
        if self.size == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Ensure float32
        query_embedding = query_embedding.astype(np.float32)

        # Limit k to actual size
        k = min(k, self.size)

        # Search
        scores, indices = self.index.search(query_embedding, k)

        # Build results
        results = []
        for _i, (idx, score) in enumerate(
            zip(indices[0], scores[0], strict=True)
        ):
            if idx < 0:  # FAISS returns -1 for missing results
                continue
            results.append(
                SearchResult(
                    index=int(idx),
                    score=float(score),
                    text=self._texts[idx],
                    metadata=self._metadata[idx],
                )
            )

        return results

    def get_all_embeddings(self) -> np.ndarray:
        """Retrieve all embeddings from the store.

        Returns:
            numpy array of shape (n, dimension)
        """
        if self.size == 0:
            return np.array([]).reshape(0, self.dimension)

        faiss = _get_faiss()
        embeddings = faiss.rev_swig_ptr(
            self.index.get_xb(), self.size * self.dimension
        ).reshape(self.size, self.dimension).copy()

        return embeddings

    def get_texts(self) -> list[str]:
        """Get all stored texts."""
        return self._texts.copy()

    def get_metadata(self) -> list[dict]:
        """Get all stored metadata."""
        return self._metadata.copy()

    def clear(self) -> None:
        """Clear all data from the store."""
        self._index = None
        self._texts = []
        self._metadata = []

    def save(self, path: Path) -> None:
        """Save the vector store to disk.

        Args:
            path: Directory path to save to (will be created if needed)
        """
        faiss = _get_faiss()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        if self._index is not None and self.size > 0:
            faiss.write_index(self.index, str(path / "index.faiss"))

        # Save metadata as JSON
        metadata = {
            "dimension": self.dimension,
            "metric": self.metric,
            "size": self.size,
            "texts": self._texts,
            "metadata": self._metadata,
        }
        with (path / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        """Load a vector store from disk.

        Args:
            path: Directory path to load from

        Returns:
            Loaded VectorStore instance
        """
        faiss = _get_faiss()
        path = Path(path)

        # Load metadata
        with (path / "metadata.json").open(encoding="utf-8") as f:
            metadata = json.load(f)

        # Create store
        store = cls(
            dimension=metadata["dimension"],
            metric=metadata["metric"],
        )

        # Load texts and metadata
        store._texts = metadata["texts"]
        store._metadata = metadata["metadata"]

        # Load FAISS index if it exists
        index_path = path / "index.faiss"
        if index_path.exists():
            store._index = faiss.read_index(str(index_path))

        return store


def create_vector_store(dimension: int, metric: str = "cosine") -> VectorStore:
    """Factory function to create a vector store.

    Args:
        dimension: Embedding dimension
        metric: Distance metric ('cosine', 'l2', or 'ip')

    Returns:
        New VectorStore instance
    """
    return VectorStore(dimension=dimension, metric=metric)

