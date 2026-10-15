"""Immutable, in-memory snapshots for comparing retrieved context."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RankedChunk:
    index: int
    text: str
    score: float
    rank: int


@dataclass(frozen=True)
class RetrievalSnapshot:
    query: str
    strategy: str
    top_k: int | None
    threshold: float | None
    reranker: str | None
    chunks: tuple[RankedChunk, ...]


def snapshot_retrieval(
    query: str, results: list[Any], settings: dict[str, Any]
) -> RetrievalSnapshot:
    """Copy only retrieval evidence, keeping credentials and mutable objects out."""
    chunks = []
    seen = set()
    for result in results:
        identity = (result.index, result.text)
        if identity in seen:
            continue
        seen.add(identity)
        chunks.append(RankedChunk(result.index, result.text, result.score, len(chunks) + 1))
    return RetrievalSnapshot(
        query=query,
        strategy=settings.get("retrieval_config", {}).get("strategy", "DenseRetriever"),
        top_k=settings.get("top_k"),
        threshold=settings.get("threshold"),
        reranker=(
            settings.get("reranking_config", {}).get("model", "ms-marco-MiniLM-L-12-v2")
            if settings.get("reranking_config", {}).get("enabled")
            else None
        ),
        chunks=tuple(chunks),
    )


def compare_retrieval(
    baseline: RetrievalSnapshot, current: RetrievalSnapshot
) -> list[dict[str, Any]]:
    """Show current ranks first, followed by chunks removed since the baseline."""
    previous = {(chunk.index, chunk.text): chunk for chunk in baseline.chunks}
    latest = {(chunk.index, chunk.text): chunk for chunk in current.chunks}
    identities = list(latest) + [identity for identity in previous if identity not in latest]
    rows = []
    for identity in identities:
        before = previous.get(identity)
        after = latest.get(identity)
        chunk = after or before
        rows.append(
            {
                "Chunk": chunk.index,
                "Change": "Retained" if before and after else "Added" if after else "Removed",
                "Baseline rank": before.rank if before else None,
                "Current rank": after.rank if after else None,
                "Rank change": before.rank - after.rank if before and after else None,
                "Baseline score": before.score if before else None,
                "Current score": after.score if after else None,
                "Excerpt": " ".join(chunk.text.split())[:160],
            }
        )
    return rows
