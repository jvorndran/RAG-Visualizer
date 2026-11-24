"""Immutable, in-memory snapshots for comparing retrieved context."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RankedChunk:
    index: int
    text: str
    score: float
    rank: int
    original_score: float | None = None


@dataclass(frozen=True)
class RetrievalExecution:
    queries_searched: tuple[str, ...]
    query_variations: tuple[str, ...]
    expansion_enabled: bool
    variation_count: int | None
    rewrite_prompt: str | None
    fusion_method: str | None
    dense_weight: float | None
    fusion_rrf_k: int | None
    query_rrf_k: int | None
    reranker_top_n: int | None


@dataclass(frozen=True)
class RetrievalSnapshot:
    query: str
    strategy: str
    top_k: int | None
    threshold: float | None
    reranker: str | None
    chunks: tuple[RankedChunk, ...]
    execution: RetrievalExecution | None = None


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
        chunks.append(RankedChunk(
            result.index, result.text, result.score, len(chunks) + 1,
            getattr(result, "metadata", {}).get("original_score", result.score),
        ))
    strategy = settings.get("retrieval_config", {}).get("strategy", "DenseRetriever")
    params = settings.get("retrieval_config", {}).get("params", {})
    fusion = params.get("fusion_method", "weighted_sum") if strategy == "HybridRetriever" else None
    reranking = settings.get("reranking_config", {})
    queries = tuple(settings.get("queries_searched", [query]))
    expansion = settings.get("enable_query_expansion", False)
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
        execution=RetrievalExecution(
            queries_searched=queries,
            query_variations=tuple(settings.get("query_variations", [])),
            expansion_enabled=expansion,
            variation_count=settings.get("variation_count", 4) if expansion else None,
            rewrite_prompt=settings.get("rewrite_prompt") if expansion else None,
            fusion_method=fusion,
            dense_weight=params.get("dense_weight", 0.7) if fusion == "weighted_sum" else None,
            fusion_rrf_k=60 if fusion == "rrf" else None,
            query_rrf_k=60 if len(queries) > 1 else None,
            reranker_top_n=reranking.get("top_n", 5) if reranking.get("enabled") else None,
        ),
    )


def context_size_profile(
    baseline: RetrievalSnapshot, current: RetrievalSnapshot
) -> list[dict[str, int]]:
    """Measure cumulative raw text at each depth, clipping to available context."""
    baseline_chars = current_chars = baseline_bytes = current_bytes = 0
    profile = []
    for position in range(max(len(baseline.chunks), len(current.chunks))):
        if position < len(baseline.chunks):
            text = baseline.chunks[position].text
            baseline_chars += len(text)
            baseline_bytes += len(text.encode("utf-8"))
        if position < len(current.chunks):
            text = current.chunks[position].text
            current_chars += len(text)
            current_bytes += len(text.encode("utf-8"))
        profile.append({
            "Depth": position + 1,
            "Baseline characters": baseline_chars,
            "Current characters": current_chars,
            "Character change": current_chars - baseline_chars,
            "Baseline UTF-8 bytes": baseline_bytes,
            "Current UTF-8 bytes": current_bytes,
        })
    return profile


def rank_overlap_profile(
    baseline: RetrievalSnapshot, current: RetrievalSnapshot
) -> list[dict[str, Any]]:
    """Compare each run's rank prefix using shared identities divided by their union."""
    previous = set()
    latest = set()
    profile = []
    for position in range(max(len(baseline.chunks), len(current.chunks))):
        if position < len(baseline.chunks):
            chunk = baseline.chunks[position]
            previous.add((chunk.index, chunk.text))
        if position < len(current.chunks):
            chunk = current.chunks[position]
            latest.add((chunk.index, chunk.text))
        shared = len(previous & latest)
        union = len(previous | latest)
        profile.append({
            "Depth": position + 1,
            "Baseline chunks": len(previous),
            "Current chunks": len(latest),
            "Shared chunks": shared,
            "Union chunks": union,
            "Overlap (%)": round(100 * shared / union, 1),
        })
    return profile


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


def context_budget(snapshot: RetrievalSnapshot, characters: int) -> dict[str, int | None]:
    """Fit a rank prefix of whole chunks without skipping or truncating text."""
    if type(characters) is not int or characters < 0:
        raise ValueError("Character budget must be a non-negative integer.")
    used = included = 0
    next_size = None
    for chunk in snapshot.chunks:
        size = len(chunk.text)
        if used + size > characters:
            next_size = size
            break
        used += size
        included += 1
    return {
        "included_chunks": included,
        "excluded_chunks": len(snapshot.chunks) - included,
        "used_characters": used,
        "remaining_characters": characters - used,
        "next_chunk_characters": next_size,
    }


def context_redundancy(snapshot: RetrievalSnapshot) -> dict[str, int]:
    """Count excess exact-text copies across chunk indices without changing evidence."""
    seen = set()
    repeated_chunks = repeated_characters = total_characters = 0
    for chunk in snapshot.chunks:
        total_characters += len(chunk.text)
        if chunk.text in seen:
            repeated_chunks += 1
            repeated_characters += len(chunk.text)
        else:
            seen.add(chunk.text)
    return {
        "chunks": len(snapshot.chunks),
        "unique_texts": len(seen),
        "repeated_chunks": repeated_chunks,
        "total_characters": total_characters,
        "repeated_characters": repeated_characters,
    }
