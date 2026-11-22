"""Serialize already captured retrieval evidence for offline experiments."""

import json
import math
from dataclasses import asdict
from typing import Any

from unravel.services.retrieval.comparison import (
    RankedChunk,
    RetrievalSnapshot,
    compare_retrieval,
    context_size_profile,
    rank_overlap_profile,
)


def serialize_rank_overlap_profile(baseline: RetrievalSnapshot, current: RetrievalSnapshot) -> str:
    """Export complete rank-prefix overlap measurements without chunk text or scores."""
    profile = [
        {
            "depth": row["Depth"],
            "baseline_chunks": row["Baseline chunks"],
            "current_chunks": row["Current chunks"],
            "shared_chunks": row["Shared chunks"],
            "union_chunks": row["Union chunks"],
            "overlap_percent": row["Overlap (%)"],
        }
        for row in rank_overlap_profile(baseline, current)
    ]
    payload = {
        "schema_version": 1,
        "scope": "full_rank_overlap_profile",
        "identity": "chunk_index_and_text",
        "metric": "jaccard",
        "baseline": {"query": baseline.query, "strategy": baseline.strategy},
        "current": {"query": current.query, "strategy": current.strategy},
        "count": len(profile),
        "profile": profile,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def serialize_context_size_profile(baseline: RetrievalSnapshot, current: RetrievalSnapshot) -> str:
    """Export full-run text sizes, query labels and strategies without raw context."""
    profile = [
        {
            "depth": row["Depth"],
            "baseline_characters": row["Baseline characters"],
            "current_characters": row["Current characters"],
            "character_change": row["Character change"],
            "baseline_utf8_bytes": row["Baseline UTF-8 bytes"],
            "current_utf8_bytes": row["Current UTF-8 bytes"],
        }
        for row in context_size_profile(baseline, current)
    ]
    payload = {
        "schema_version": 1,
        "scope": "full_context_size_profile",
        "measurement": "raw_chunk_text",
        "baseline": {"query": baseline.query, "strategy": baseline.strategy},
        "current": {"query": current.query, "strategy": current.strategy},
        "count": len(profile),
        "profile": profile,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def serialize_retrieval_experiment(
    current: RetrievalSnapshot, baseline: RetrievalSnapshot | None = None
) -> str:
    """Export allowlisted snapshot fields, full context and optional comparison."""
    payload = {"schema_version": 1, "current": asdict(current)}
    if baseline is not None:
        payload["baseline"] = asdict(baseline)
        payload["comparison"] = compare_retrieval(baseline, current)
    return json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def serialize_visible_comparison(
    baseline_query: str,
    current_query: str,
    visible: list[tuple[dict[str, Any], RankedChunk]],
    filters: dict[str, Any],
) -> str:
    """Export exactly the selected comparison rows, with raw text and original ranks."""
    payload = {
        "schema_version": 1,
        "scope": "visible_comparison",
        "baseline_query": baseline_query,
        "current_query": current_query,
        "filters": filters,
        "count": len(visible),
        "chunks": [
            {
                "chunk_index": row["Chunk"],
                "change": row["Change"],
                "baseline_rank": row["Baseline rank"],
                "current_rank": row["Current rank"],
                "rank_change": row["Rank change"],
                "baseline_score": row["Baseline score"],
                "current_score": row["Current score"],
                "text": chunk.text,
            }
            for row, chunk in visible
        ],
    }
    return json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def serialize_visible_context(
    query: str, ranked_results: list[tuple[int, Any]], filters: dict[str, Any]
) -> str:
    """Export the displayed cards without deduplication or arbitrary result metadata."""
    payload = {
        "schema_version": 1,
        "scope": "visible_context",
        "query": query,
        "filters": filters,
        "count": len(ranked_results),
        "chunks": [
            {
                "chunk_index": result.index,
                "rank": rank,
                "score": result.score,
                "original_score": result.metadata.get("original_score", result.score),
                "text": result.text,
            }
            for rank, result in ranked_results
        ],
    }
    return json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _json_safe(value: object) -> object:
    """Represent unavailable non-finite measurements as JSON null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value
