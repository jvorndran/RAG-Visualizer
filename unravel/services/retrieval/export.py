"""Serialize already captured retrieval evidence for offline experiments."""

import json
import math
from dataclasses import asdict
from typing import Any

from unravel.services.retrieval.comparison import RankedChunk, RetrievalSnapshot, compare_retrieval


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


def _json_safe(value: object) -> object:
    """Represent unavailable non-finite measurements as JSON null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value
