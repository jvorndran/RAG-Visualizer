"""Serialize already captured retrieval evidence for offline experiments."""

import json
import math
from dataclasses import asdict
from typing import Any

from unravel.services.retrieval.comparison import RetrievalSnapshot, compare_retrieval


def serialize_retrieval_experiment(
    current: RetrievalSnapshot, baseline: RetrievalSnapshot | None = None
) -> str:
    """Export allowlisted snapshot fields, full context and optional comparison."""
    payload = {"schema_version": 1, "current": asdict(current)}
    if baseline is not None:
        payload["baseline"] = asdict(baseline)
        payload["comparison"] = compare_retrieval(baseline, current)
    return json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _json_safe(value: Any) -> Any:
    """Represent unavailable non-finite measurements as JSON null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value
