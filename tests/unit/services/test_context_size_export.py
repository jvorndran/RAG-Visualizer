"""Export measured context sizes without copying raw context or metadata."""

import json
from types import SimpleNamespace

from unravel.services.retrieval.comparison import snapshot_retrieval
from unravel.services.retrieval.export import serialize_context_size_profile


def test_profile_export_preserves_queries_sizes_and_excludes_raw_context_and_metadata():
    baseline = snapshot_retrieval("café?", [
        SimpleNamespace(index=9, text="private 🙂", score=float("nan"),
                        metadata={"api_key": "hidden"}),
    ], {"retrieval_config": {"strategy": "SparseRetriever", "api_key": "hidden"}})
    current = snapshot_retrieval("shorter?", [], {})
    document = serialize_context_size_profile(baseline, current)
    assert document.endswith("\n")
    assert document == serialize_context_size_profile(baseline, current)
    assert "café?" in document
    assert "private" not in document
    assert "hidden" not in document
    assert "score" not in document
    assert json.loads(document) == {
        "schema_version": 1,
        "scope": "full_context_size_profile",
        "measurement": "raw_chunk_text",
        "baseline": {"query": "café?", "strategy": "SparseRetriever"},
        "current": {"query": "shorter?", "strategy": "DenseRetriever"},
        "count": 1,
        "profile": [{"depth": 1, "baseline_characters": 9, "current_characters": 0,
                     "character_change": -9, "baseline_utf8_bytes": 12, "current_utf8_bytes": 0}],
    }


def test_empty_profile_export_has_an_explicit_empty_array():
    empty = snapshot_retrieval("empty", [], {})
    payload = json.loads(serialize_context_size_profile(empty, empty))
    assert payload["count"] == 0
    assert payload["profile"] == []
