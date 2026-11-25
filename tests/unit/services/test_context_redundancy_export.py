"""Offline redundancy export preserves exact text equality without disclosing context."""
import json
from types import SimpleNamespace

from unravel.services.retrieval.comparison import snapshot_retrieval
from unravel.services.retrieval.export import serialize_context_redundancy


def snapshot(query, texts):
    return snapshot_retrieval(query, [
        SimpleNamespace(index=i, text=t, score=float("nan")) for i, t in enumerate(texts)
    ], {"api_key": "must-not-export"})


def test_export_uses_complete_exact_text_statistics_without_scores_or_raw_text():
    baseline = snapshot("基準", ["abc", "abc", "", "", "é", "e\u0301", "ABC", "abc "])
    current = snapshot("current", ["🧪", "🧪", "🧪"])
    raw = serialize_context_redundancy(baseline, current)
    doc = json.loads(raw)
    assert doc == {
        "schema_version": 1, "scope": "full_context_redundancy",
        "identity": "exact_raw_text", "measurement": "raw_unicode_characters",
        "baseline": {"query": "基準", "strategy": "DenseRetriever", "chunks": 8,
                     "unique_texts": 6, "repeated_chunks": 2, "total_characters": 16,
                     "repeated_characters": 3},
        "current": {"query": "current", "strategy": "DenseRetriever", "chunks": 3,
                    "unique_texts": 1, "repeated_chunks": 2, "total_characters": 3,
                    "repeated_characters": 2},
    }
    assert "abc" not in raw and "must-not-export" not in raw and "NaN" not in raw
    assert "基準" in raw and raw.endswith("\n")


def test_export_handles_empty_snapshots():
    doc = json.loads(serialize_context_redundancy(snapshot("b", []), snapshot("c", [])))
    assert doc["baseline"]["chunks"] == doc["current"]["repeated_characters"] == 0
