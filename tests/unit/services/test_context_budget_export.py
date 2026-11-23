"""Versioned context budget measurements contain no raw chunks or scores."""

import json

from unravel.services.retrieval.comparison import RankedChunk, RetrievalSnapshot
from unravel.services.retrieval.export import serialize_context_budget


def snapshot(query, texts):
    return RetrievalSnapshot(query, "DenseRetriever", None, None, None, tuple(
        RankedChunk(i, text, float("nan"), i + 1) for i, text in enumerate(texts)
    ))


def test_export_records_full_run_budget_with_unicode_labels_and_no_context():
    baseline = snapshot("質問", ["é\n🙂", "private text"])
    current = snapshot("next", ["a", "bc"])
    document = serialize_context_budget(baseline, current, 3)
    assert document.endswith("\n")
    assert "質問" in document
    assert "private text" not in document
    assert json.loads(document) == {
        "schema_version": 1, "scope": "full_context_budget",
        "measurement": "raw_unicode_characters", "selection": "whole_chunk_rank_prefix",
        "character_budget": 3,
        "baseline": {"query": "質問", "strategy": "DenseRetriever", "included_chunks": 1,
                     "excluded_chunks": 1, "used_characters": 3, "remaining_characters": 0,
                     "next_chunk_characters": 12},
        "current": {"query": "next", "strategy": "DenseRetriever", "included_chunks": 2,
                    "excluded_chunks": 0, "used_characters": 3, "remaining_characters": 0,
                    "next_chunk_characters": None},
    }


def test_empty_budget_export_retains_null_next_size_and_zero_values():
    empty = snapshot("", [])
    payload = json.loads(serialize_context_budget(empty, empty, 0))
    assert payload["character_budget"] == 0
    assert payload["baseline"]["included_chunks"] == 0
    assert payload["current"]["next_chunk_characters"] is None
