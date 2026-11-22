"""Full-run overlap exports retain identity semantics without exporting context."""

import json
from types import SimpleNamespace

from unravel.services.retrieval.comparison import snapshot_retrieval
from unravel.services.retrieval.export import serialize_rank_overlap_profile


def snapshot(query, records):
    return snapshot_retrieval(query, [
        SimpleNamespace(index=index, text=text, score=float("nan")) for index, text in records
    ], {"retrieval_config": {"strategy": "DenseRetriever"}, "secret": "must not escape"})


def test_rank_overlap_export_preserves_changed_text_identity_and_unequal_prefixes():
    baseline = snapshot("baseline é", [(0, "private A"), (1, "private B")])
    current = snapshot("current 🙂", [(1, "private B"), (0, "changed A"), (2, "other")])
    original = (baseline, current)
    document = serialize_rank_overlap_profile(baseline, current)
    payload = json.loads(document)
    assert payload == {
        "schema_version": 1, "scope": "full_rank_overlap_profile",
        "identity": "chunk_index_and_text", "metric": "jaccard",
        "baseline": {"query": "baseline é", "strategy": "DenseRetriever"},
        "current": {"query": "current 🙂", "strategy": "DenseRetriever"},
        "count": 3,
        "profile": [
            {"depth": 1, "baseline_chunks": 1, "current_chunks": 1,
             "shared_chunks": 0, "union_chunks": 2, "overlap_percent": 0.0},
            {"depth": 2, "baseline_chunks": 2, "current_chunks": 2,
             "shared_chunks": 1, "union_chunks": 3, "overlap_percent": 33.3},
            {"depth": 3, "baseline_chunks": 2, "current_chunks": 3,
             "shared_chunks": 1, "union_chunks": 4, "overlap_percent": 25.0},
        ],
    }
    assert document.endswith("\n") and not document.endswith("\n\n")
    assert "🙂" in document
    assert all(value not in document for value in [
        "private", "changed A", "must not escape", "NaN",
    ])
    assert (baseline, current) == original


def test_empty_rank_overlap_exports_have_no_invented_depths():
    empty = snapshot("empty", [])
    assert json.loads(serialize_rank_overlap_profile(empty, empty))["profile"] == []
    assert json.loads(serialize_rank_overlap_profile(empty, empty))["count"] == 0
    populated = snapshot("populated", [(0, ""), (1, "")])
    for baseline, current in [(empty, populated), (populated, empty)]:
        payload = json.loads(serialize_rank_overlap_profile(baseline, current))
        assert payload["count"] == 2
        assert [row["overlap_percent"] for row in payload["profile"]] == [0.0, 0.0]
        assert [row["union_chunks"] for row in payload["profile"]] == [1, 2]
