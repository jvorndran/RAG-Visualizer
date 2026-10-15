"""Offline rank comparison coverage for pinned retrieval runs."""

from types import SimpleNamespace

from unravel.services.retrieval.comparison import compare_retrieval, snapshot_retrieval


def result(index, text, score=0.5):
    return SimpleNamespace(index=index, text=text, score=score)


def test_compares_rank_movement_additions_and_removals():
    baseline = snapshot_retrieval("first", [result(1, "one"), result(2, "two")], {})
    current = snapshot_retrieval("second", [result(2, "two"), result(3, "three")], {})
    rows = compare_retrieval(baseline, current)
    assert [row["Change"] for row in rows] == ["Retained", "Added", "Removed"]
    assert rows[0]["Rank change"] == 1
    assert rows[0]["Baseline rank"] == 2
    assert rows[0]["Current rank"] == 1
    assert rows[1]["Baseline rank"] is None
    assert rows[2]["Current rank"] is None
    assert rows[2]["Rank change"] is None


def test_snapshot_is_detached_from_mutable_results_and_settings():
    original = result(1, "original", 0.125)
    settings = {"top_k": 3, "threshold": 0.1, "retrieval_config": {"strategy": "DenseRetriever"}}
    settings["reranking_config"] = {"enabled": True, "model": "baseline-reranker"}
    snapshot = snapshot_retrieval("question", [original], settings)
    original.text = "changed"
    original.score = 9
    settings["retrieval_config"]["strategy"] = "SparseRetriever"
    settings["reranking_config"]["model"] = "changed-reranker"
    assert snapshot.chunks[0].text == "original"
    assert snapshot.chunks[0].score == 0.125
    assert snapshot.strategy == "DenseRetriever"
    assert snapshot.top_k == 3
    assert snapshot.threshold == 0.1
    assert snapshot.reranker == "baseline-reranker"


def test_reused_index_with_changed_text_is_a_different_chunk():
    baseline = snapshot_retrieval("q", [result(0, "old document")], {})
    current = snapshot_retrieval("q", [result(0, "new document")], {})
    assert [row["Change"] for row in compare_retrieval(baseline, current)] == ["Added", "Removed"]


def test_deduplicates_identity_and_preserves_distinct_equal_text():
    snapshot = snapshot_retrieval(
        "q", [result(1, "same"), result(1, "same"), result(2, "same")], {}
    )
    assert [chunk.rank for chunk in snapshot.chunks] == [1, 2]
    assert [chunk.index for chunk in snapshot.chunks] == [1, 2]


def test_empty_current_reports_all_baseline_chunks_removed():
    baseline = snapshot_retrieval("q", [result(0, "text")], {})
    current = snapshot_retrieval("q", [], {})
    assert compare_retrieval(baseline, current)[0]["Change"] == "Removed"
    assert compare_retrieval(current, current) == []


def test_score_scales_are_preserved_without_inventing_a_cross_strategy_delta():
    baseline = snapshot_retrieval(
        "q", [result(1, "text", 0.123456)], {"retrieval_config": {"strategy": "DenseRetriever"}}
    )
    current = snapshot_retrieval(
        "q", [result(1, "text", 9.876543)], {"retrieval_config": {"strategy": "SparseRetriever"}}
    )
    row = compare_retrieval(baseline, current)[0]
    assert row["Baseline score"] == 0.123456
    assert row["Current score"] == 9.876543
    assert "Score change" not in row
