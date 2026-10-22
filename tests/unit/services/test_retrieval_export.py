"""Portable retrieval evidence without credentials or provider calls."""

import json
from types import SimpleNamespace
from unittest.mock import patch

from unravel.services.retrieval.comparison import RetrievalSnapshot, snapshot_retrieval
from unravel.services.retrieval.export import serialize_retrieval_experiment


def snapshot(
    query: str = "Why café?", text: str = "Full context\n" + "é" * 500
) -> RetrievalSnapshot:
    return snapshot_retrieval(
        query,
        [SimpleNamespace(index=7, text=text, score=0.123456789, metadata={"secret": "hidden"})],
        {
            "top_k": 5,
            "threshold": 0.1,
            "retrieval_config": {"strategy": "SparseRetriever", "api_key": "hidden"},
        },
    )


def test_export_preserves_full_text_scores_and_executed_settings_without_mutation() -> None:
    current = snapshot()
    document = serialize_retrieval_experiment(current)
    payload = json.loads(document)
    assert payload["schema_version"] == 1
    assert payload["current"]["query"] == current.query
    assert payload["current"]["strategy"] == "SparseRetriever"
    assert payload["current"]["top_k"] == 5
    assert payload["current"]["threshold"] == 0.1
    assert payload["current"]["chunks"] == [
        {"index": 7, "text": current.chunks[0].text, "score": 0.123456789,
         "rank": 1, "original_score": 0.123456789}
    ]
    assert "baseline" not in payload
    assert "hidden" not in document
    assert "é" in document
    assert serialize_retrieval_experiment(current) == document


def test_export_includes_pinned_run_and_comparison_for_an_empty_current_run() -> None:
    baseline = snapshot()
    current = snapshot_retrieval("no matches", [], {"top_k": 2})
    payload = json.loads(serialize_retrieval_experiment(current, baseline))
    assert payload["current"]["chunks"] == []
    assert payload["baseline"]["chunks"][0]["text"] == baseline.chunks[0].text
    assert payload["comparison"][0]["Change"] == "Removed"


def test_download_uses_supplied_snapshot_and_keeps_session_settings_out() -> None:
    from unravel.ui.components.retrieval_export import render_retrieval_export

    current = snapshot()
    with patch("unravel.ui.components.retrieval_export.st") as streamlit:
        streamlit.session_state = {"retrieval_comparison_baseline": snapshot("baseline")}
        render_retrieval_export(current)
        kwargs = streamlit.download_button.call_args.kwargs
        assert kwargs["mime"] == "application/json"
        assert kwargs["file_name"] == "retrieval-experiment.json"
        assert json.loads(kwargs["data"])["current"]["query"] == current.query
        assert json.loads(kwargs["data"])["baseline"]["query"] == "baseline"


def test_no_download_before_any_run() -> None:
    from unravel.ui.components.retrieval_export import render_retrieval_export

    with patch("unravel.ui.components.retrieval_export.st") as streamlit:
        render_retrieval_export(snapshot_retrieval("", [], {}))
        streamlit.download_button.assert_not_called()


def test_export_is_strict_json_for_unavailable_measurements() -> None:
    current = snapshot_retrieval(
        "query", [SimpleNamespace(index=0, text="context", score=float("nan"))],
        {"threshold": float("inf")},
    )
    payload = json.loads(serialize_retrieval_experiment(current, current))
    assert payload["current"]["threshold"] is None
    assert payload["current"]["chunks"][0]["score"] is None
    assert payload["comparison"][0]["Current score"] is None


def test_export_captures_executed_fusion_expansion_and_reranking_settings() -> None:
    settings = {
        "enable_query_expansion": True, "variation_count": 3,
        "rewrite_prompt": "Rewrite {query}",
        "query_variations": ["alternate", "question"],
        "queries_searched": ["question", "alternate"],
        "retrieval_config": {"strategy": "HybridRetriever", "params": {
            "fusion_method": "weighted_sum", "dense_weight": 0.25,
            "bm25_index_data": {"secret": "excluded"}, "api_key": "excluded",
        }},
        "reranking_config": {"enabled": True, "model": "reranker", "top_n": 2,
                             "api_key": "excluded"},
    }
    current = snapshot_retrieval("question", [], settings)
    settings["query_variations"].append("later")
    settings["retrieval_config"]["params"]["dense_weight"] = 0.9
    payload = json.loads(serialize_retrieval_experiment(current, current))
    execution = payload["current"]["execution"]
    assert execution == {
        "queries_searched": ["question", "alternate"],
        "query_variations": ["alternate", "question"],
        "expansion_enabled": True, "variation_count": 3,
        "rewrite_prompt": "Rewrite {query}", "fusion_method": "weighted_sum",
        "dense_weight": 0.25, "fusion_rrf_k": None, "query_rrf_k": 60,
        "reranker_top_n": 2,
    }
    assert payload["baseline"]["execution"] == execution
    assert "excluded" not in serialize_retrieval_experiment(current)


def test_export_captures_effective_defaults_and_original_scores() -> None:
    current = snapshot_retrieval("q", [SimpleNamespace(
        index=1, text="evidence", score=0.99,
        metadata={"original_score": 0.0123456789, "secret": "excluded"},
    )], {"retrieval_config": {"strategy": "HybridRetriever", "params": {
        "fusion_method": "rrf", "dense_weight": 0.2,
    }}, "reranking_config": {"enabled": True}})
    payload = json.loads(serialize_retrieval_experiment(current))
    assert payload["current"]["chunks"][0]["original_score"] == 0.0123456789
    assert payload["current"]["chunks"][0]["score"] == 0.99
    execution = payload["current"]["execution"]
    assert execution["queries_searched"] == ["q"]
    assert execution["fusion_rrf_k"] == 60
    assert execution["dense_weight"] is None
    assert execution["query_rrf_k"] is None
    assert execution["reranker_top_n"] == 5
    assert "excluded" not in serialize_retrieval_experiment(current)


def test_export_preserves_retrieval_score_after_expansion_and_reranking() -> None:
    from unravel.services.retrieval.reranking import RerankerConfig, rerank_results
    from unravel.services.vector_store import SearchResult
    from unravel.ui.steps.query import _merge_search_results

    raw = SearchResult(index=7, text="context", score=0.8123456789, metadata={})
    expanded = _merge_search_results([[raw], [raw]])
    with patch("unravel.services.retrieval.reranking_backends.get_backend") as backend:
        backend.return_value.rerank.return_value = [{"id": 0, "score": 0.95}]
        reranked = rerank_results("q", expanded, RerankerConfig(enabled=True))
    payload = json.loads(serialize_retrieval_experiment(snapshot_retrieval("q", reranked, {})))
    assert payload["current"]["chunks"][0]["original_score"] == raw.score
    assert payload["current"]["chunks"][0]["score"] == 0.95
    assert raw.score == 0.8123456789
    assert expanded[0].score != raw.score


def test_export_accepts_numpy_scores_returned_by_flashrank() -> None:
    import numpy as np

    from unravel.services.retrieval.reranking import RerankerConfig, rerank_results
    from unravel.services.vector_store import SearchResult

    raw = SearchResult(index=7, text="context", score=0.8, metadata={})
    with patch("unravel.services.retrieval.reranking_backends.get_backend") as backend:
        backend.return_value.rerank.return_value = [{"id": 0, "score": np.float32(0.95)}]
        reranked = rerank_results("q", [raw], RerankerConfig(enabled=True))
    payload = json.loads(serialize_retrieval_experiment(snapshot_retrieval("q", reranked, {})))
    assert payload["current"]["chunks"][0]["score"] == float(np.float32(0.95))
