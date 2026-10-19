"""Portable retrieval evidence without credentials or provider calls."""

import json
from types import SimpleNamespace
from unittest.mock import patch

from unravel.services.retrieval.comparison import snapshot_retrieval
from unravel.services.retrieval.export import serialize_retrieval_experiment


def snapshot(query="Why café?", text="Full context\n" + "é" * 500):
    return snapshot_retrieval(
        query,
        [SimpleNamespace(index=7, text=text, score=0.123456789, metadata={"secret": "hidden"})],
        {
            "top_k": 5,
            "threshold": 0.1,
            "retrieval_config": {"strategy": "SparseRetriever", "api_key": "hidden"},
        },
    )


def test_export_preserves_full_text_scores_and_executed_settings_without_mutation():
    current = snapshot()
    document = serialize_retrieval_experiment(current)
    payload = json.loads(document)
    assert payload["schema_version"] == 1
    assert payload["current"]["query"] == current.query
    assert payload["current"]["strategy"] == "SparseRetriever"
    assert payload["current"]["top_k"] == 5
    assert payload["current"]["threshold"] == 0.1
    assert payload["current"]["chunks"] == [
        {"index": 7, "text": current.chunks[0].text, "score": 0.123456789, "rank": 1}
    ]
    assert "baseline" not in payload
    assert "hidden" not in document
    assert "é" in document
    assert serialize_retrieval_experiment(current) == document


def test_export_includes_pinned_run_and_comparison_for_an_empty_current_run():
    baseline = snapshot()
    current = snapshot_retrieval("no matches", [], {"top_k": 2})
    payload = json.loads(serialize_retrieval_experiment(current, baseline))
    assert payload["current"]["chunks"] == []
    assert payload["baseline"]["chunks"][0]["text"] == baseline.chunks[0].text
    assert payload["comparison"][0]["Change"] == "Removed"


def test_download_uses_supplied_snapshot_and_keeps_session_settings_out():
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


def test_no_download_before_any_run():
    from unravel.ui.components.retrieval_export import render_retrieval_export

    with patch("unravel.ui.components.retrieval_export.st") as streamlit:
        render_retrieval_export(snapshot_retrieval("", [], {}))
        streamlit.download_button.assert_not_called()


def test_export_is_strict_json_for_unavailable_measurements():
    current = snapshot_retrieval(
        "query", [SimpleNamespace(index=0, text="context", score=float("nan"))],
        {"threshold": float("inf")},
    )
    payload = json.loads(serialize_retrieval_experiment(current, current))
    assert payload["current"]["threshold"] is None
    assert payload["current"]["chunks"][0]["score"] is None
    assert payload["comparison"][0]["Current score"] is None
