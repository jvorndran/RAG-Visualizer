"""Budget simulations leave retrieved evidence and visible comparisons intact."""

from streamlit.testing.v1 import AppTest

APP = """
from types import SimpleNamespace
import streamlit as st
from unravel.ui.components.retrieval_comparison import render_retrieval_comparison
from unravel.services.retrieval.comparison import snapshot_retrieval
records = st.session_state.get("results", [(0, "é\\n🙂", 0.9), (1, "longer", 0.8)])
current = snapshot_retrieval("query", [
    SimpleNamespace(index=i, text=text, score=score) for i, text, score in records
], {})
render_retrieval_comparison(current)
"""


def test_budget_compares_complete_runs_and_follows_clear_and_repin():
    app = AppTest.from_string(APP, default_timeout=30).run()
    app.button(key="retrieval_comparison_pin").click().run()
    baseline = app.session_state["retrieval_comparison_baseline"]
    assert not any(item.key == "retrieval_comparison_budget_chars" for item in app.number_input)
    app.session_state["results"] = [(9, "x", 0.2), (8, "abcdef", 0.1)]
    app.run().checkbox(key="retrieval_comparison_budget").check().run()
    app.number_input(key="retrieval_comparison_budget_chars").set_value(3).run()
    frame = app.dataframe[0].value
    assert list(frame["Run"]) == ["Baseline", "Current"]
    assert list(frame["Included chunks"]) == [1, 1]
    assert list(frame["Used characters"]) == [3, 1]
    app.text_input(key="retrieval_comparison_search").input("no matches").run()
    assert len(app.dataframe) == 1
    assert list(app.dataframe[0].value["Used characters"]) == [3, 1]
    assert app.session_state["retrieval_comparison_baseline"] == baseline
    app.session_state["results"] = []
    app.run()
    assert list(app.dataframe[0].value["Included chunks"]) == [1, 0]
    app.button(key="retrieval_comparison_clear").click().run()
    assert not app.dataframe
    app.session_state["results"] = [(4, "new", 0.2)]
    app.run().button(key="retrieval_comparison_pin").click().run()
    assert not app.checkbox(key="retrieval_comparison_budget").value
    assert not app.exception


def test_budget_download_uses_active_budget_and_complete_runs():
    import json
    from unittest.mock import patch

    with patch("streamlit.download_button") as download:
        app = AppTest.from_string(APP, default_timeout=30).run()
        app.button(key="retrieval_comparison_pin").click().run()
        app.checkbox(key="retrieval_comparison_budget").check().run()
        app.number_input(key="retrieval_comparison_budget_chars").set_value(3).run()
        download.reset_mock()
        app.text_input(key="retrieval_comparison_search").input("nothing matches").run()
        calls = [call.kwargs for call in download.call_args_list
                 if call.kwargs["key"] == "retrieval_comparison_budget_download"]
        assert len(calls) == 1
        payload = json.loads(calls[0]["data"])
        assert payload["character_budget"] == 3
        assert payload["baseline"]["included_chunks"] == 1
        assert payload["current"]["included_chunks"] == 1
        assert calls[0]["mime"] == "application/json"
        app.session_state["results"] = []
        download.reset_mock()
        app.run()
        data = next(call.kwargs["data"] for call in download.call_args_list
                    if call.kwargs["key"] == "retrieval_comparison_budget_download")
        assert json.loads(data)["current"]["included_chunks"] == 0
        download.reset_mock()
        app.checkbox(key="retrieval_comparison_budget").uncheck().run()
        assert not any(call.kwargs["key"] == "retrieval_comparison_budget_download"
                       for call in download.call_args_list)
        app.checkbox(key="retrieval_comparison_budget").check().run()
        download.reset_mock()
        app.button(key="retrieval_comparison_clear").click().run()
        assert not download.call_args_list
        assert not app.exception
