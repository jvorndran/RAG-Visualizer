"""Full-run redundancy diagnostics follow baseline lifecycle and ignore view filters."""

from streamlit.testing.v1 import AppTest

APP = """
from types import SimpleNamespace
import streamlit as st
from unravel.ui.components.retrieval_comparison import render_retrieval_comparison
from unravel.services.retrieval.comparison import snapshot_retrieval
texts = st.session_state.get("texts", ["abc", "abc", "x"])
render_retrieval_comparison(snapshot_retrieval("q", [
    SimpleNamespace(index=i, text=text, score=0.1) for i, text in enumerate(texts)
], {}))
"""


def test_redundancy_uses_complete_runs_and_clear_repin_lifecycle():
    app = AppTest.from_string(APP, default_timeout=30).run()
    app.button(key="retrieval_comparison_pin").click().run()
    baseline = app.session_state["retrieval_comparison_baseline"]
    assert not app.checkbox(key="retrieval_comparison_redundancy").value
    app.session_state["texts"] = ["xy", "xy", "xy"]
    app.run().checkbox(key="retrieval_comparison_redundancy").check().run()
    frame = app.dataframe[0].value
    assert list(frame["Repeated chunks"]) == [1, 2]
    assert list(frame["Repeated characters"]) == [3, 4]
    app.text_input(key="retrieval_comparison_search").input("absent").run()
    assert len(app.dataframe) == 1
    assert list(app.dataframe[0].value["Repeated characters"]) == [3, 4]
    assert app.session_state["retrieval_comparison_baseline"] == baseline
    app.session_state["texts"] = []
    app.run()
    assert list(app.dataframe[0].value["Chunks"]) == [3, 0]
    app.checkbox(key="retrieval_comparison_redundancy").uncheck().run()
    assert not app.dataframe
    app.checkbox(key="retrieval_comparison_redundancy").check().run()
    app.button(key="retrieval_comparison_clear").click().run()
    assert not app.dataframe
    app.session_state["texts"] = ["new"]
    app.run().button(key="retrieval_comparison_pin").click().run()
    assert not app.checkbox(key="retrieval_comparison_redundancy").value
    assert not app.exception


def test_redundancy_download_tracks_complete_runs_and_baseline_lifecycle():
    import json
    from unittest.mock import patch

    key = "retrieval_comparison_redundancy_download"
    with patch("streamlit.download_button") as download:
        app = AppTest.from_string(APP, default_timeout=30).run()
        app.button(key="retrieval_comparison_pin").click().run()
        assert not any(call.kwargs["key"] == key for call in download.call_args_list)
        app.session_state["texts"] = ["xy", "xy", "xy"]
        app.run().checkbox(key="retrieval_comparison_redundancy").check().run()
        download.reset_mock()
        app.text_input(key="retrieval_comparison_search").input("nothing matches").run()
        calls = [call.kwargs for call in download.call_args_list if call.kwargs["key"] == key]
        assert len(calls) == 1
        doc = json.loads(calls[0]["data"])
        assert doc["baseline"]["repeated_characters"] == 3
        assert doc["current"]["repeated_characters"] == 4
        assert calls[0]["mime"] == "application/json"
        assert calls[0]["file_name"] == "retrieval-context-redundancy.json"
        app.session_state["texts"] = []
        download.reset_mock()
        app.run()
        doc = json.loads(next(call.kwargs["data"] for call in download.call_args_list
                              if call.kwargs["key"] == key))
        assert doc["current"]["chunks"] == 0
        download.reset_mock()
        app.checkbox(key="retrieval_comparison_redundancy").uncheck().run()
        assert not any(call.kwargs["key"] == key for call in download.call_args_list)
        app.checkbox(key="retrieval_comparison_redundancy").check().run()
        download.reset_mock()
        app.button(key="retrieval_comparison_clear").click().run()
        assert not download.call_args_list
        app.session_state["texts"] = ["new", "new"]
        app.run().button(key="retrieval_comparison_pin").click().run()
        app.checkbox(key="retrieval_comparison_redundancy").check().run()
        doc = json.loads(next(call.kwargs["data"] for call in reversed(download.call_args_list)
                              if call.kwargs["key"] == key))
        assert doc["baseline"]["repeated_characters"] == 3
        assert not app.exception
