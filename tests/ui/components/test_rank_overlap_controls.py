"""Exercise the profile toggle and full-snapshot scope without provider calls."""

import json
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

APP = """
from types import SimpleNamespace
import streamlit as st
from unravel.ui.components.retrieval_comparison import render_retrieval_comparison
from unravel.services.retrieval.comparison import snapshot_retrieval
records = st.session_state.get("results", [(0, "a", 0.9), (1, "b", 0.8)])
current = snapshot_retrieval("query", [
    SimpleNamespace(index=i, text=text, score=score) for i, text, score in records
], {})
render_retrieval_comparison(current)
"""


def test_profile_uses_full_runs_despite_hidden_comparison_rows():
    app = AppTest.from_string(APP, default_timeout=30).run()
    assert len(app.checkbox) == 0
    app.button(key="retrieval_comparison_pin").click().run()
    assert not app.checkbox(key="retrieval_comparison_overlap").value
    assert len(app.dataframe) == 1
    app.session_state["results"] = [(1, "b", 0.1), (0, "a", 0.2), (2, "c", 0.3)]
    app.run()
    app.checkbox(key="retrieval_comparison_overlap").check().run()
    profile = app.dataframe[0].value.copy()
    assert list(profile["Shared chunks"]) == [0, 2, 2]
    assert list(profile["Overlap (%)"]) == [0.0, 100.0, 66.7]
    app.text_input(key="retrieval_comparison_search").input("no match").run()
    app.number_input(key="retrieval_comparison_top_ranks").set_value(1).run()
    assert len(app.dataframe) == 1
    assert app.dataframe[0].value.equals(profile)
    app.checkbox(key="retrieval_comparison_overlap").uncheck().run()
    assert len(app.dataframe) == 0
    assert not app.exception


def test_profile_repin_empty_current_and_clear_lifecycle():
    app = AppTest.from_string(APP, default_timeout=30).run()
    app.button(key="retrieval_comparison_pin").click().run()
    app.checkbox(key="retrieval_comparison_overlap").check().run()
    app.session_state["results"] = []
    app.run()
    assert list(app.dataframe[0].value["Current chunks"]) == [0, 0]
    assert list(app.dataframe[0].value["Overlap (%)"]) == [0.0, 0.0]
    app.session_state["results"] = [(9, "new", 0.5)]
    app.run()
    app.button(key="retrieval_comparison_pin").click().run()
    assert list(app.dataframe[0].value["Overlap (%)"]) == [100.0]
    app.button(key="retrieval_comparison_clear").click().run()
    assert len(app.checkbox) == 0
    app.button(key="retrieval_comparison_pin").click().run()
    assert not app.checkbox(key="retrieval_comparison_overlap").value
    assert not app.exception


def test_rank_overlap_download_uses_complete_runs_and_follows_repin_clear():
    with patch("streamlit.download_button") as download:
        app = AppTest.from_string(APP, default_timeout=30).run()
        app.button(key="retrieval_comparison_pin").click().run()
        app.checkbox(key="retrieval_comparison_overlap").check().run()
        app.session_state["results"] = [(1, "b", 0.1), (0, "a", 0.2), (2, "c", 0.3)]
        app.run()
        download.reset_mock()
        app.number_input(key="retrieval_comparison_min_chars").set_value(100).run()
        calls = [call.kwargs for call in download.call_args_list
                 if call.kwargs.get("key") == "retrieval_comparison_overlap_download"]
        assert len(calls) == 1
        assert calls[0]["mime"] == "application/json"
        assert calls[0]["file_name"] == "retrieval-rank-overlap.json"
        payload = json.loads(calls[0]["data"])
        assert payload["count"] == 3
        assert [row["overlap_percent"] for row in payload["profile"]] == [0.0, 100.0, 66.7]
        assert len(app.dataframe) == 1
        app.session_state["results"] = []
        download.reset_mock()
        app.run()
        calls = [call.kwargs for call in download.call_args_list
                 if call.kwargs.get("key") == "retrieval_comparison_overlap_download"]
        assert [row["current_chunks"] for row in json.loads(calls[0]["data"])["profile"]] == [0, 0]
        app.session_state["results"] = [(9, "new", 0.5)]
        app.run().button(key="retrieval_comparison_pin").click().run()
        assert list(app.dataframe[0].value["Overlap (%)"]) == [100.0]
        download.reset_mock()
        app.checkbox(key="retrieval_comparison_overlap").uncheck().run()
        assert not any(call.kwargs.get("key") == "retrieval_comparison_overlap_download"
                       for call in download.call_args_list)
        app.button(key="retrieval_comparison_clear").click().run()
        assert not app.checkbox
        assert not app.exception
