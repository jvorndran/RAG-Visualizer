"""Exercise the profile toggle and full-snapshot scope without provider calls."""

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
