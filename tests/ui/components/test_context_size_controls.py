"""Context size comparison reads complete snapshots without provider requests."""

from streamlit.testing.v1 import AppTest

APP = """
from types import SimpleNamespace
import streamlit as st
from unravel.ui.components.retrieval_comparison import render_retrieval_comparison
from unravel.services.retrieval.comparison import snapshot_retrieval
texts = st.session_state.get("texts", ["é\\n", "🙂"])
current = snapshot_retrieval("query", [
    SimpleNamespace(index=i, text=text, score=0.5) for i, text in enumerate(texts)
], {})
render_retrieval_comparison(current)
"""


def test_profile_uses_complete_snapshots_and_tracks_empty_current_repin_clear():
    app = AppTest.from_string(APP, default_timeout=30).run()
    app.button(key="retrieval_comparison_pin").click().run()
    assert not app.checkbox(key="retrieval_comparison_context_size").value
    app.session_state["texts"] = ["abcdef"]
    app.run()
    app.checkbox(key="retrieval_comparison_context_size").check().run()
    assert list(app.dataframe[0].value["Baseline characters"]) == [2, 3]
    assert list(app.dataframe[0].value["Current characters"]) == [6, 6]
    assert list(app.dataframe[0].value["Character change"]) == [4, 3]
    app.text_input(key="retrieval_comparison_search").input("absent").run()
    assert len(app.dataframe) == 1
    assert list(app.dataframe[0].value["Baseline UTF-8 bytes"]) == [3, 7]
    app.session_state["texts"] = []
    app.run()
    assert list(app.dataframe[0].value["Current characters"]) == [0, 0]
    app.session_state["texts"] = ["x"]
    app.run()
    app.button(key="retrieval_comparison_pin").click().run()
    assert list(app.dataframe[0].value["Character change"]) == [0]
    app.button(key="retrieval_comparison_clear").click().run()
    assert len(app.checkbox) == 0
    app.button(key="retrieval_comparison_pin").click().run()
    assert not app.checkbox(key="retrieval_comparison_context_size").value
    assert not app.exception
