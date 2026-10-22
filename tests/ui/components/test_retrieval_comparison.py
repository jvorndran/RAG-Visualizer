"""Exercise the baseline controls without embeddings, provider calls, or an LLM."""

from streamlit.testing.v1 import AppTest

APP = """
from types import SimpleNamespace
import streamlit as st
from unravel.ui.components.retrieval_comparison import render_retrieval_comparison
from unravel.services.retrieval.comparison import snapshot_retrieval

records = st.session_state.get("results", [(0, "first", 0.8), (1, "second", 0.6)])
results = [SimpleNamespace(index=i, text=text, score=score) for i, text, score in records]
settings = {"top_k": 2, "threshold": 0.1, "retrieval_config": {"strategy": "DenseRetriever"}}
current = snapshot_retrieval(st.session_state.get("query", "original query"), results, settings)
render_retrieval_comparison(current)
"""


def test_pin_compare_clear_and_replace_baseline():
    app = AppTest.from_string(APP).run()
    assert not app.exception
    app.button(key="retrieval_comparison_pin").click().run()
    assert not app.exception
    assert len(app.dataframe) == 1
    app.session_state["results"] = [(1, "second", 0.7), (2, "third", 0.5)]
    app.session_state["query"] = "follow-up query"
    app.run()
    frame = app.dataframe[0].value
    assert list(frame["Change"]) == ["Retained", "Added", "Removed"]
    assert frame.iloc[0]["Rank change"] == 1
    assert app.session_state["retrieval_comparison_baseline"].query == "original query"
    app.button(key="retrieval_comparison_pin").click().run()
    assert app.session_state["retrieval_comparison_baseline"].query == "follow-up query"
    assert list(app.dataframe[0].value["Change"]) == ["Retained", "Retained"]
    app.button(key="retrieval_comparison_clear").click().run()
    assert len(app.dataframe) == 0
    assert not app.exception


def test_empty_results_keep_baseline_visible_and_cannot_replace_it():
    app = AppTest.from_string(APP).run()
    app.button(key="retrieval_comparison_pin").click().run()
    app.session_state["results"] = []
    app.run()
    assert not app.exception
    assert app.button(key="retrieval_comparison_pin").disabled
    assert list(app.dataframe[0].value["Change"]) == ["Removed", "Removed"]


def test_inspector_shows_full_text_for_added_and_removed_chunks_with_reused_index():
    app = AppTest.from_string(APP)
    original = "old context\n" + "x" * 200 + "\nBASELINE END"
    replacement = "new context\n" + "y" * 200 + "\nCURRENT END"
    app.session_state["results"] = [(0, original, 0.8)]
    app.run()
    app.button(key="retrieval_comparison_pin").click().run()
    app.session_state["results"] = [(0, replacement, 0.4)]
    app.run()
    selector = app.selectbox(key="retrieval_comparison_inspect")
    assert len(selector.options) == 2
    assert "Added" in selector.options[0]
    assert "Removed" in selector.options[1]
    assert app.text[0].value == replacement
    selector.select(1).run()
    assert app.text[0].value == original
    assert app.session_state["retrieval_comparison_baseline"].chunks[0].text == original
    assert list(app.dataframe[0].value["Change"]) == ["Added", "Removed"]
    assert not app.exception


def test_inspector_follows_original_ranks_and_handles_fewer_results_and_clear():
    app = AppTest.from_string(APP).run()
    app.button(key="retrieval_comparison_pin").click().run()
    app.selectbox(key="retrieval_comparison_inspect").select(1).run()
    assert app.text[0].value == "second"
    app.session_state["results"] = []
    app.run()
    assert "Removed" in app.selectbox(key="retrieval_comparison_inspect").options[0]
    app.selectbox(key="retrieval_comparison_inspect").select(1).run()
    assert app.text[0].value == "second"
    app.session_state["results"] = [(1, "second", 0.9)]
    app.run()
    app.button(key="retrieval_comparison_pin").click().run()
    assert len(app.selectbox(key="retrieval_comparison_inspect").options) == 1
    assert app.text[0].value == "second"
    app.button(key="retrieval_comparison_clear").click().run()
    assert len(app.selectbox) == 0
    assert len(app.text) == 0
    assert not app.exception


QUERY_APP = """
from types import SimpleNamespace
from unittest.mock import patch
import streamlit as st
from unravel.services.retrieval.comparison import snapshot_retrieval
from unravel.ui.steps.query import _render_retrieved_chunks

results = [SimpleNamespace(index=0, text="context", score=0.8, metadata={})]
st.session_state.current_query = "pending query"
st.session_state.last_query_settings = {"retrieval_config": {"strategy": "SparseRetriever"}}
if not st.session_state.get("legacy"):
    st.session_state.last_retrieval_snapshot = snapshot_retrieval(
        "executed query", results, {"retrieval_config": {"strategy": "DenseRetriever"}}
    )
with patch("unravel.ui.steps.query.prepare_chunk_display_data", return_value=[]), \
     patch("unravel.ui.steps.query.render_chunk_cards"):
    _render_retrieved_chunks(results)
"""


def test_query_view_pins_executed_run_without_relabeling_pending_settings():
    app = AppTest.from_string(QUERY_APP).run(timeout=45)
    assert not app.exception
    app.button(key="retrieval_comparison_pin").click().run()
    assert not app.exception
    baseline = app.session_state["retrieval_comparison_baseline"]
    assert baseline.query == "executed query"
    assert baseline.strategy == "DenseRetriever"


def test_query_view_supports_results_created_before_snapshot_tracking():
    app = AppTest.from_string(QUERY_APP)
    app.session_state["legacy"] = True
    app.run(timeout=45)
    assert not app.exception
    app.button(key="retrieval_comparison_pin").click().run()
    assert not app.exception
    baseline = app.session_state["retrieval_comparison_baseline"]
    assert baseline.query == "pending query"
    assert baseline.chunks[0].text == "context"


def test_query_view_offers_experiment_download_alongside_baseline_controls() -> None:
    app = AppTest.from_string(QUERY_APP).run(timeout=45)
    assert not app.exception
    assert app.get("download_button")[0].label == "Download retrieval experiment"
    app.button(key="retrieval_comparison_pin").click().run()
    assert not app.exception
    assert len(app.get("download_button")) == 1
