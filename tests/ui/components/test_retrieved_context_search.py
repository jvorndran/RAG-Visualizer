"""Inspect actual query cards through offline Streamlit interactions."""

import json

from streamlit.testing.v1 import AppTest

APP = """
from types import SimpleNamespace
from unittest.mock import patch
import streamlit as st
from unravel.ui.steps.query import _render_retrieved_chunks

results = [
    SimpleNamespace(index=7, text="Revenue rose", score=0.8, metadata={}),
    SimpleNamespace(
        index=9, text="Operating RISK fell", score=0.6, metadata={"original_score": 0.9}
    ),
    SimpleNamespace(index=12, text="Straße outlook", score=0.4, metadata={}),
]
st.session_state.current_query = "original query"
st.session_state.last_search_results = results
st.session_state.rendered_texts = []
st.session_state.rendered_badges = []
st.session_state.retrieval_downloads = {}

def prepare(chunks, **kwargs):
    st.session_state.rendered_texts = [chunk.text for chunk in chunks]
    st.session_state.rendered_positions = [chunk.start_index for chunk in chunks]
    return []

def render(**kwargs):
    st.session_state.rendered_badges = kwargs["custom_badges"]

def download(**kwargs):
    st.session_state.retrieval_downloads[kwargs["key"]] = kwargs["data"]
    if kwargs["key"] == "retrieval_experiment_download":
        st.session_state.retrieval_download_data = kwargs["data"]

with patch("unravel.ui.steps.query.prepare_chunk_display_data", side_effect=prepare), \\
     patch("unravel.ui.steps.query.render_chunk_cards", side_effect=render), \\
     patch("unravel.ui.components.retrieval_export.st.download_button", side_effect=download):
    _render_retrieved_chunks(results, show_scores=st.session_state.get("show_scores", True))
"""


def test_rank_window_composes_with_search_preserves_ranks_and_complete_snapshots():
    app = AppTest.from_string(APP).run(timeout=60)
    app.number_input(key="retrieved_context_first_rank").set_value(2).run()
    app.number_input(key="retrieved_context_last_rank").set_value(2).run()
    assert not app.exception
    assert app.session_state["rendered_texts"] == ["Operating RISK fell"]
    assert app.session_state["rendered_positions"] == [1]
    assert app.session_state["rendered_badges"][0][0]["value"] == "2"
    assert len(app.session_state["last_search_results"]) == 3
    app.text_input(key="retrieved_context_search").input("revenue").run()
    assert app.session_state["rendered_texts"] == []
    app.text_input(key="retrieved_context_search").input("risk").run()
    app.button(key="retrieval_comparison_pin").click().run()
    assert len(app.session_state["retrieval_comparison_baseline"].chunks) == 3
    exported = json.loads(app.session_state["retrieval_download_data"])
    assert len(exported["current"]["chunks"]) == 3
    assert len(exported["baseline"]["chunks"]) == 3
    assert not app.exception
    assert list(app.dataframe[0].value["Change"]) == ["Retained"] * 3


def test_rank_window_recovers_from_reversed_or_empty_range_and_can_be_cleared():
    app = AppTest.from_string(APP).run(timeout=60)
    app.number_input(key="retrieved_context_first_rank").set_value(3).run()
    app.number_input(key="retrieved_context_last_rank").set_value(2).run()
    assert not app.exception
    assert any("First displayed rank" in error.value for error in app.error)
    assert app.session_state["rendered_texts"] == []
    app.number_input(key="retrieved_context_last_rank").set_value(None).run()
    assert not app.error
    assert app.session_state["rendered_texts"] == ["Straße outlook"]
    app.number_input(key="retrieved_context_first_rank").set_value(4).run()
    assert app.session_state["rendered_texts"] == []
    app.number_input(key="retrieved_context_first_rank").set_value(1).run()
    assert len(app.session_state["rendered_texts"]) == 3
    assert not app.exception


def test_search_preserves_original_rank_scores_context_and_baseline():
    app = AppTest.from_string(APP).run(timeout=60)
    assert not app.exception
    assert len(app.session_state["rendered_texts"]) == 3
    app.text_input(key="retrieved_context_search").input("  risk  ").run()
    assert not app.exception
    assert app.session_state["rendered_texts"] == ["Operating RISK fell"]
    assert app.session_state["rendered_positions"] == [1]
    badges = app.session_state["rendered_badges"][0]
    assert [(badge["label"], badge["value"]) for badge in badges] == [
        ("Rank", "2"), ("Score", "0.9000")
    ]
    assert any("Showing 1 of 3" in caption.value for caption in app.caption)
    assert len(app.session_state["last_search_results"]) == 3
    app.button(key="retrieval_comparison_pin").click().run()
    assert len(app.session_state["retrieval_comparison_baseline"].chunks) == 3
    assert list(app.dataframe[0].value["Change"]) == ["Retained"] * 3


def test_no_matches_clear_unicode_and_hidden_scores():
    app = AppTest.from_string(APP).run(timeout=60)
    app.text_input(key="retrieved_context_search").input("absent").run()
    assert not app.exception
    assert app.session_state["rendered_texts"] == []
    assert any("No retrieved chunks match" in info.value for info in app.info)
    app.session_state["show_scores"] = False
    app.text_input(key="retrieved_context_search").input("STRASSE").run()
    assert app.session_state["rendered_texts"] == ["Straße outlook"]
    badges = app.session_state["rendered_badges"][0]
    assert [(badge["label"], badge["value"]) for badge in badges] == [("Rank", "3")]
    app.text_input(key="retrieved_context_search").input("  ").run()
    assert len(app.session_state["rendered_texts"]) == 3
    assert not app.exception


REAL_CARDS_APP = """
from types import SimpleNamespace
import streamlit as st
from unravel.ui.steps.query import _render_retrieved_chunks

results = [
    SimpleNamespace(index=0, text="Risk", score=0.9, metadata={"heading_text": "Risk"}),
    SimpleNamespace(index=1, text="Risk disclosure", score=0.5, metadata={}),
]
_render_retrieved_chunks(results)
"""


def test_real_cards_include_header_only_context_with_matching_rank_and_score():
    app = AppTest.from_string(REAL_CARDS_APP).run(timeout=60)
    assert not app.exception
    cards = "\n".join(element.value for element in app.markdown)
    assert cards.count('<details class="chunk-details"') == 2
    assert "Rank: 1</span>" in cards and "Rank: 2</span>" in cards
    assert "0.9000" in cards and "0.5000" in cards
    app.text_input(key="retrieved_context_search").input("disclosure").run()
    cards = "\n".join(element.value for element in app.markdown)
    assert cards.count('<details class="chunk-details"') == 1
    assert "Rank: 2</span>" in cards and "Rank: 1</span>" not in cards
    assert "0.5000" in cards and "0.9000" not in cards
    assert not app.exception
