"""Exercise executed retrieval evidence through the real Query step offline."""

import json

import pytest
from streamlit.testing.v1 import AppTest

from unravel.services.retrieval.export import serialize_retrieval_experiment

APP = """
from types import SimpleNamespace
from unittest.mock import patch
import streamlit as st
from unravel.services.vector_store import SearchResult
from unravel.ui.steps.query import render_query_step

st.session_state.last_embeddings_result = {"vector_store": SimpleNamespace(size=10)}
st.session_state.llm_provider = "OpenAI"
st.session_state.llm_model = "offline-model"
st.session_state.retrieval_calls = []


def retrieve(**kwargs):
    st.session_state.retrieval_calls.append(kwargs)
    if st.session_state.get("fail_retrieval"):
        raise RuntimeError("offline retrieval failure")
    return [SearchResult(index=7, text="Full café context", score=0.8, metadata={})]


def preprocess(*args, **kwargs):
    if st.session_state.get("fail_preprocess"):
        raise RuntimeError("offline preprocessing failure")
    return {"bm25_index": "offline index"}


def backend(name):
    if not st.session_state.get("reranking_succeeds"):
        raise ImportError("offline reranker unavailable")
    return SimpleNamespace(rerank=lambda *args: [{"id": 0, "score": 0.95}])


variations = ["alternate", "question"]
model = SimpleNamespace(stream=lambda *args: iter(["answer"]))

with patch("unravel.ui.steps.query.get_embedder", return_value=object()), \
     patch("unravel.services.llm.get_api_key_from_env", return_value="offline-key"), \
     patch("unravel.services.llm.validate_config", return_value=(True, "")), \
     patch("unravel.services.llm.rewrite_query_variations", return_value=variations), \
     patch("unravel.services.retrieval.retrieve", side_effect=retrieve), \
     patch("unravel.services.retrieval.preprocess_retriever", side_effect=preprocess), \
     patch("unravel.services.retrieval.reranking_backends.get_backend", side_effect=backend), \
     patch("unravel.ui.steps.query.render_api_endpoint_expander"), \
     patch("unravel.ui.steps.query._save_query_settings"), \
     patch("unravel.ui.steps.query.get_model", return_value=model):
    render_query_step()
"""


def query_app(strategy: str = "HybridRetriever", fusion: str = "weighted_sum") -> AppTest:
    app = AppTest.from_string(APP)
    app.session_state["retrieval_config"] = {
        "strategy": strategy,
        "params": {"fusion_method": fusion, "dense_weight": 0.25},
    }
    app.session_state["reranking_config"] = {"enabled": True}
    app.run(timeout=60)
    assert not app.exception
    app.text_input(key="query_input").input(" question ")
    app.toggle(key="query_expansion_enabled").set_value(True)
    for slider in app.slider:
        if slider.label == "Minimum similarity score":
            slider.set_value(0.0)
    return app


@pytest.mark.parametrize("fusion", ["weighted_sum", "rrf"])
def test_snapshot_matches_retrieval_calls_and_keeps_pinned_run(fusion: str) -> None:
    app = query_app(fusion=fusion)
    app.button(key="ask_button").click().run(timeout=60)
    assert not app.exception
    current = app.session_state["last_retrieval_snapshot"]
    calls = app.session_state["retrieval_calls"]
    assert current.execution.queries_searched == tuple(call["query"] for call in calls)
    assert current.execution.queries_searched == ("question", "alternate")
    assert current.execution.query_variations == ("alternate", "question")
    assert all(call["retriever_name"] == current.strategy for call in calls)
    assert all(call["k"] == current.top_k for call in calls)
    assert all(call["fusion_method"] == current.execution.fusion_method for call in calls)
    if fusion == "weighted_sum":
        assert all(call["dense_weight"] == current.execution.dense_weight for call in calls)
    else:
        assert current.execution.fusion_rrf_k == 60
    assert current.execution.query_rrf_k == 60
    assert current.reranker is None
    assert current.execution.reranker_top_n is None
    assert current.chunks[0].original_score == 0.8
    assert current.chunks[0].score == pytest.approx(2 / 61)
    app.button(key="retrieval_comparison_pin").click().run(timeout=60)
    before = serialize_retrieval_experiment(app.session_state["retrieval_comparison_baseline"])
    app.session_state["retrieval_config"] = {"strategy": "DenseRetriever", "params": {}}
    app.run(timeout=60)
    assert not app.exception
    assert app.session_state["retrieval_calls"] == []
    assert app.session_state["last_retrieval_snapshot"] == current
    for slider in app.slider:
        if slider.label == "Minimum similarity score":
            slider.set_value(0.0)
    app.text_input(key="query_input").input("next question")
    app.button(key="ask_button").click().run(timeout=60)
    assert not app.exception
    baseline = app.session_state["retrieval_comparison_baseline"]
    assert serialize_retrieval_experiment(baseline) == before
    payload = json.loads(
        serialize_retrieval_experiment(app.session_state["last_retrieval_snapshot"], baseline)
    )
    assert payload["baseline"]["strategy"] == "HybridRetriever"
    assert payload["current"]["strategy"] == "DenseRetriever"
    assert payload["comparison"][0]["Change"] == "Retained"


def test_snapshot_records_dense_fallback_and_only_attempted_queries() -> None:
    app = query_app()
    app.session_state["fail_preprocess"] = True
    app.session_state["fail_retrieval"] = True
    app.button(key="ask_button").click().run(timeout=60)
    assert not app.exception
    current = app.session_state["last_retrieval_snapshot"]
    assert current.strategy == "DenseRetriever"
    assert current.execution.fusion_method is None
    assert current.execution.queries_searched == tuple(
        call["query"] for call in app.session_state["retrieval_calls"]
    )
    assert current.chunks == ()
    assert current.reranker is None


def test_snapshot_records_successful_reranking_settings() -> None:
    app = query_app()
    app.session_state["reranking_succeeds"] = True
    app.session_state["reranking_config"] = {
        "enabled": True, "model": "ms-marco-TinyBERT-L-2-v2", "top_n": 1
    }
    app.button(key="ask_button").click().run(timeout=60)
    assert not app.exception
    current = app.session_state["last_retrieval_snapshot"]
    assert current.reranker == "ms-marco-TinyBERT-L-2-v2"
    assert current.execution.reranker_top_n == 1
    assert current.chunks[0].score == 0.95
    assert current.chunks[0].original_score == 0.8
