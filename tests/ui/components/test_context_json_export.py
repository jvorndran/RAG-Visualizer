"""Visible JSON evidence follows the actual card selection without another retrieval."""

import json

from streamlit.testing.v1 import AppTest

from tests.ui.components.test_retrieved_context_search import APP


def test_visible_json_follows_order_search_ranks_and_executed_query():
    app = AppTest.from_string(APP).run(timeout=60)
    app.selectbox(key="retrieved_context_order").select("Longest text first").run()
    app.number_input(key="retrieved_context_first_rank").set_value(2).run()
    app.text_input(key="retrieved_context_search").input(" STRASSE ").run()
    assert not app.exception
    selected = json.loads(app.session_state["retrieval_downloads"][
        "retrieved_context_json_download"
    ])
    assert selected == {
        "schema_version": 1, "scope": "visible_context", "query": "original query",
        "filters": {"search": "strasse", "first_rank": 2, "last_rank": None,
                    "order": "Longest text first"},
        "count": 1, "chunks": [{"chunk_index": 12, "rank": 3,
                                  "score": 0.4, "original_score": 0.4,
                                  "text": "Straße outlook"}],
    }
    complete = json.loads(app.session_state["retrieval_download_data"])
    assert len(complete["current"]["chunks"]) == 3
    app.text_input(key="retrieved_context_search").input("").run()
    selected = json.loads(app.session_state["retrieval_downloads"][
        "retrieved_context_json_download"
    ])
    assert [row["rank"] for row in selected["chunks"]] == [2, 3]
    assert selected["chunks"][0]["original_score"] == 0.9
    app.text_input(key="retrieved_context_search").input("absent").run()
    assert "retrieved_context_json_download" not in app.session_state["retrieval_downloads"]
    app.number_input(key="retrieved_context_last_rank").set_value(1).run()
    assert "retrieved_context_json_download" not in app.session_state["retrieval_downloads"]
    assert not app.exception


def test_visible_json_preserves_duplicate_identities_raw_text_and_safe_scores():
    from types import SimpleNamespace

    from unravel.services.retrieval.export import serialize_visible_context

    raw = "  Café <tag>\n\n"
    results = [(5, SimpleNamespace(index=0, text=raw, score=float("nan"),
                                  metadata={"original_score": 0.0, "secret": "hidden"})),
               (8, SimpleNamespace(index=0, text=raw, score=0.25,
                                  metadata={"original_score": float("inf")}))]
    serialized = serialize_visible_context("executed", results, {"search": ""})
    payload = json.loads(serialized)
    assert serialized.endswith("\n")
    assert payload["count"] == 2
    assert [row["rank"] for row in payload["chunks"]] == [5, 8]
    assert [row["text"] for row in payload["chunks"]] == [raw, raw]
    assert payload["chunks"][0]["score"] is None
    assert payload["chunks"][0]["original_score"] == 0.0
    assert payload["chunks"][1]["original_score"] is None
    assert "hidden" not in serialized and "metadata" not in serialized
    assert len(results) == 2
    assert json.loads(serialize_visible_context("", [], {}))["chunks"] == []
