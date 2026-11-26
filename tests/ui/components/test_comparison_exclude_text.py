"""Literal exclusion in comparison inspection and its visible exports."""

import json
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

APP = """
from types import SimpleNamespace
import streamlit as st
from unravel.ui.components.retrieval_comparison import render_retrieval_comparison
from unravel.services.retrieval.comparison import snapshot_retrieval
records = st.session_state.get("results", [])
results = [SimpleNamespace(index=i, text=text, score=score) for i, text, score in records]
current = snapshot_retrieval("query", results, {"top_k": 5, "threshold": 0})
render_retrieval_comparison(current)
"""


def test_exclusion_composes_with_search_order_and_both_visible_exports():
    app = AppTest.from_string(APP, default_timeout=30)
    app.session_state["results"] = [
        (0, "keep Straße [old]", 0.8), (1, "keep [literal].*", 0.7), (2, "keep plain", 0.6),
    ]
    with patch("unravel.ui.components.retrieval_comparison.st.download_button") as download:
        app.run().button(key="retrieval_comparison_pin").click().run()
        baseline = app.session_state["retrieval_comparison_baseline"]
        app.session_state["results"] = [(2, "keep plain", 0.4), (3, "keep STRASSE new", 0.3)]
        app.run()
        app.text_input(key="retrieval_comparison_search").input(" KEEP ").run()
        app.text_input(key="retrieval_comparison_exclude").input("  STRASSE  ").run()
        assert not app.exception
        assert list(app.dataframe[0].value["Chunk"]) == [2, 1]
        assert list(app.dataframe[0].value["Change"]) == ["Retained", "Removed"]
        app.radio(key="retrieval_comparison_order").set_value("Baseline rank").run()
        assert list(app.dataframe[0].value["Chunk"]) == [1, 2]
        exports = {call.kwargs["key"]: call.kwargs["data"] for call in download.call_args_list}
        document = json.loads(exports["retrieval_comparison_json_download"])
        assert [chunk["chunk_index"] for chunk in document["chunks"]] == [1, 2]
        assert [chunk["text"] for chunk in document["chunks"]] == ["keep [literal].*", "keep plain"]
        assert document["chunks"][0]["baseline_rank"] == 2
        assert document["chunks"][1]["current_rank"] == 1
        assert document["filters"]["exclude_text"] == "strasse"
        assert document["filters"]["search"] == "keep"
        assert "Straße" not in exports["retrieval_comparison_text_download"]
        assert "STRASSE" not in exports["retrieval_comparison_text_download"]
        assert "keep [literal].*" in exports["retrieval_comparison_text_download"]
        app.text_input(key="retrieval_comparison_exclude").input("[literal].*").run()
        assert list(app.dataframe[0].value["Chunk"]) == [0, 2, 3]
        app.text_input(key="retrieval_comparison_exclude").input("   ").run()
        assert len(app.dataframe[0].value) == 4
        assert app.session_state["retrieval_comparison_baseline"] == baseline
        assert any("1 retained · 1 added · 2 removed" in item.value for item in app.caption)


def test_exclusion_handles_removed_only_empty_selection_and_baseline_reset():
    app = AppTest.from_string(APP, default_timeout=30)
    app.session_state["results"] = [(0, "hidden", 0.8), (1, "", 0.0)]
    with patch("unravel.ui.components.retrieval_comparison.st.download_button") as download:
        app.run().button(key="retrieval_comparison_pin").click().run()
        app.session_state["results"] = []
        app.run().text_input(key="retrieval_comparison_exclude").input("hidden").run()
        assert list(app.dataframe[0].value["Chunk"]) == [1]
        assert app.text[0].value == ""
        app.text_input(key="retrieval_comparison_search").input("hidden").run()
        download.reset_mock()
        app.run()
        assert not app.dataframe and not app.text and not app.exception
        download.assert_not_called()
        app.button(key="retrieval_comparison_clear").click().run()
        assert not app.text_input
        app.session_state["results"] = [(5, "fresh", 0.5)]
        app.run().button(key="retrieval_comparison_pin").click().run()
        assert app.text_input(key="retrieval_comparison_exclude").value == ""
        assert list(app.dataframe[0].value["Chunk"]) == [5]
        assert not app.exception
