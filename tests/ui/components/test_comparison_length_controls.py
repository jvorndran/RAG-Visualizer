"""Compare raw character bounds without changing captured retrieval evidence."""

import json
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

APP = """
from types import SimpleNamespace
import streamlit as st
from unravel.ui.components.retrieval_comparison import render_retrieval_comparison
from unravel.services.retrieval.comparison import snapshot_retrieval
records = st.session_state.get("results", [(0, "", 0.9), (1, "é\\n🙂", 0.8), (2, "longer", 0.7)])
current = snapshot_retrieval("query", [
    SimpleNamespace(index=i, text=text, score=score) for i, text, score in records
], {})
render_retrieval_comparison(current)
"""


def test_length_bounds_include_raw_unicode_and_removed_text_in_both_exports():
    with patch("streamlit.download_button") as download:
        app = AppTest.from_string(APP, default_timeout=30).run()
        app.button(key="retrieval_comparison_pin").click().run()
        baseline = app.session_state["retrieval_comparison_baseline"]
        app.session_state["results"] = [(2, "longer", 0.2), (9, "new", 0.1)]
        app.run()
        app.number_input(key="retrieval_comparison_min_chars").set_value(3).run()
        download.reset_mock()
        app.number_input(key="retrieval_comparison_max_chars").set_value(3).run()
        assert list(app.dataframe[0].value["Chunk"]) == [9, 1]
        assert list(app.dataframe[0].value["Change"]) == ["Added", "Removed"]
        assert app.text[0].value == "new"
        exports = {call.kwargs["key"]: call.kwargs["data"] for call in download.call_args_list}
        payload = json.loads(exports["retrieval_comparison_json_download"])
        assert payload["filters"]["min_characters"] == 3
        assert payload["filters"]["max_characters"] == 3
        assert [row["text"] for row in payload["chunks"]] == ["new", "é\n🙂"]
        assert [row["chunk_index"] for row in payload["chunks"]] == [9, 1]
        assert "é\n🙂" in exports["retrieval_comparison_text_download"]
        assert "longer" not in exports["retrieval_comparison_text_download"]
        app.selectbox(key="retrieval_comparison_change").select("Removed").run()
        assert app.text[0].value == "é\n🙂"
        app.text_input(key="retrieval_comparison_search").input("🙂").run()
        assert list(app.dataframe[0].value["Chunk"]) == [1]
        assert app.session_state["retrieval_comparison_baseline"] == baseline
        assert not app.exception


def test_length_range_empty_results_recover_and_reset_after_clear():
    app = AppTest.from_string(APP, default_timeout=30).run()
    app.button(key="retrieval_comparison_pin").click().run()
    assert list(app.dataframe[0].value["Chunk"]) == [0, 1, 2]
    app.number_input(key="retrieval_comparison_max_chars").set_value(2).run()
    assert list(app.dataframe[0].value["Chunk"]) == [0]
    app.number_input(key="retrieval_comparison_min_chars").set_value(3).run()
    assert not app.dataframe
    assert any("Maximum characters" in item.value for item in app.warning)
    app.number_input(key="retrieval_comparison_max_chars").set_value(0).run()
    assert not app.warning
    assert list(app.dataframe[0].value["Chunk"]) == [1, 2]
    app.session_state["results"] = []
    app.run()
    assert list(app.dataframe[0].value["Change"]) == ["Removed", "Removed"]
    app.button(key="retrieval_comparison_clear").click().run()
    assert not app.number_input
    app.session_state["results"] = [(5, "x", 0.2)]
    app.run().button(key="retrieval_comparison_pin").click().run()
    assert app.number_input(key="retrieval_comparison_min_chars").value == 0
    assert app.number_input(key="retrieval_comparison_max_chars").value == 0
    assert list(app.dataframe[0].value["Chunk"]) == [5]
    assert not app.exception
