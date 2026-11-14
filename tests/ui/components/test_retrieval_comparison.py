"""Exercise the baseline controls without embeddings, provider calls, or an LLM."""

from unittest.mock import patch

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
    app = AppTest.from_string(APP, default_timeout=30).run()
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


def test_minimum_rank_movement_filters_both_directions_and_visible_download():
    app = AppTest.from_string(APP, default_timeout=30)
    app.session_state["results"] = [
        (0, "down", 0.9), (1, "same", 0.8), (2, "small down", 0.7),
        (3, "UP Café", 0.6), (4, "removed", 0.5),
    ]
    with patch("unravel.ui.components.retrieval_comparison.st.download_button") as download:
        app.run().button(key="retrieval_comparison_pin").click().run()
        baseline = app.session_state["retrieval_comparison_baseline"]
        app.session_state["results"] = [
            (3, "UP Café", 0.2), (1, "same", 0.4), (5, "added", 0.9),
            (2, "small down", 0.3), (0, "down", 0.8),
        ]
        app.run()
        assert len(app.dataframe[0].value) == 6
        app.number_input(key="retrieval_comparison_min_movement").set_value(3).run()
        assert not app.exception
        frame = app.dataframe[0].value
        assert list(frame["Chunk"]) == [3, 0]
        assert list(frame["Rank change"]) == [3, -4]
        assert list(frame["Baseline rank"]) == [4, 1]
        assert list(frame["Current rank"]) == [1, 5]
        assert list(frame["Current score"]) == [0.2, 0.8]
        assert app.text[0].value == "UP Café"
        document = download.call_args.kwargs["data"]
        assert "Chunk index 3" in document and "Chunk index 0" in document
        assert "small down" not in document and "removed" not in document
        assert any("4 retained · 1 added · 1 removed" in item.value for item in app.caption)
        assert app.session_state["retrieval_comparison_baseline"] == baseline
        app.radio(key="retrieval_comparison_order").set_value("Largest rank movement").run()
        assert list(app.dataframe[0].value["Chunk"]) == [0, 3]
        assert download.call_args.kwargs["data"].index("Chunk index 0") < (
            download.call_args.kwargs["data"].index("Chunk index 3")
        )
        app.text_input(key="retrieval_comparison_search").input("CAFÉ").run()
        assert list(app.dataframe[0].value["Chunk"]) == [3]
        app.selectbox(key="retrieval_comparison_change").select("Moved down").run()
        assert not app.dataframe
        app.text_input(key="retrieval_comparison_search").input("").run()
        assert list(app.dataframe[0].value["Chunk"]) == [0]
        app.number_input(key="retrieval_comparison_min_movement").set_value(5).run()
        assert not app.dataframe
        assert not app.text
        download.reset_mock()
        app.run()
        download.assert_not_called()
        app.selectbox(key="retrieval_comparison_change").select("All changes").run()
        app.number_input(key="retrieval_comparison_min_movement").set_value(0).run()
        assert len(app.dataframe[0].value) == 6
        assert app.session_state["retrieval_comparison_baseline"] == baseline


def test_minimum_rank_movement_handles_removed_only_runs_and_resets_with_baseline():
    app = AppTest.from_string(APP, default_timeout=30).run()
    app.button(key="retrieval_comparison_pin").click().run()
    app.number_input(key="retrieval_comparison_min_movement").set_value(1).run()
    assert not app.dataframe  # Identical ranks do not meet a positive threshold.
    app.session_state["results"] = []
    app.run()
    assert not app.exception
    assert not app.dataframe
    app.number_input(key="retrieval_comparison_min_movement").set_value(0).run()
    assert list(app.dataframe[0].value["Change"]) == ["Removed", "Removed"]
    app.number_input(key="retrieval_comparison_min_movement").set_value(7).run()
    app.button(key="retrieval_comparison_clear").click().run()
    assert not app.number_input
    app.session_state["results"] = [(9, "new", 0.5)]
    app.run().button(key="retrieval_comparison_pin").click().run()
    assert app.number_input(key="retrieval_comparison_min_movement").value == 0
    assert list(app.dataframe[0].value["Chunk"]) == [9]
    assert not app.exception


def test_empty_results_keep_baseline_visible_and_cannot_replace_it():
    app = AppTest.from_string(APP).run()
    app.button(key="retrieval_comparison_pin").click().run()
    app.session_state["results"] = []
    app.run()
    assert not app.exception
    assert app.button(key="retrieval_comparison_pin").disabled
    assert list(app.dataframe[0].value["Change"]) == ["Removed", "Removed"]


def test_filter_changes_keeps_original_ranks_and_full_text_inspection():
    app = AppTest.from_string(APP).run()
    app.button(key="retrieval_comparison_pin").click().run()
    app.session_state["results"] = [(1, "second", 0.7), (2, "third", 0.5)]
    app.run()
    app.selectbox(key="retrieval_comparison_change").select("Removed").run()
    assert not app.exception
    frame = app.dataframe[0].value
    assert list(frame["Change"]) == ["Removed"]
    assert frame.iloc[0]["Baseline rank"] == 1
    assert app.text[0].value == "first"
    assert any("1 retained · 1 added · 1 removed" in item.value for item in app.caption)
    app.selectbox(key="retrieval_comparison_change").select("Retained").run()
    assert app.dataframe[0].value.iloc[0]["Rank change"] == 1
    assert app.text[0].value == "second"
    app.selectbox(key="retrieval_comparison_change").select("All changes").run()
    assert list(app.dataframe[0].value["Change"]) == ["Retained", "Added", "Removed"]


def test_search_uses_full_baseline_text_and_literal_unicode_matching():
    app = AppTest.from_string(APP)
    original = "baseline " + "x" * 200 + " Straße [needle]"
    app.session_state["results"] = [(0, original, 0.8)]
    app.run()
    app.button(key="retrieval_comparison_pin").click().run()
    app.session_state["results"] = [(0, "replacement [other]", 0.4)]
    app.run()
    app.text_input(key="retrieval_comparison_search").input("  STRASSE [NEEDLE]  ").run()
    assert not app.exception
    assert list(app.dataframe[0].value["Change"]) == ["Removed"]
    assert app.text[0].value == original
    assert "needle" not in app.dataframe[0].value.iloc[0]["Excerpt"]
    app.selectbox(key="retrieval_comparison_change").select("Added").run()
    assert len(app.text) == 0
    assert len(app.selectbox) == 1
    assert any("No comparison chunks match" in item.value for item in app.info)
    app.text_input(key="retrieval_comparison_search").input("[other]").run()
    assert list(app.dataframe[0].value["Change"]) == ["Added"]
    assert app.text[0].value == "replacement [other]"
    app.text_input(key="retrieval_comparison_search").input(".*").run()
    assert len(app.text) == 0
    assert app.session_state["retrieval_comparison_baseline"].chunks[0].text == original


def test_filters_recover_after_empty_results_and_clear_with_baseline():
    app = AppTest.from_string(APP).run()
    app.button(key="retrieval_comparison_pin").click().run()
    app.selectbox(key="retrieval_comparison_inspect").select(1).run()
    app.selectbox(key="retrieval_comparison_change").select("Added").run()
    assert not app.exception
    assert len(app.text) == 0
    app.session_state["results"] = [(9, "new", 0.5)]
    app.run()
    assert not app.exception
    assert app.text[0].value == "new"
    app.button(key="retrieval_comparison_clear").click().run()
    assert len(app.selectbox) == 0
    assert len(app.text_input) == 0
    app.button(key="retrieval_comparison_pin").click().run()
    assert not app.exception
    assert app.selectbox(key="retrieval_comparison_change").value == "All changes"
    assert app.text[0].value == "new"


def test_rank_movement_filters_preserve_ranks_scores_and_full_text():
    app = AppTest.from_string(APP)
    app.session_state["results"] = [
        (0, "down", 0.9), (1, "up", 0.8), (2, "same", 0.7), (3, "removed", 0.6)
    ]
    app.run()
    app.button(key="retrieval_comparison_pin").click().run()
    app.session_state["results"] = [
        (1, "up", 0.1), (0, "down", 0.95), (2, "same", 0.2), (4, "added", 0.5)
    ]
    app.run()
    for choice, index, before, after, text in [
        ("Moved up", 1, 2, 1, "up"),
        ("Moved down", 0, 1, 2, "down"),
        ("Same rank", 2, 3, 3, "same"),
    ]:
        app.selectbox(key="retrieval_comparison_change").select(choice).run()
        assert not app.exception
        frame = app.dataframe[0].value
        assert list(frame["Chunk"]) == [index]
        assert list(frame["Change"]) == ["Retained"]
        assert frame.iloc[0]["Baseline rank"] == before
        assert frame.iloc[0]["Current rank"] == after
        assert frame.iloc[0]["Rank change"] == before - after
        assert app.text[0].value == text
        assert any("3 retained · 1 added · 1 removed" in item.value for item in app.caption)
        assert any("Showing 1 of 5" in item.value for item in app.caption)
    app.selectbox(key="retrieval_comparison_change").select("Moved up").run()
    assert app.dataframe[0].value.iloc[0]["Baseline score"] == 0.8
    assert app.dataframe[0].value.iloc[0]["Current score"] == 0.1
    assert len(app.session_state["retrieval_comparison_baseline"].chunks) == 4


def test_rank_movement_filters_compose_with_literal_full_text_search():
    original = "context " + "x" * 200 + " Straße [needle]"
    app = AppTest.from_string(APP)
    app.session_state["results"] = [(0, "first", 0.8), (1, original, 0.7)]
    app.run()
    app.button(key="retrieval_comparison_pin").click().run()
    app.session_state["results"] = [(1, original, 0.2), (0, "first", 0.9)]
    app.run()
    app.selectbox(key="retrieval_comparison_change").select("Moved up").run()
    app.text_input(key="retrieval_comparison_search").input(" STRASSE [NEEDLE] ").run()
    assert not app.exception
    assert list(app.dataframe[0].value["Chunk"]) == [1]
    assert app.text[0].value == original
    app.text_input(key="retrieval_comparison_search").input(".*").run()
    assert len(app.dataframe) == 0
    assert any("No comparison chunks match" in item.value for item in app.info)
    app.text_input(key="retrieval_comparison_search").input("").run()
    assert app.text[0].value == original


def test_rank_filters_exclude_unmatched_chunks_and_recover_after_repin_and_clear():
    app = AppTest.from_string(APP).run()
    app.button(key="retrieval_comparison_pin").click().run()
    app.session_state["results"] = [(0, "changed text", 0.5)]
    app.run()
    for choice in ["Moved up", "Moved down", "Same rank"]:
        app.selectbox(key="retrieval_comparison_change").select(choice).run()
        assert not app.exception
        assert len(app.dataframe) == 0
    app.session_state["results"] = []
    app.run()
    assert not app.exception
    assert len(app.dataframe) == 0
    app.selectbox(key="retrieval_comparison_change").select("All changes").run()
    assert list(app.dataframe[0].value["Change"]) == ["Removed", "Removed"]
    app.session_state["results"] = [(1, "second", 0.5)]
    app.run()
    app.selectbox(key="retrieval_comparison_change").select("Moved up").run()
    assert app.text[0].value == "second"
    app.button(key="retrieval_comparison_pin").click().run()
    assert not app.exception
    assert len(app.dataframe) == 0
    app.selectbox(key="retrieval_comparison_change").select("Same rank").run()
    assert app.text[0].value == "second"
    app.button(key="retrieval_comparison_clear").click().run()
    app.button(key="retrieval_comparison_pin").click().run()
    assert not app.exception
    assert app.selectbox(key="retrieval_comparison_change").value == "All changes"
    assert app.text[0].value == "second"


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


def test_comparison_order_preserves_ranks_ties_and_full_text_inspection():
    app = AppTest.from_string(APP)
    app.session_state["results"] = [
        (0, "first", 0.9), (1, "second", 0.8), (2, "third", 0.7), (3, "fourth", 0.6),
    ]
    app.run().button(key="retrieval_comparison_pin").click().run()
    app.session_state["results"] = [
        (2, "third", 0.8), (1, "second", 0.7), (0, "first", 0.6), (4, "added", 0.5),
    ]
    app.run()
    original = app.dataframe[0].value.copy()
    baseline = app.session_state["retrieval_comparison_baseline"]
    app.radio(key="retrieval_comparison_order").set_value("Baseline rank").run()
    frame = app.dataframe[0].value
    assert list(frame["Chunk"]) == [0, 1, 2, 3, 4]
    assert list(frame["Change"]) == ["Retained", "Retained", "Retained", "Removed", "Added"]
    assert frame.iloc[0]["Current rank"] == 3
    assert app.text[0].value == "first"
    app.radio(key="retrieval_comparison_order").set_value("Largest rank movement").run()
    frame = app.dataframe[0].value
    assert list(frame["Chunk"]) == [2, 0, 1, 4, 3]
    assert list(frame["Rank change"][:3]) == [2, -2, 0]
    app.selectbox(key="retrieval_comparison_inspect").select(1).run()
    assert app.text[0].value == "first"
    assert app.session_state["retrieval_comparison_baseline"] == baseline
    app.radio(key="retrieval_comparison_order").set_value("Current retrieval order").run()
    assert app.dataframe[0].value.equals(original)
    assert not app.exception


def test_comparison_length_order_uses_full_text_and_preserves_ties_and_ranks():
    removed = "café " + "x" * 160 + "\nlast line"
    app = AppTest.from_string(APP, default_timeout=30)
    app.session_state["results"] = [(1, "café", 0.8), (2, removed, 0.7)]
    with patch("unravel.ui.components.retrieval_comparison.st.download_button") as download:
        app.run().button(key="retrieval_comparison_pin").click().run()
        app.session_state["results"] = [(3, "éabc", 0.9), (1, "café", 0.6), (4, "a ", 0.5)]
        app.run()
        app.radio(key="retrieval_comparison_order").set_value("Shortest text first").run()
        assert list(app.dataframe[0].value["Chunk"]) == [4, 3, 1, 2]
        app.radio(key="retrieval_comparison_order").set_value("Longest text first").run()
        frame = app.dataframe[0].value
        assert list(frame["Chunk"]) == [2, 3, 1, 4]
        assert list(frame["Current rank"].dropna()) == [1, 2, 3]
        assert frame.iloc[0]["Baseline rank"] == 2
        assert app.text[0].value == removed
        document = download.call_args.kwargs["data"]
        assert removed in document
        assert document.index("Chunk index 2") < document.index("Chunk index 3")
        app.text_input(key="retrieval_comparison_search").input("CAFÉ").run()
        assert list(app.dataframe[0].value["Chunk"]) == [2, 1]
        assert not app.exception


def test_comparison_order_composes_with_filters_and_resets_after_clear():
    app = AppTest.from_string(APP).run()
    app.button(key="retrieval_comparison_pin").click().run()
    app.session_state["results"] = [(1, "second", 0.7), (9, "new", 0.5)]
    app.run()
    app.radio(key="retrieval_comparison_order").set_value("Baseline rank").run()
    app.selectbox(key="retrieval_comparison_change").select("Removed").run()
    assert app.text[0].value == "first"
    app.text_input(key="retrieval_comparison_search").input("missing").run()
    assert len(app.dataframe) == 0
    app.text_input(key="retrieval_comparison_search").input("").run()
    assert app.text[0].value == "first"
    app.button(key="retrieval_comparison_clear").click().run()
    assert len(app.radio) == 0
    app.button(key="retrieval_comparison_pin").click().run()
    assert app.radio(key="retrieval_comparison_order").value == "Current retrieval order"
    assert not app.exception


def test_visible_comparison_text_download_tracks_filters_order_and_removed_text():
    original = "  Straße\n\t" + "x" * 200 + " [needle]"
    app = AppTest.from_string(APP)
    app.session_state["results"] = [(0, original, 0.8), (1, "second", 0.6)]
    with patch("unravel.ui.components.retrieval_comparison.st.download_button") as download:
        app.run(timeout=30)
        download.assert_not_called()
        app.button(key="retrieval_comparison_pin").click().run()
        app.session_state["results"] = [(1, "second", 0.7), (9, "new", 0.5)]
        app.run()
        app.radio(key="retrieval_comparison_order").set_value("Baseline rank").run()
        payload = download.call_args.kwargs
        assert payload["file_name"] == "retrieval-comparison.txt"
        assert payload["mime"] == "text/plain; charset=utf-8"
        document = payload["data"]
        assert document.index("Chunk index 0") < document.index("Chunk index 1")
        assert document.index("Chunk index 1") < document.index("Chunk index 9")
        assert (
            "Chunk index 0 | Removed\nBaseline rank 1 | Current rank —\n\n" + original
        ) in document
        assert "Chunk index 1 | Retained\nBaseline rank 2 | Current rank 1" in document
        app.text_input(key="retrieval_comparison_search").input("[needle]").run()
        document = download.call_args.kwargs["data"]
        assert original in document
        assert "Chunk index 1" not in document
        assert "Chunk index 9" not in document
        calls = download.call_count
        app.selectbox(key="retrieval_comparison_change").select("Added").run()
        assert download.call_count == calls
        app.text_input(key="retrieval_comparison_search").input("").run()
        assert "Chunk index 9 | Added" in download.call_args.kwargs["data"]
        assert app.session_state["retrieval_comparison_baseline"].chunks[0].text == original
        app.button(key="retrieval_comparison_clear").click().run()
        calls = download.call_count
        app.run()
        assert download.call_count == calls
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
    expected_downloads = ["Download retrieval experiment", "Download visible context as text"]
    assert [button.label for button in app.get("download_button")] == expected_downloads
    app.button(key="retrieval_comparison_pin").click().run()
    assert not app.exception
    assert [button.label for button in app.get("download_button")] == [
        "Download visible comparison as JSON", "Download visible comparison as text",
        *expected_downloads,
    ]


def test_visible_comparison_json_preserves_filtered_order_full_text_and_scores():
    import json

    with patch("streamlit.download_button") as download:
        app = AppTest.from_string(APP, default_timeout=30).run()
        app.session_state["results"] = [(0, "  café\nraw  ", 0.8), (1, "removed", 0.0)]
        app.run().button(key="retrieval_comparison_pin").click().run()
        baseline = app.session_state["retrieval_comparison_baseline"]
        app.session_state["results"] = [(2, "added", 0.9), (0, "  café\nraw  ", 0.7)]
        app.run()
        app.radio(key="retrieval_comparison_order").set_value("Baseline rank").run()

        def exported():
            calls = [c.kwargs for c in download.call_args_list
                     if c.kwargs.get("key") == "retrieval_comparison_json_download"]
            assert calls[-1]["mime"] == "application/json"
            return json.loads(calls[-1]["data"])

        document = exported()
        assert document["schema_version"] == 1
        assert document["scope"] == "visible_comparison"
        assert document["count"] == 3
        assert [c["chunk_index"] for c in document["chunks"]] == [0, 1, 2]
        assert document["chunks"][0] == {
            "chunk_index": 0, "change": "Retained", "baseline_rank": 1,
            "current_rank": 2, "rank_change": -1, "baseline_score": 0.8,
            "current_score": 0.7, "text": "  café\nraw  ",
        }
        assert document["chunks"][1]["current_score"] is None
        assert document["chunks"][1]["baseline_score"] == 0.0
        app.text_input(key="retrieval_comparison_search").input("CAFÉ").run()
        document = exported()
        assert document["count"] == 1
        assert document["filters"]["search"] == "café"
        assert document["filters"]["order"] == "Baseline rank"
        assert document["baseline_query"] == "original query"
        assert document["current_query"] == "original query"
        assert len(app.session_state["retrieval_comparison_baseline"].chunks) == 2
        assert app.session_state["retrieval_comparison_baseline"] == baseline
        app.selectbox(key="retrieval_comparison_change").select("Removed").run()
        download.reset_mock()
        app.run()
        download.assert_not_called()
        assert not app.exception


def test_visible_comparison_json_exports_removed_only_and_nonfinite_scores():
    import json

    with patch("streamlit.download_button") as download:
        app = AppTest.from_string(APP, default_timeout=30).run()
        app.session_state["results"] = [(7, "removed\nΩ", float("inf"))]
        app.run().button(key="retrieval_comparison_pin").click().run()
        app.session_state["results"] = []
        app.run()
        calls = [c.kwargs for c in download.call_args_list
                 if c.kwargs.get("key") == "retrieval_comparison_json_download"]
        assert calls
        document = json.loads(calls[-1]["data"])
        assert document["count"] == 1
        assert document["chunks"][0]["text"] == "removed\nΩ"
        assert document["chunks"][0]["baseline_score"] is None
        assert document["chunks"][0]["current_rank"] is None
        assert not app.exception
