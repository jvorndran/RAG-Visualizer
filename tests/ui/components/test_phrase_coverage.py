"""Phrase coverage remains independent of the visible comparison filters."""
from streamlit.testing.v1 import AppTest

from tests.ui.components.test_comparison_exclude_text import APP


def test_phrase_coverage_uses_full_runs_and_handles_opt_in_and_baseline_lifecycle():
    app = AppTest.from_string(APP, default_timeout=30)
    app.session_state["results"] = [(0, "red red", 0.9), (1, "blue", 0.8)]
    app.run().button(key="retrieval_comparison_pin").click().run()
    assert not app.text_area
    baseline = app.session_state["retrieval_comparison_baseline"]
    app.session_state["results"] = [(1, "blue", 0.7), (2, "red", 0.6)]
    app.run().checkbox(key="retrieval_comparison_phrases").check().run()
    app.text_area(key="retrieval_comparison_phrase_text").input("red\nblue\ngreen").run()
    assert list(app.dataframe[0].value["Current chunks"]) == [1, 1, 0]
    app.text_input(key="retrieval_comparison_search").input("absent").run()
    assert len(app.dataframe) == 1
    assert list(app.dataframe[0].value["Baseline chunks"]) == [1, 1, 0]
    assert app.session_state["retrieval_comparison_baseline"] == baseline
    app.session_state["results"] = []
    app.run()
    assert list(app.dataframe[0].value["Chunk change"]) == [-1, -1, 0]
    app.checkbox(key="retrieval_comparison_phrases").uncheck().run()
    assert not app.text_area
    app.checkbox(key="retrieval_comparison_phrases").check().run()
    assert app.text_area[0].value == ""
    app.button(key="retrieval_comparison_clear").click().run()
    assert not app.text_area
    app.session_state["results"] = [(3, "fresh", 0.5)]
    app.run().button(key="retrieval_comparison_pin").click().run()
    assert not app.checkbox(key="retrieval_comparison_phrases").value
    assert not app.exception


def test_first_match_ranks_use_full_runs_and_refresh_on_empty_results_and_repin():
    app = AppTest.from_string(APP, default_timeout=30)
    app.session_state["results"] = [
        (90, "blue", 0.1), (5, "red red", 0.9), (6, "lost", 0.8),
    ]
    app.run().button(key="retrieval_comparison_pin").click().run()
    baseline = app.session_state["retrieval_comparison_baseline"]
    app.session_state["results"] = [(5, "red red", 0.2), (90, "blue", 0.1), (7, "new", 0.0)]
    app.run().checkbox(key="retrieval_comparison_phrases").check().run()
    app.text_area(key="retrieval_comparison_phrase_text").input("red\nblue\nlost\nnew").run()
    frame = app.dataframe[0].value
    assert list(frame["Baseline first rank"][:3]) == [2, 1, 3]
    assert frame["Baseline first rank"].isna().tolist() == [False, False, False, True]
    assert list(frame["Current first rank"][:2]) == [1, 2]
    assert frame["Current first rank"].isna().tolist() == [False, False, True, False]
    assert frame["Current first rank"].iloc[3] == 3
    assert list(frame["First rank change"][:2]) == [1, -1]
    assert frame["First rank change"].isna().tolist() == [False, False, True, True]
    app.text_input(key="retrieval_comparison_search").input("absent").run()
    app.number_input(key="retrieval_comparison_top_ranks").set_value(1).run()
    app.radio(key="retrieval_comparison_order").set_value("Baseline rank").run()
    assert len(app.dataframe) == 1
    assert app.dataframe[0].value.equals(frame)
    assert app.session_state["retrieval_comparison_baseline"] == baseline
    app.session_state["results"] = []
    app.run()
    empty_frame = app.dataframe[0].value
    assert empty_frame["Current first rank"].isna().all()
    assert empty_frame["First rank change"].isna().all()
    assert list(empty_frame["Baseline first rank"][:3]) == [2, 1, 3]
    app.session_state["results"] = [(8, "red", 0.0)]
    app.run().button(key="retrieval_comparison_pin").click().run()
    repinned = app.dataframe[0].value
    assert repinned["Baseline first rank"].iloc[0] == 1
    assert repinned["Current first rank"].iloc[0] == 1
    assert repinned["First rank change"].iloc[0] == 0
    assert repinned["Baseline first rank"].isna().tolist() == [False, True, True, True]
    assert any("Positive first rank changes" in caption.value for caption in app.caption)
    assert not app.exception
