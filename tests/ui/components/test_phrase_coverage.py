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
