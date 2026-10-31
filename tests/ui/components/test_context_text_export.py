"""Visible-context text downloads follow the actual card selection."""

import json

from streamlit.testing.v1 import AppTest

from tests.ui.components.test_retrieved_context_search import APP


def test_download_follows_search_and_rank_window_without_changing_complete_export():
    app = AppTest.from_string(APP).run(timeout=60)
    assert not app.exception
    text = app.session_state["retrieval_downloads"]["retrieved_context_text_download"]
    assert "Rank 1 | Chunk index 7\nRevenue rose" in text
    assert "Rank 3 | Chunk index 12\nStraße outlook" in text
    app.number_input(key="retrieved_context_first_rank").set_value(2).run()
    app.number_input(key="retrieved_context_last_rank").set_value(2).run()
    app.text_input(key="retrieved_context_search").input(" risk ").run()
    assert not app.exception
    assert app.session_state["retrieval_downloads"]["retrieved_context_text_download"] == (
        "Rank 2 | Chunk index 9\nOperating RISK fell\n"
    )
    assert app.session_state["rendered_texts"] == ["Operating RISK fell"]
    complete = json.loads(app.session_state["retrieval_downloads"]["retrieval_experiment_download"])
    assert len(complete["current"]["chunks"]) == 3
    assert len(app.session_state["last_search_results"]) == 3
    app.number_input(key="retrieved_context_last_rank").set_value(None).run()
    app.text_input(key="retrieved_context_search").input("STRASSE").run()
    assert "Rank 3 | Chunk index 12\nStraße outlook" in (
        app.session_state["retrieval_downloads"]["retrieved_context_text_download"]
    )
    app.text_input(key="retrieved_context_search").input("absent").run()
    assert "retrieved_context_text_download" not in app.session_state["retrieval_downloads"]
    app.number_input(key="retrieved_context_first_rank").set_value(3).run()
    app.number_input(key="retrieved_context_last_rank").set_value(2).run()
    assert "retrieved_context_text_download" not in app.session_state["retrieval_downloads"]
    assert not app.exception


def test_download_preserves_raw_text_and_excludes_metadata_and_scores():
    from types import SimpleNamespace
    from unittest.mock import patch

    from unravel.ui.components.retrieval_export import render_context_text_export

    text = "  ```\nCafé <tag>\n```\n\n"
    result = SimpleNamespace(index=42, text=text, score=0.5, metadata={"secret": "hidden"})
    with patch("unravel.ui.components.retrieval_export.st") as streamlit:
        render_context_text_export([(4, result)])
        kwargs = streamlit.download_button.call_args.kwargs
        assert kwargs["data"] == f"Rank 4 | Chunk index 42\n{text}\n"
        assert kwargs["mime"] == "text/plain"
        assert kwargs["file_name"] == "retrieved-context.txt"
        assert "hidden" not in kwargs["data"]
        assert "0.5" not in kwargs["data"]
        streamlit.download_button.reset_mock()
        render_context_text_export([])
        streamlit.download_button.assert_not_called()
