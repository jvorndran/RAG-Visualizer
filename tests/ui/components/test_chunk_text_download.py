from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from unravel.ui.components.chunk_explorer import serialize_chunk_text


def test_text_bundle_keeps_order_identity_full_whitespace_and_unicode():
    chunks = [SimpleNamespace(text=text, metadata={"secret": "excluded"})
              for text in ["  Straße\n\t", "", "x" * 10000]]
    before = deepcopy(chunks)
    output = serialize_chunk_text(chunks, [2, 0, 1])
    assert output == (
        "Chunk 3\n\n" + "x" * 10000 + "\n\n---\n\n"
        "Chunk 1\n\n  Straße\n\t\n\n---\n\nChunk 2\n\n"
    )
    assert serialize_chunk_text(chunks, []) == ""
    assert chunks == before


SCRIPT = '''
from types import SimpleNamespace
import streamlit as st
from unravel.ui.components.chunk_explorer import render_chunk_explorer
chunks = [SimpleNamespace(text=text, start_index=i, end_index=i + len(text),
    metadata={"element_type": "table"}) for i, text in enumerate(["é", "Long é", "other"])]
render_chunk_explorer(chunks, st.session_state.get("generation", "first"))
'''


def test_text_download_prepares_explicitly_tracks_filters_and_resets():
    with patch("unravel.ui.components.chunk_explorer.st.download_button") as download:
        app = AppTest.from_string(SCRIPT).run(timeout=30)
        assert not app.exception
        download.assert_not_called()
        app.text_input[0].set_value("é").run()
        app.radio[0].set_value("Longest first").run()
        app.button[0].click().run()
        assert not app.exception
        calls = {call.args[0]: call.kwargs for call in download.call_args_list}
        text = calls["Download matching chunks as text"]
        assert text["data"] == "Chunk 2\n\nLong é\n\n---\n\nChunk 1\n\né"
        assert text["file_name"] == "chunks.txt"
        assert text["mime"] == "text/plain; charset=utf-8"
        assert "Download matching chunks" in calls
        previous = download.call_count
        app.text_input[0].set_value("missing").run()
        assert download.call_count == previous
        assert app.button[0].disabled
        app.session_state["generation"] = "second"
        app.run()
        assert not app.exception
        assert download.call_count == previous
        assert "chunk_explorer_prepared_export" not in app.session_state


def test_json_only_preparation_from_an_older_session_is_invalidated():
    app = AppTest.from_string(SCRIPT)
    app.session_state["chunk_explorer_prepared_export"] = {
        "generation": "first", "query": "", "min_characters": 0, "max_characters": None,
        "data": "old JSON", "sort_order": "Original order",
    }
    with patch("unravel.ui.components.chunk_explorer.st.download_button") as download:
        app.run(timeout=30)
        assert not app.exception
        download.assert_not_called()
        assert "chunk_explorer_prepared_export" not in app.session_state
