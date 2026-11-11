import json
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

SCRIPT = '''
from copy import deepcopy
from types import SimpleNamespace
import streamlit as st
from unravel.ui.components.chunk_explorer import render_chunk_explorer

texts = ["First", "Before", "Straße\\ncontext", "Focus needle", "After", "Longest chunk text"]
chunks = [SimpleNamespace(text=text, start_index=i * 100, end_index=i * 100 + len(text),
    metadata={"chunk_index": i}) for i, text in enumerate(texts)]
before = deepcopy(chunks)
render_chunk_explorer(chunks, st.session_state.get("generation", "first"))
st.session_state["chunks_unchanged"] = chunks == before
'''


def test_neighbors_use_original_order_despite_filters_and_preserve_downloads():
    with patch("unravel.ui.components.chunk_explorer.st.download_button") as download:
        app = AppTest.from_string(SCRIPT).run(timeout=30)
        app.text_input[0].set_value("needle").run()
        app.button[0].click().run()
        prepared = dict(app.session_state["chunk_explorer_prepared_export"])
        app.number_input(key="chunk_explorer_neighbors_first").set_value(1).run()
        assert not app.exception
        assert [code.value for code in app.code] == ["Straße\ncontext", "Focus needle", "After"]
        assert any("Chunk 3 · Before selected chunk" in caption.value for caption in app.caption)
        assert any("Chunk 4 · Selected chunk" in caption.value for caption in app.caption)
        assert any("Chunk 5 · After selected chunk" in caption.value for caption in app.caption)
        assert any("Source offsets: 200–214" in caption.value for caption in app.caption)
        assert len(app.selectbox[0].options) == 1 and app.selectbox[0].value == 3
        assert app.session_state["chunk_explorer_prepared_export"] == prepared
        assert json.loads(download.call_args.kwargs["data"])["count"] == 1
        assert [row["chunk_number"] for row in json.loads(prepared["data"])["chunks"]] == [4]
        assert prepared["text_data"] == "Chunk 4\n\nFocus needle"
        assert app.session_state["chunks_unchanged"]
        app.number_input(key="chunk_explorer_neighbors_first").set_value(0).run()
        assert [code.value for code in app.code] == ["Focus needle"]
        assert not app.exception


def test_neighbor_window_clips_to_document_edges_after_sorting_and_resets():
    app = AppTest.from_string(SCRIPT).run(timeout=30)
    assert app.number_input(key="chunk_explorer_neighbors_first").value == 0
    app.radio[0].set_value("Longest first").run()
    assert app.selectbox[0].value == 5
    app.number_input(key="chunk_explorer_neighbors_first").set_value(3).run()
    assert [code.value for code in app.code] == [
        "Straße\ncontext", "Focus needle", "After", "Longest chunk text"
    ]
    app.selectbox[0].select(0).run()
    assert [code.value for code in app.code] == [
        "First", "Before", "Straße\ncontext", "Focus needle"
    ]
    assert app.selectbox[0].value == 0
    app.session_state["generation"] = "next"
    app.run()
    assert app.number_input(key="chunk_explorer_neighbors_next").value == 0
    assert [code.value for code in app.code] == ["First"]
    assert app.session_state["chunks_unchanged"]
    assert not app.exception


def test_neighbor_window_handles_single_empty_chunk_and_no_matches():
    script = SCRIPT.replace(
        '["First", "Before", "Straße\\ncontext", "Focus needle", "After", "Longest chunk text"]',
        '[""]',
    )
    app = AppTest.from_string(script).run(timeout=30)
    app.number_input(key="chunk_explorer_neighbors_first").set_value(3).run()
    assert len(app.code) == 1 and app.code[0].value == ""
    assert any("Chunk 1 · Selected chunk" in caption.value for caption in app.caption)
    app.text_input[0].set_value("missing").run()
    assert not app.code and not app.selectbox
    assert app.button[0].disabled
    assert not app.exception
