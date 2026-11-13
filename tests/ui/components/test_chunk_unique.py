import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from unravel.ui.components.chunk_explorer import matching_chunk_indices, serialize_chunk_selection


def make_chunks():
    return [
        SimpleNamespace(
            text=text, metadata={"page_no": page}, start_index=i * 10, end_index=i * 10 + len(text)
        )
        for i, (text, page) in enumerate(
            [("é", 1), ("", 1), ("é", 2), ("", 2), ("É", 2), (" é", 2), ("long", 2)]
        )
    ]


def test_unique_text_keeps_first_matching_occurrence_and_composes_with_other_filters():
    chunks = make_chunks()
    before = deepcopy(chunks)
    assert matching_chunk_indices(chunks, "", unique_text_only=True) == [0, 1, 4, 5, 6]
    assert matching_chunk_indices(chunks, "", pages=[2], unique_text_only=True) == [2, 3, 4, 5, 6]
    assert matching_chunk_indices(chunks, "É", 1, 1, True, unique_text_only=True) == [0]
    assert matching_chunk_indices(
        chunks, "", sort_order="Longest first", unique_text_only=True
    ) == [6, 5, 0, 4, 1]
    assert matching_chunk_indices(chunks, "", unique_text_only=False) == list(range(7))
    assert matching_chunk_indices([], "", unique_text_only=True) == []
    assert chunks == before


def test_unique_export_retains_original_identity_and_raw_text():
    output = json.loads(
        serialize_chunk_selection(make_chunks(), "", "doc.pdf", pages=[2], unique_text_only=True)
    )
    assert output["unique_text_only"] is True
    assert output["count"] == 5
    assert output["total_chunks"] == 7
    assert output["chunks"][0] == {
        "chunk_number": 3,
        "text": "é",
        "start_index": 20,
        "end_index": 21,
    }
    assert (
        json.loads(serialize_chunk_selection(make_chunks(), "", None))["unique_text_only"] is False
    )


def test_unique_control_invalidates_both_downloads_and_resets_with_generation():
    script = """
from types import SimpleNamespace
import streamlit as st
from unravel.ui.components.chunk_explorer import render_chunk_explorer
chunks = [SimpleNamespace(text=text, metadata={}, start_index=i, end_index=i+len(text))
          for i, text in enumerate(["same", "long raw text", "same"])]
render_chunk_explorer(chunks, st.session_state.get("generation", "first"))
"""
    with patch("unravel.ui.components.chunk_explorer.st.download_button") as download:
        app = AppTest.from_string(script).run(timeout=30)
        assert not app.exception
        app.button[0].click().run()
        previous = download.call_count
        next(w for w in app.checkbox if w.label == "One chunk per exact text").check().run()
        assert not app.exception
        assert download.call_count == previous
        assert app.selectbox[0].options == ["Chunk 1 · 4 characters", "Chunk 2 · 13 characters"]
        app.button[0].click().run()
        assert json.loads(download.call_args.kwargs["data"])["unique_text_only"] is True
        assert (
            download.call_args_list[-2].kwargs["data"]
            == "Chunk 1\n\nsame\n\n---\n\nChunk 2\n\nlong raw text"
        )
        app.session_state["generation"] = "next"
        app.run()
        assert not next(w for w in app.checkbox if w.label == "One chunk per exact text").value
        assert len(app.selectbox[0].options) == 3
        assert "chunk_explorer_prepared_export" not in app.session_state
