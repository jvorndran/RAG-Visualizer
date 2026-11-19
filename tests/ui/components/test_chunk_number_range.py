import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from unravel.ui.components.chunk_explorer import matching_chunk_indices, serialize_chunk_selection


def make_chunks():
    return [SimpleNamespace(text=text, start_index=i * 10, end_index=i * 10 + len(text),
                            metadata={"page_no": 1})
            for i, text in enumerate(["repeat", "repeat", "long repeat", "repeat"])]


def test_chunk_range_uses_original_numbers_before_deduplication_and_sorting():
    chunks = make_chunks()
    before = deepcopy(chunks)
    assert matching_chunk_indices(chunks, "", min_chunk_number=2, max_chunk_number=3) == [1, 2]
    assert matching_chunk_indices(chunks, "repeat", repeated_text_only=True,
                                  unique_text_only=True, min_chunk_number=2,
                                  max_chunk_number=4, sort_order="Longest first") == [1]
    assert matching_chunk_indices(chunks, "", min_chunk_number=2, pages=[1],
                                  sort_order="Longest first") == [2, 1, 3]
    assert matching_chunk_indices(chunks, "", min_chunk_number=4, max_chunk_number=2) == []
    assert matching_chunk_indices(chunks, "", min_chunk_number=99) == []
    assert chunks == before


def test_chunk_range_export_keeps_raw_text_offsets_and_original_numbers():
    output = json.loads(serialize_chunk_selection(
        make_chunks(), "", "source.txt", min_chunk_number=2, max_chunk_number=3,
        sort_order="Longest first"
    ))
    assert output["min_chunk_number"] == 2
    assert output["max_chunk_number"] == 3
    assert output["total_chunks"] == 4 and output["count"] == 2
    assert [row["chunk_number"] for row in output["chunks"]] == [3, 2]
    assert output["chunks"][0] == {
        "chunk_number": 3, "text": "long repeat", "start_index": 20, "end_index": 31
    }


def test_chunk_range_controls_invalidate_downloads_and_reset_on_new_generation():
    script = '''
from types import SimpleNamespace
import streamlit as st
from unravel.ui.components.chunk_explorer import render_chunk_explorer
chunks = [SimpleNamespace(text=str(i), start_index=i, end_index=i+1, metadata={})
          for i in range(4)]
render_chunk_explorer(chunks, st.session_state.get("generation", "first"))
'''
    with patch("unravel.ui.components.chunk_explorer.st.download_button") as download:
        app = AppTest.from_string(script).run(timeout=30)
        assert not app.exception
        app.button[0].click().run()
        previous = download.call_count
        next(w for w in app.number_input if w.label == "First chunk number").set_value(2).run()
        assert not app.exception and download.call_count == previous
        next(w for w in app.number_input if w.label == "Last chunk number").set_value(3).run()
        assert app.selectbox[0].options == ["Chunk 2 · 1 characters", "Chunk 3 · 1 characters"]
        app.button[0].click().run()
        assert json.loads(download.call_args.kwargs["data"])["count"] == 2
        neighbors = next(
            w for w in app.number_input if w.label == "Neighboring chunks on each side"
        )
        neighbors.set_value(1).run()
        assert [w.value for w in app.code] == ["0", "1", "2"]
        next(w for w in app.number_input if w.label == "First chunk number").set_value(4).run()
        assert app.error and not app.selectbox
        assert "chunk_explorer_prepared_export" not in app.session_state
        app.session_state["generation"] = "second"
        app.run()
        assert not app.exception and not app.error
        assert next(w for w in app.number_input if w.label == "First chunk number").value == 1
        assert next(w for w in app.number_input if w.label == "Last chunk number").value is None
        assert len(app.selectbox[0].options) == 4
