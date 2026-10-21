import json
from copy import deepcopy
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

from unravel.ui.components.chunk_explorer import matching_chunk_indices


def test_literal_unicode_search_keeps_original_indices_and_inputs():
    chunks = [SimpleNamespace(text=text) for text in ["", "Straße\nDetails", "[Raw]", "STRASSE"]]
    before = deepcopy(chunks)
    assert matching_chunk_indices(chunks, " STRASSE ") == [1, 3]
    assert matching_chunk_indices(chunks, "[raw]") == [2]
    assert matching_chunk_indices(chunks, "a.*") == []
    assert matching_chunk_indices(chunks, "  ") == [0, 1, 2, 3]
    assert matching_chunk_indices([], "") == []
    assert chunks == before


SCRIPT = '''
from types import SimpleNamespace
import streamlit as st
from unravel.ui.components.chunk_explorer import render_chunk_explorer
chunks = [SimpleNamespace(text=text, start_index=i * 100, end_index=i * 100 + len(text),
    metadata={"source": "fixture", "chunk_index": i})
    for i, text in enumerate(["First", "# Heading", "Straße\\n" + "x" * 10000])]
render_chunk_explorer(chunks, st.session_state.get("generation", "first"))
'''


def test_explorer_filters_selects_full_text_and_recovers_from_no_matches():
    app = AppTest.from_string(SCRIPT).run()
    assert not app.exception
    assert len(app.selectbox[0].options) == 3
    app.text_input[0].set_value("strasse").run()
    assert not app.exception
    assert len(app.selectbox[0].options) == 1
    assert app.selectbox[0].value == 2
    assert "Chunk 3" in app.selectbox[0].options[0]
    assert app.code[0].value == "Straße\n" + "x" * 10000
    assert "Source offsets: 200" in app.caption[1].value
    app.text_input[0].set_value("missing").run()
    assert len(app.selectbox) == 0
    assert len(app.code) == 0
    assert "No matching chunks" in app.info[0].value
    app.text_input[0].set_value("").run()
    app.selectbox[0].select(1).run()
    assert app.code[0].value == "# Heading"
    assert not app.exception


def test_new_generation_uses_fresh_widget_state_and_empty_chunks_are_safe():
    app = AppTest.from_string(SCRIPT).run()
    app.text_input[0].set_value("missing").run()
    app.session_state["generation"] = "second"
    app.run()
    assert app.text_input[0].value == ""
    assert len(app.selectbox[0].options) == 3
    empty_script = SCRIPT.replace("render_chunk_explorer(chunks,", "render_chunk_explorer([],"
    )
    empty = AppTest.from_string(empty_script).run()
    assert not empty.exception
    assert len(empty.selectbox) == 0
    assert "0 of 0" in empty.caption[0].value


def test_chunk_export_is_unicode_json_with_full_text_original_numbers_and_offsets():
    from unravel.ui.components.chunk_explorer import serialize_chunk_selection

    chunks = [
        SimpleNamespace(text="Hidden", start_index=0, end_index=6, metadata={}),
        SimpleNamespace(text="Straße\n" + "x" * 10000, start_index=None, end_index=None,
                        metadata={"api_key": "must-not-be-exported"}),
        SimpleNamespace(text="STRASSE", start_index=200, end_index=207, metadata={}),
    ]
    before = deepcopy(chunks)
    raw = serialize_chunk_selection(chunks, " STRASSE ", "example.txt")
    output = json.loads(raw)
    assert output["schema_version"] == 1
    assert output["document"] == "example.txt"
    assert output["search_text"] == "STRASSE"
    assert output["case_sensitive"] is False
    assert output["total_chunks"] == 3
    assert output["count"] == 2
    assert output["chunks"] == [
        {"chunk_number": 2, "text": chunks[1].text, "start_index": None, "end_index": None},
        {"chunk_number": 3, "text": "STRASSE", "start_index": 200, "end_index": 207},
    ]
    assert "Straße" in raw
    assert "must-not-be-exported" not in raw
    assert raw.endswith("\n")
    assert chunks == before
    assert json.loads(serialize_chunk_selection(chunks, "absent", "test"))["chunks"] == []
    assert json.loads(serialize_chunk_selection([], "", None))["count"] == 0


def test_download_control_tracks_search_and_disables_empty_selection():
    from unittest.mock import patch

    with patch("unravel.ui.components.chunk_explorer.st.download_button") as download:
        app = AppTest.from_string(SCRIPT).run()
        assert not app.exception
        assert json.loads(download.call_args.kwargs["data"])["count"] == 3
        assert download.call_args.kwargs["mime"] == "application/json"
        assert download.call_args.kwargs["file_name"] == "chunks.json"
        assert download.call_args.kwargs["disabled"] is False
        app.text_input[0].set_value("strasse").run()
        assert json.loads(download.call_args.kwargs["data"])["chunks"][0]["chunk_number"] == 3
        app.text_input[0].set_value("absent").run()
        assert json.loads(download.call_args.kwargs["data"])["count"] == 0
        assert download.call_args.kwargs["disabled"] is True
