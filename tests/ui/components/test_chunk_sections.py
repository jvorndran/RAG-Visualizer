import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from unravel.ui.components.chunk_explorer import (
    chunk_sections,
    matching_chunk_indices,
    serialize_chunk_selection,
)


def test_section_labels_include_hierarchy_and_legacy_headers_without_changing_chunks():
    chunks = [
        SimpleNamespace(text=text, metadata=metadata, start_index=i, end_index=i + len(text))
        for i, (text, metadata) in enumerate([
            ("Needle", {"section_hierarchy": ["Overview", "Straße"], "element_type": "table"}),
            ("Needle", {"Header 1": "Overview", "Header 6": "Legacy"}),
            ("Needle", {}),
            ("Long needle", {"section_hierarchy": ["Overview", "Details", None, 7]}),
            ("Other", {"section_hierarchy": [], "Header 1": "Ignored"}),
        ])
    ]
    before = deepcopy(chunks)
    assert chunk_sections(chunks[0]) == {"Overview", "Straße"}
    assert chunk_sections(chunks[1]) == {"Overview", "Legacy"}
    assert chunk_sections(chunks[4]) == set()
    assert matching_chunk_indices(chunks, "", sections=["Overview"]) == [0, 1, 3]
    assert matching_chunk_indices(chunks, "", sections=["Straße", "Legacy"]) == [0, 1]
    assert matching_chunk_indices(chunks, "needle", 6, 6, True,
                                  element_types=["table"], sections=["Overview"]) == [0]
    assert matching_chunk_indices(chunks, "", sections=["Missing"]) == []
    assert matching_chunk_indices(chunks, "", sections=[]) == list(range(5))
    output = json.loads(serialize_chunk_selection(
        chunks, "needle", "doc.md", sort_order="Longest first", sections=["Overview"]
    ))
    assert output["sections"] == ["Overview"]
    assert [row["chunk_number"] for row in output["chunks"]] == [4, 1, 2]
    assert output["chunks"][0]["start_index"] == 3
    assert json.loads(serialize_chunk_selection(chunks, "", None))["sections"] == []
    assert chunks == before


def test_section_control_invalidates_download_and_resets_on_new_generation():
    script = '''
from types import SimpleNamespace
import streamlit as st
from unravel.ui.components.chunk_explorer import render_chunk_explorer
chunks = [SimpleNamespace(text=text, start_index=i, end_index=i + len(text), metadata=metadata)
    for i, (text, metadata) in enumerate([
        ("First", {"section_hierarchy": ["Intro"], "element_type": "table"}),
        ("Second", {"Header 1": "Details", "element_type": "code"}),
        ("No heading", {}),
    ])]
render_chunk_explorer(chunks, st.session_state.get("generation", "first"))
'''
    with patch("unravel.ui.components.chunk_explorer.st.download_button") as download:
        app = AppTest.from_string(script).run(timeout=30)
        assert not app.exception
        assert app.multiselect[1].options == ["Details", "Intro"]
        app.button[0].click().run()
        downloads = download.call_count
        app.multiselect[1].set_value(["Details"]).run()
        assert not app.exception
        assert download.call_count == downloads
        assert app.selectbox[0].value == 1
        app.button[0].click().run()
        output = json.loads(download.call_args.kwargs["data"])
        assert output["sections"] == ["Details"]
        assert [row["chunk_number"] for row in output["chunks"]] == [2]
        app.multiselect[0].set_value(["table"]).run()
        assert not app.selectbox and app.button[0].disabled
        app.multiselect[0].set_value([]).run()
        assert app.selectbox[0].value == 1
        app.session_state["generation"] = "next"
        app.run()
        assert app.multiselect[1].value == []
        assert len(app.selectbox[0].options) == 3
        assert "chunk_explorer_prepared_export" not in app.session_state
