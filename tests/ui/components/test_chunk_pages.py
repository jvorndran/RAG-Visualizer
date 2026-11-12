import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from unravel.ui.components.chunk_explorer import (
    matching_chunk_indices,
    serialize_chunk_selection,
)


def test_page_filter_matches_reported_pages_and_composes_without_mutation():
    chunks = [
        SimpleNamespace(text=text, metadata=meta, start_index=i, end_index=i + len(text))
        for i, (text, meta) in enumerate([
            ("Needle", {"page_no": [1, 2], "element_type": "table"}),
            ("Needle", {"page_number": 2, "section_hierarchy": ["Details"]}),
            ("Long needle", {"page": 3}),
            ("No page", {}),
            ("Invalid", {"page_no": [True, 0, -1, "2", 2.5, None]}),
            ("Primary", {"page_no": [], "page_number": 2}),
        ])
    ]
    before = deepcopy(chunks)
    assert matching_chunk_indices(chunks, "", pages=[2]) == [0, 1]
    assert matching_chunk_indices(chunks, "", pages=[1, 3]) == [0, 2]
    assert matching_chunk_indices(chunks, "needle", 6, 6, True,
                                  element_types=["table"], pages=[2]) == [0]
    assert matching_chunk_indices(chunks, "", sections=["Details"], pages=[2]) == [1]
    assert matching_chunk_indices(chunks, "", pages=[99]) == []
    assert matching_chunk_indices(chunks, "", pages=[]) == list(range(6))
    assert chunks == before


def test_page_export_retains_original_numbers_offsets_and_sorted_text():
    chunks = [
        SimpleNamespace(text=text, metadata={"page_no": [page]}, start_index=i * 10,
                        end_index=i * 10 + len(text))
        for i, (text, page) in enumerate([("é", 2), ("long\nraw text", 2), ("Other", 3)])
    ]
    output = json.loads(serialize_chunk_selection(
        chunks, "", "doc.pdf", sort_order="Longest first", pages=[2]
    ))
    assert output["pages"] == [2]
    assert [row["chunk_number"] for row in output["chunks"]] == [2, 1]
    assert output["chunks"][0] == {
        "chunk_number": 2, "text": "long\nraw text", "start_index": 10, "end_index": 23
    }
    assert json.loads(serialize_chunk_selection(chunks, "", None))["pages"] == []


def test_page_control_invalidates_downloads_composes_and_resets_with_generation():
    script = '''
from types import SimpleNamespace
import streamlit as st
from unravel.ui.components.chunk_explorer import render_chunk_explorer
chunks = [SimpleNamespace(text=text, start_index=i, end_index=i + len(text), metadata=meta)
    for i, (text, meta) in enumerate([
        ("First", {"page_no": [1, 2]}), ("Second", {"page_number": 3}), ("No page", {})
    ])]
render_chunk_explorer(chunks, st.session_state.get("generation", "first"))
'''
    with patch("unravel.ui.components.chunk_explorer.st.download_button") as download:
        app = AppTest.from_string(script).run(timeout=30)
        assert not app.exception
        pages = next(widget for widget in app.multiselect if widget.label == "Source pages")
        assert pages.options == ["1", "2", "3"]
        app.button[0].click().run()
        prior_downloads = download.call_count
        next(w for w in app.multiselect if w.label == "Source pages").set_value([3]).run()
        assert not app.exception
        assert download.call_count == prior_downloads
        assert app.selectbox[0].value == 1
        app.button[0].click().run()
        output = json.loads(download.call_args.kwargs["data"])
        assert output["pages"] == [3]
        assert [row["chunk_number"] for row in output["chunks"]] == [2]
        app.text_input[0].set_value("First").run()
        assert not app.selectbox and app.button[0].disabled
        app.text_input[0].set_value("").run()
        app.session_state["generation"] = "next"
        app.run()
        assert next(w for w in app.multiselect if w.label == "Source pages").value == []
        assert len(app.selectbox[0].options) == 3
        assert "chunk_explorer_prepared_export" not in app.session_state
