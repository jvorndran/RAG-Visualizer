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


def test_size_sorting_is_stable_preserves_identity_and_composes_with_filters():
    chunks = [SimpleNamespace(text=text, start_index=i, end_index=i + len(text))
              for i, text in enumerate(["xx", "", "long text", "yy", "xx"])]
    before = deepcopy(chunks)
    assert matching_chunk_indices(chunks, "", sort_order="Shortest first") == [1, 0, 3, 4, 2]
    assert matching_chunk_indices(chunks, "", sort_order="Longest first") == [2, 0, 3, 4, 1]
    assert matching_chunk_indices(chunks, "", 2, 2, True, "Longest first") == [0, 4]
    assert matching_chunk_indices(chunks, "text", sort_order="Shortest first") == [2]
    assert matching_chunk_indices([], "", sort_order="Longest first") == []
    assert chunks == before


def test_sorted_export_matches_inspection_order_without_renumbering():
    from unravel.ui.components.chunk_explorer import serialize_chunk_selection

    chunks = [SimpleNamespace(text=text, start_index=i, end_index=i + len(text))
              for i, text in enumerate(["é", "long text", "xy"])]
    output = json.loads(serialize_chunk_selection(
        chunks, "", "doc.txt", sort_order="Longest first"
    ))
    assert output["sort_order"] == "Longest first"
    assert [row["chunk_number"] for row in output["chunks"]] == [2, 3, 1]
    assert output["chunks"][0] == {
        "chunk_number": 2, "text": "long text", "start_index": 1, "end_index": 10
    }
    assert json.loads(serialize_chunk_selection(chunks, "", None))["sort_order"] == "Original order"


def test_size_bounds_are_inclusive_use_raw_characters_and_compose_with_search():
    chunks = [SimpleNamespace(text=text) for text in ["", "é", "Straße", "STRASSE", "long text"]]
    before = deepcopy(chunks)
    assert matching_chunk_indices(chunks, "", 0, 0) == [0]
    assert matching_chunk_indices(chunks, "", 1, 6) == [1, 2]
    assert matching_chunk_indices(chunks, "strasse", 6, 6) == [2]
    assert matching_chunk_indices(chunks, "", 7) == [3, 4]
    assert matching_chunk_indices([], "", 1, 5) == []
    assert chunks == before


def test_size_filtered_export_records_bounds_and_original_chunk_identity():
    from unravel.ui.components.chunk_explorer import serialize_chunk_selection

    chunks = [SimpleNamespace(text=text, start_index=i, end_index=i + len(text))
              for i, text in enumerate(["tiny", "needle", "Long needle"])]
    output = json.loads(serialize_chunk_selection(chunks, "needle", "file.txt", 6, 6))
    assert output["min_characters"] == 6
    assert output["max_characters"] == 6
    assert output["count"] == 1
    assert output["total_chunks"] == 3
    assert output["chunks"] == [
        {"chunk_number": 2, "text": "needle", "start_index": 1, "end_index": 7}
    ]


def test_repeated_text_filter_keeps_every_exact_occurrence_and_composes_with_bounds():
    texts = ["", "Header", "header", " Header", "Header", "", "Straße", "STRASSE", "Straße"]
    chunks = [SimpleNamespace(text=text) for text in texts]
    before = deepcopy(chunks)
    assert matching_chunk_indices(chunks, "", repeated_text_only=True) == [0, 1, 4, 5, 6, 8]
    assert matching_chunk_indices(chunks, "HEADER", 6, 6, repeated_text_only=True) == [1, 4]
    assert matching_chunk_indices(chunks, "strasse", repeated_text_only=True) == [6, 8]
    assert matching_chunk_indices(chunks, "", 0, 0, repeated_text_only=True) == [0, 5]
    assert matching_chunk_indices(chunks, "", repeated_text_only=False) == list(range(9))
    assert matching_chunk_indices([], "", repeated_text_only=True) == []
    assert matching_chunk_indices(chunks[:2], "", repeated_text_only=True) == []
    assert chunks == before


def test_repeated_text_export_records_filter_without_renumbering_or_removing_copies():
    from unravel.ui.components.chunk_explorer import serialize_chunk_selection

    chunks = [SimpleNamespace(text=text, start_index=i * 100, end_index=i * 100 + len(text))
              for i, text in enumerate(["unique", "Straße", "STRASSE", "Straße"])]
    before = deepcopy(chunks)
    output = json.loads(serialize_chunk_selection(
        chunks, "STRASSE", "file.txt", 6, 6, repeated_text_only=True
    ))
    assert output["repeated_text_only"] is True
    assert output["count"] == 2
    assert output["total_chunks"] == 4
    assert output["chunks"] == [
        {"chunk_number": 2, "text": "Straße", "start_index": 100, "end_index": 106},
        {"chunk_number": 4, "text": "Straße", "start_index": 300, "end_index": 306},
    ]
    assert json.loads(serialize_chunk_selection(chunks, "", None))["repeated_text_only"] is False
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


def test_sort_control_invalidates_exports_and_resets_for_a_new_generation():
    from unittest.mock import patch

    with patch("unravel.ui.components.chunk_explorer.st.download_button") as download:
        app = AppTest.from_string(SCRIPT).run()
        app.button[0].click().run()
        previous_downloads = download.call_count
        app.radio[0].set_value("Longest first").run()
        assert not app.exception
        assert app.selectbox[0].value == 2
        assert "Chunk 3" in app.selectbox[0].options[0]
        assert download.call_count == previous_downloads
        app.selectbox[0].select(1).run()
        assert app.code[0].value == "# Heading"
        app.button[0].click().run()
        exported = json.loads(download.call_args.kwargs["data"])
        assert exported["sort_order"] == "Longest first"
        assert [chunk["chunk_number"] for chunk in exported["chunks"]] == [3, 2, 1]
        app.text_input[0].set_value("missing").run()
        assert not app.selectbox
        assert app.button[0].disabled
        app.text_input[0].set_value("").run()
        assert app.selectbox[0].value == 2
        app.radio[0].set_value("Shortest first").run()
        assert app.selectbox[0].value == 0
        app.session_state["generation"] = "second"
        app.run()
        assert app.radio[0].value == "Original order"
        assert app.selectbox[0].value == 0
        assert "chunk_explorer_prepared_export" not in app.session_state


def test_repeated_text_control_invalidates_exports_recovers_and_resets_with_generation():
    from unittest.mock import patch

    from unravel.ui.components.chunk_explorer import serialize_chunk_selection

    script = SCRIPT.replace(
        '["First", "# Heading", ', '["First", "# Heading", "# Heading", ',
    )
    with (
        patch("unravel.ui.components.chunk_explorer.st.download_button") as download,
        patch("unravel.ui.components.chunk_explorer.serialize_chunk_selection",
              wraps=serialize_chunk_selection) as serialize,
    ):
        app = AppTest.from_string(script).run()
        assert not app.exception
        assert len(app.selectbox[0].options) == 4
        app.button[0].click().run()
        assert serialize.call_count == 1
        previous_downloads = download.call_count
        app.checkbox[0].check().run()
        assert not app.exception
        assert serialize.call_count == 1
        assert download.call_count == previous_downloads
        assert "2 of 4" in app.caption[0].value
        assert app.selectbox[0].value == 1
        app.selectbox[0].select(2).run()
        app.button[0].click().run()
        assert serialize.call_count == 2
        exported = json.loads(download.call_args.kwargs["data"])
        assert exported["repeated_text_only"] is True
        assert [chunk["chunk_number"] for chunk in exported["chunks"]] == [2, 3]
        app.selectbox[0].select(1).run()
        assert serialize.call_count == 2
        app.text_input[0].set_value("First").run()
        assert not app.selectbox
        assert app.button[0].disabled
        app.checkbox[0].uncheck().run()
        assert app.code[0].value == "First"
        app.checkbox[0].check().run()
        app.session_state["generation"] = "second"
        app.run()
        assert not app.exception
        assert app.checkbox[0].value is False
        assert len(app.selectbox[0].options) == 4
        assert "chunk_explorer_prepared_export" not in app.session_state


def test_size_controls_invalidate_downloads_and_handle_invalid_ranges():
    from unittest.mock import patch

    with patch("unravel.ui.components.chunk_explorer.st.download_button") as download:
        app = AppTest.from_string(SCRIPT).run()
        app.button[0].click().run()
        previous_downloads = download.call_count
        app.number_input[0].set_value(6).run()
        assert not app.exception
        assert download.call_count == previous_downloads
        assert app.selectbox[0].value == 1
        assert len(app.selectbox[0].options) == 2
        app.number_input[1].set_value(9).run()
        assert len(app.selectbox[0].options) == 1
        assert app.code[0].value == "# Heading"
        app.button[0].click().run()
        assert json.loads(download.call_args.kwargs["data"])["count"] == 1
        previous_downloads = download.call_count
        app.number_input[1].set_value(5).run()
        assert not app.exception
        assert app.error
        assert not app.selectbox
        assert download.call_count == previous_downloads
        app.number_input[1].set_value(None).run()
        assert not app.error
        assert len(app.selectbox[0].options) == 2
        app.session_state["generation"] = "second"
        app.run()
        assert app.number_input[0].value == 0
        assert app.number_input[1].value is None
        assert len(app.selectbox[0].options) == 3


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


def test_download_preparation_is_explicit_reused_and_invalidated_by_search_or_generation():
    from unittest.mock import patch

    from unravel.ui.components.chunk_explorer import serialize_chunk_selection

    with (
        patch("unravel.ui.components.chunk_explorer.st.download_button") as download,
        patch("unravel.ui.components.chunk_explorer.serialize_chunk_selection",
              wraps=serialize_chunk_selection) as serialize,
    ):
        app = AppTest.from_string(SCRIPT).run()
        assert not app.exception
        serialize.assert_not_called()
        download.assert_not_called()
        app.selectbox[0].select(1).run()
        serialize.assert_not_called()
        app.button[0].click().run()
        assert serialize.call_count == 1
        assert json.loads(download.call_args.kwargs["data"])["count"] == 3
        assert download.call_args.kwargs["mime"] == "application/json"
        assert download.call_args.kwargs["file_name"] == "chunks.json"
        app.selectbox[0].select(2).run()
        assert serialize.call_count == 1
        previous_downloads = download.call_count
        app.text_input[0].set_value("strasse").run()
        assert serialize.call_count == 1
        assert download.call_count == previous_downloads
        app.button[0].click().run()
        assert serialize.call_count == 2
        assert json.loads(download.call_args.kwargs["data"])["chunks"][0]["chunk_number"] == 3
        previous_downloads = download.call_count
        app.text_input[0].set_value("absent").run()
        assert serialize.call_count == 2
        assert download.call_count == previous_downloads
        assert app.button[0].disabled
        app.session_state["generation"] = "second"
        app.run()
        assert app.text_input[0].value == ""
        assert serialize.call_count == 2
        assert download.call_count == previous_downloads
        app.button[0].click().run()
        assert serialize.call_count == 3
        assert json.loads(download.call_args.kwargs["data"])["count"] == 3
