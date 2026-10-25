import hashlib
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

SCRIPT = '''
import streamlit as st
from unravel.ui.components.text_ingestion import render_text_ingestion
render_text_ingestion()
'''


def test_paste_saves_only_on_submit_preserves_unicode_and_invalidates_pipeline():
    text = "  Straße\n\n# Heading\n尾\t "
    content = text.encode("utf-8")
    filename = f"pasted-{hashlib.sha256(content).hexdigest()}.txt"
    with patch("unravel.ui.components.text_ingestion.save_document",
               return_value=Path("/isolated") / filename) as save:
        app = AppTest.from_string(SCRIPT).run()
        assert not app.exception
        app.text_area[0].set_value(text).run()
        save.assert_not_called()
        for key in ["chunks", "last_embeddings_result", "search_results", "bm25_index"]:
            app.session_state[key] = "stale"
        app.session_state["document_metadata"] = {"name": "previous.pdf"}
        app.session_state["embedding_model_name"] = "keep-model"
        app.button[0].click().run()
        assert not app.exception
        save.assert_called_once_with(filename, content)
        assert app.session_state["doc_name"] == filename
        assert app.session_state["document_metadata"] == {
            "name": filename, "format": "TXT", "size_bytes": len(content),
            "path": str(Path("/isolated") / filename), "source": "text",
        }
        for key in ["chunks", "last_embeddings_result", "search_results", "bm25_index"]:
            assert key not in app.session_state
        assert app.session_state["embedding_model_name"] == "keep-model"
        app.run()
        assert save.call_count == 1


def test_blank_text_never_saves_or_changes_existing_document():
    with patch("unravel.ui.components.text_ingestion.save_document") as save:
        app = AppTest.from_string(SCRIPT).run()
        app.session_state["doc_name"] = "previous.txt"
        app.session_state["chunks"] = ["keep"]
        app.text_area[0].set_value(" \n\t ")
        app.button[0].click().run()
        assert not app.exception
        assert "Enter some text" in app.error[0].value
        save.assert_not_called()
        assert app.session_state["doc_name"] == "previous.txt"
        assert app.session_state["chunks"] == ["keep"]


def test_storage_failure_keeps_document_state_and_draft_for_retry():
    with patch("unravel.ui.components.text_ingestion.save_document",
               side_effect=OSError("disk unavailable")):
        app = AppTest.from_string(SCRIPT).run()
        app.session_state["doc_name"] = "previous.txt"
        app.session_state["chunks"] = ["keep"]
        app.text_area[0].set_value("New document")
        app.button[0].click().run()
        assert not app.exception
        assert "disk unavailable" in app.error[0].value
        assert app.text_area[0].value == "New document"
        assert app.session_state["doc_name"] == "previous.txt"
        assert app.session_state["chunks"] == ["keep"]


def test_replacing_paste_changes_document_identity_so_parsed_caches_cannot_be_reused():
    with patch("unravel.ui.components.text_ingestion.save_document",
               return_value=Path("/isolated/new.txt")) as save:
        app = AppTest.from_string(SCRIPT).run()
        app.text_area[0].set_value("First").run()
        app.button[0].click().run()
        first_name = app.session_state["doc_name"]
        app.text_area[0].set_value("Second").run()
        app.button[0].click().run()
        assert not app.exception
        assert app.session_state["doc_name"] != first_name
        assert save.call_count == 2


def test_upload_step_routes_paste_text_and_keeps_demo_url_only(monkeypatch):
    script = '''
from unravel.ui.steps.upload import render_upload_step
render_upload_step()
'''
    monkeypatch.delenv("DEMO_MODE", raising=False)
    with (
        patch("unravel.ui.steps.upload.get_current_document", return_value=None),
        patch("unravel.ui.steps.upload.st.segmented_control", return_value="Paste text"),
    ):
        app = AppTest.from_string(script).run()
        assert not app.exception
        assert app.text_area[0].label == "Document text"
        assert app.button[0].label == "Add text"
        monkeypatch.setenv("DEMO_MODE", "true")
        with patch("unravel.ui.steps.upload._render_url_scraping") as scrape:
            app.run()
            assert not app.exception
            assert not app.text_area
            scrape.assert_called_once()


def test_real_storage_round_trip_replaces_previous_document_in_isolated_directory(tmp_path):
    from unravel.services.storage import get_current_document, save_document

    with patch("unravel.services.storage.DEFAULT_STORAGE_DIR", tmp_path):
        save_document("previous.txt", b"previous")
        app = AppTest.from_string(SCRIPT).run()
        text = "  document\n尾\t "
        app.text_area[0].set_value(text).run()
        assert get_current_document() == ("previous.txt", b"previous")
        app.button[0].click().run()
        assert not app.exception
        assert get_current_document() == (app.session_state["doc_name"], text.encode("utf-8"))
        assert len(list((tmp_path / "documents").iterdir())) == 1
