"""Add pasted text through the normal single-document storage workflow."""

import hashlib

import streamlit as st

from unravel.services.storage import list_documents, save_document


def render_text_ingestion() -> None:
    """Save UTF-8 text only after an explicit, nonempty form submission."""
    st.caption("Paste text or Markdown as your source. Adding it replaces the current document.")
    with st.form("upload_paste_text"):
        source_format = st.selectbox(
            "Source format", ["Plain text", "Markdown"], key="upload_paste_text_format"
        )
        text = st.text_area("Document text", height=240, key="upload_paste_text_content")
        submitted = st.form_submit_button("Add text", type="primary")
    if not submitted:
        return
    if not text.strip():
        st.error("Enter some text before adding the document.")
        return

    content = text.encode("utf-8")
    # Parsing caches use document names. Changed text must get a fresh identity.
    extension = "md" if source_format == "Markdown" else "txt"
    filename = f"pasted-{hashlib.sha256(content).hexdigest()}.{extension}"
    try:
        doc_path = save_document(filename, content)
    except Exception as error:
        st.error(f"Failed to add text: {error}")
        return

    list_documents.clear()
    st.session_state.document_metadata = {
        "name": filename,
        "format": extension.upper(),
        "size_bytes": len(content),
        "path": str(doc_path),
        "source": "text",
    }
    st.session_state.doc_name = filename
    for key in ["chunks", "last_embeddings_result", "search_results", "bm25_index"]:
        st.session_state.pop(key, None)
    st.rerun()
