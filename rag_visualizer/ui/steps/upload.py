import html
from pathlib import Path

import streamlit as st
import streamlit_shadcn_ui as ui

from rag_visualizer.services.storage import (
    clear_documents,
    get_current_document,
    save_document,
)
from rag_visualizer.utils.parsers import get_file_preview, parse_document

DEFAULT_MAX_DOC_CHAR_COUNT = 40_000


def render_upload_step() -> None:
    # Initialize session state for document metadata
    if "document_metadata" not in st.session_state:
        st.session_state.document_metadata = None

    # Initialize character limit state
    if "max_char_limit_enabled" not in st.session_state:
        st.session_state.max_char_limit_enabled = True
    if "parsing_params" not in st.session_state:
        st.session_state.parsing_params = {}
    if "max_characters" not in st.session_state.parsing_params:
        st.session_state.parsing_params["max_characters"] = DEFAULT_MAX_DOC_CHAR_COUNT

    # Check if stored document matches session state
    current_doc = get_current_document()
    if current_doc:
        doc_name, _ = current_doc
        # If session state has different document, clear it
        if st.session_state.document_metadata and st.session_state.document_metadata.get("name") != doc_name:
            st.session_state.document_metadata = None
    else:
        # No document stored, clear metadata
        st.session_state.document_metadata = None

    # File uploader section
    with st.container(border=True):
        st.markdown("### Upload Document")
        st.caption("Supported formats: PDF, DOCX, PPTX, XLSX, HTML, MD, TXT, PNG, JPG")

        limit_enabled = st.checkbox(
            "Limit parsed content",
            value=st.session_state.get("max_char_limit_enabled", True),
            help="Keeps parsing fast. Turn off to process the entire document (may be slow).",
            key="upload_char_limit_checkbox",
        )
        st.session_state.max_char_limit_enabled = limit_enabled

        current_limit = st.session_state.parsing_params.get("max_characters", DEFAULT_MAX_DOC_CHAR_COUNT) or DEFAULT_MAX_DOC_CHAR_COUNT
        max_chars_value = st.number_input(
            "Max characters to parse",
            min_value=1_000,
            max_value=1_000_000,
            step=1_000,
            value=int(current_limit),
            key="upload_max_chars_input",
            disabled=not limit_enabled,
            help="Only the first N characters will be parsed when enabled.",
        )

        effective_max_chars = int(max_chars_value) if limit_enabled else None

        # Sync limit into parsing params so downstream steps honor it
        parsing_params = st.session_state.get("parsing_params", {})
        previous_limit = parsing_params.get("max_characters")
        if previous_limit != effective_max_chars:
            parsing_params["max_characters"] = effective_max_chars
            st.session_state.parsing_params = parsing_params
            # Invalidate derived state when the limit changes
            for key in ["chunks", "last_embeddings_result", "search_results"]:
                if key in st.session_state:
                    del st.session_state[key]

        uploaded_file = st.file_uploader(
            "Choose a file to upload",
            type=["pdf", "docx", "pptx", "xlsx", "html", "htm", "md", "txt", "png", "jpg", "jpeg", "bmp", "tiff", "tif"],
            accept_multiple_files=False,
            label_visibility="collapsed"
        )

        # Handle file upload
        if uploaded_file is not None:
            # Check if this is a new file (different from current)
            is_new_file = (
                st.session_state.document_metadata is None
                or st.session_state.document_metadata.get("name") != uploaded_file.name
            )

            if is_new_file:
                try:
                    content = uploaded_file.read()

                    # Save document (replaces any existing document)
                    doc_path = save_document(uploaded_file.name, content)

                    # Parse document to get text preview
                    try:
                        parsing_params = st.session_state.get("parsing_params", {})
                        parsed_text, file_format, _ = parse_document(
                            uploaded_file.name,
                            content,
                            parsing_params,
                        )
                        preview = get_file_preview(parsed_text)
                        char_count = len(parsed_text)
                    except Exception as parse_error:
                        preview = f"Error parsing document: {str(parse_error)}"
                        char_count = len(content)
                        file_format = Path(uploaded_file.name).suffix.upper().lstrip(".")

                    # Store metadata in session state (single document)
                    st.session_state.document_metadata = {
                        "name": uploaded_file.name,
                        "format": file_format,
                        "size_bytes": len(content),
                        "char_count": char_count,
                        "preview": preview,
                        "path": str(doc_path),
                    }

                    # Set as current document
                    st.session_state.doc_name = uploaded_file.name

                    # Invalidate chunks and embeddings for new file
                    for key in ["chunks", "last_embeddings_result", "search_results"]:
                        if key in st.session_state:
                            del st.session_state[key]

                    st.success(f"Uploaded: {uploaded_file.name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to upload {uploaded_file.name}: {str(e)}")

    # Load metadata for existing document if not in session state
    if current_doc and st.session_state.document_metadata is None:
        doc_name, content = current_doc
        try:
            parsing_params = st.session_state.get("parsing_params", {})
            parsed_text, file_format, _ = parse_document(doc_name, content, parsing_params)
            preview = get_file_preview(parsed_text)
            char_count = len(parsed_text)
        except Exception:
            preview = "Unable to parse document"
            char_count = len(content)
            file_format = Path(doc_name).suffix.upper().lstrip(".")

        st.session_state.document_metadata = {
            "name": doc_name,
            "format": file_format,
            "size_bytes": len(content),
            "char_count": char_count,
            "preview": preview,
            "path": "",
        }
        st.session_state.doc_name = doc_name

    # Display current document
    metadata = st.session_state.document_metadata
    if metadata:
        st.write("")
        st.markdown("### Current Document")

        with st.container(border=True):
            col1, col2 = st.columns([5, 1])

            with col1:
                file_format = metadata.get("format", "Unknown")
                size_bytes = metadata.get("size_bytes", 0)
                char_count = metadata.get("char_count", 0)

                # Format file size
                if size_bytes < 1024:
                    size_str = f"{size_bytes} B"
                elif size_bytes < 1024 * 1024:
                    size_str = f"{size_bytes / 1024:.1f} KB"
                else:
                    size_str = f"{size_bytes / (1024 * 1024):.1f} MB"

                st.markdown(f"**{html.escape(metadata.get('name', 'Unknown'))}**")
                st.caption(f"{html.escape(file_format)} | {size_str} | {char_count:,} characters")

            with col2:
                if ui.button("Delete", variant="destructive", key="delete_current_doc"):
                    clear_documents()
                    st.session_state.document_metadata = None
                    st.session_state.doc_name = None
                    for key in ["chunks", "last_embeddings_result", "search_results"]:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()

            # Document preview
            preview = metadata.get("preview", "")
            if preview:
                with st.expander("Preview Content", expanded=False):
                    st.markdown(f"<div style='color: #666; font-size: 0.9em;'>{html.escape(preview)}</div>", unsafe_allow_html=True)
    else:
        st.info("No document uploaded yet. Use the file uploader above to add a document.")
