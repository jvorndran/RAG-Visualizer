import hashlib
import json
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit_shadcn_ui as ui

from unravel.services.chunking import get_chunks
from unravel.services.storage import get_documents_dir, load_document, save_rag_config
from unravel.ui.components.chunk_viewer import (
    prepare_chunk_display_data,
    render_chunk_cards,
)
from unravel.ui.components.chunking_config import render_chunking_configuration
from unravel.ui.components.progress_bar import timed_progress_bar
from unravel.ui.components.scroll_utils import scroll_to_element
from unravel.ui.constants import WidgetKeys
from unravel.utils.cache import (
    get_parsed_text_key,
    load_parsed_text,
    save_parsed_text,
)
from unravel.utils.parsers import parse_document
from unravel.utils.ui import render_page_header

# Seconds per MB by file extension
_RATE_PER_MB: dict[str, float] = {
    ".pdf": 80.0,  # ~1 min for 0.8MB
    ".docx": 10.0,
    ".pptx": 10.0,
    ".xlsx": 10.0,
    ".html": 10.0,
    ".htm": 10.0,
    ".png": 20.0,
    ".jpg": 20.0,
    ".jpeg": 20.0,
    ".bmp": 20.0,
    ".tiff": 20.0,
    ".tif": 20.0,
    ".md": 2.0,
    ".markdown": 2.0,
    ".txt": 2.0,
}
_BASE_OVERHEAD_SECONDS = 10.0
_OCR_RATE_PER_MB = 150.0
_DEFAULT_CHUNKS_PER_PAGE = 20


def _get_chunks_state_key(
    selected_doc: str,
    source_text: str,
    output_format: str,
    chunking_params: dict[str, Any],
) -> str:
    """Build a stable key for the currently displayed chunks."""
    payload = {
        "doc": selected_doc,
        "source_hash": hashlib.md5(
            source_text.encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest(),
        "output_format": output_format,
        "chunking_params": chunking_params,
    }
    key_text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(key_text.encode("utf-8"), usedforsecurity=False).hexdigest()


def estimate_parsing_time(doc_path: Path, params: dict[str, Any]) -> float:
    """Estimate parsing duration based on file size, type, and OCR setting."""
    try:
        size_mb = doc_path.stat().st_size / (1024 * 1024)
    except (FileNotFoundError, OSError):
        # If file doesn't exist (e.g., in tests), use a default size estimate
        size_mb = 0.5  # Default to 0.5MB

    ext = doc_path.suffix.lower()

    if ext == ".pdf" and params.get("docling_enable_ocr"):
        rate = _OCR_RATE_PER_MB
    else:
        rate = _RATE_PER_MB.get(ext, 2.0)

    return _BASE_OVERHEAD_SECONDS + (size_mb * rate)


def render_chunks_step() -> None:
    import os

    # Check if demo mode is enabled
    is_demo_mode = os.getenv("DEMO_MODE") == "true"

    render_page_header(
        "Document processing & text splitting",
        "Configure parsing, text splitting, and chunk inspection for the selected document.",
        "Text splitting",
    )

    # Render chunking configuration

    with st.expander("Configuration", expanded=False):
        new_parsing_params, new_chunking_params, has_changes = render_chunking_configuration()

        if st.button(
            "Apply configuration",
            type="primary",
            disabled=not has_changes,
            key=WidgetKeys.CHUNKS_APPLY_BTN,
        ):
            # Update session state
            st.session_state.parsing_params = new_parsing_params
            st.session_state.chunking_params = new_chunking_params
            st.session_state.applied_parsing_params = new_parsing_params.copy()
            st.session_state.applied_chunking_params = new_chunking_params.copy()

            # Invalidate downstream caches
            for key in [
                "chunks",
                "chunks_cache_key",
                "chunk_display_cache_key",
                "chunk_display_data",
                "last_embeddings_result",
                "search_results",
                "bm25_index_data",
            ]:
                if key in st.session_state:
                    del st.session_state[key]

            # Persist to disk (skip in demo mode)
            if not is_demo_mode:
                save_rag_config(
                    {
                        "doc_name": st.session_state.doc_name,
                        "parsing_params": new_parsing_params,
                        "chunking_params": new_chunking_params,
                        "embedding_model_name": st.session_state.embedding_model_name,
                    }
                )

            st.success("Configuration applied successfully", icon=":material/check_circle:")
            st.rerun()

    # Read configuration from session state
    selected_doc = st.session_state.get("doc_name")
    chunking_params = st.session_state.get(
        "applied_chunking_params",
        st.session_state.get(
            "chunking_params",
            {
                "provider": "Docling",
                "splitter": "HybridChunker",
                "max_tokens": 512,
                "chunk_overlap": 50,
                "tokenizer": "cl100k_base",
            },
        ),
    )
    # Get current parsing params (from sidebar) and applied params (what was used)
    current_parsing_params = st.session_state.get("parsing_params", {})
    applied_parsing_params = st.session_state.get(
        "applied_parsing_params", current_parsing_params.copy()
    )

    provider = chunking_params.get("provider", "Docling")
    splitter = chunking_params.get("splitter", "HybridChunker")

    # Check if document is selected
    if not selected_doc:
        st.info(
            "No document selected. Upload a file in the Upload step or "
            "select a document in the sidebar.",
            icon=":material/info:",
        )
        if st.button("Go to upload", key=WidgetKeys.CHUNKS_GOTO_UPLOAD, type="primary"):
            st.session_state.current_step = "upload"
            st.rerun(scope="app")
        return

    # Check if parsing settings have changed (compare current vs applied)
    parsing_settings_changed = current_parsing_params != applied_parsing_params

    # Always use applied params for lookup (what was actually used for the current parsed text)
    # This ensures we show the existing parsed document even after "Save & Apply"
    params_for_lookup = applied_parsing_params
    parsed_text_key = get_parsed_text_key(selected_doc, params_for_lookup)

    # Try to load parsed text from session state first, then from persistent storage
    # Use applied params to get the currently displayed parsed text
    source_text = st.session_state.get(parsed_text_key, "")
    if not source_text:
        # Try loading from persistent storage using applied params
        source_text = load_parsed_text(selected_doc, params_for_lookup) or ""
        if source_text:
            # Restore to session state
            st.session_state[parsed_text_key] = source_text

    # Show parse button if no parsed text OR if parsing settings have changed
    needs_parsing = not source_text or parsing_settings_changed
    if needs_parsing:

        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            button_text = (
                "Reparse Document" if parsing_settings_changed and source_text else "Parse Document"
            )
            if st.button(button_text, key=WidgetKeys.CHUNKS_PARSE_BTN, type="primary"):
                try:
                    doc_path = get_documents_dir() / selected_doc
                    estimated_time = estimate_parsing_time(doc_path, current_parsing_params)

                    def parse_task(doc_name, params):
                        content = load_document(doc_name)
                        if not content:
                            return None
                        return parse_document(doc_name, content, params)

                    result = timed_progress_bar(
                        task=parse_task,
                        args=(selected_doc, current_parsing_params),
                        label="Parsing document...",
                        estimated_time=estimated_time,
                    )

                    if result:
                        parsed_text, _, _ = result
                        # Update applied params to match current
                        # (so they're in sync after reparse)
                        st.session_state["applied_parsing_params"] = current_parsing_params.copy()
                        new_applied_params = current_parsing_params.copy()
                        new_parsed_text_key = get_parsed_text_key(selected_doc, new_applied_params)
                        # Cache parsed text in session state
                        st.session_state[new_parsed_text_key] = parsed_text
                        # Save to persistent storage (skip in demo mode)
                        if not is_demo_mode:
                            save_parsed_text(selected_doc, new_applied_params, parsed_text)
                        # Set flag to trigger scroll after rerun
                        st.session_state["trigger_scroll_to_chunks"] = True
                        st.rerun()
                    else:
                        st.error(f"Failed to load document: {selected_doc}")
                except Exception as e:
                    st.error(f"Error parsing document: {str(e)}")

        if parsing_settings_changed and source_text:
            st.info(
                "Parsing settings changed. Reparse the document to use the new settings.",
                icon=":material/info:",
            )
        else:
            st.info(
                "Parse the document to generate chunks.",
                icon=":material/info:",
            )

        # If we have no parsed text, return early until the user parses.
        # Otherwise, continue to show the existing parsed text even if settings changed.
        if not source_text:
            return

    # Generate Chunks (only if we have parsed text)
    output_format = applied_parsing_params.get("output_format", "markdown")
    if source_text:
        # Extract splitter-specific params (exclude provider/splitter keys)
        splitter_params = {
            k: v for k, v in chunking_params.items() if k not in ["provider", "splitter"]
        }
        chunk_state_key = _get_chunks_state_key(
            selected_doc,
            source_text,
            output_format,
            chunking_params,
        )
        cached_chunk_key = st.session_state.get("chunks_cache_key")
        cached_chunks = st.session_state.get("chunks")

        if cached_chunk_key == chunk_state_key and cached_chunks:
            chunks = cached_chunks
        else:
            chunks = get_chunks(
                provider=provider,
                splitter=splitter,
                text=source_text,
                output_format=output_format,
                **splitter_params,
            )
            st.session_state["chunks_cache_key"] = chunk_state_key

        # Save chunks to session state (config already set in sidebar)
        st.session_state["chunks"] = chunks
    else:
        chunks = []
        st.session_state["chunks"] = []

    # Main Visualization

    # Add anchor element for auto-scroll
    st.markdown('<div id="chunks-visualization-section"></div>', unsafe_allow_html=True)

    # Trigger scroll if flag is set (after parsing completes)
    if st.session_state.get("trigger_scroll_to_chunks", False):
        scroll_to_element("chunks-visualization-section")
        st.session_state["trigger_scroll_to_chunks"] = False

    # Prepare chunk display data (includes overlap calculation)
    display_cache_key = f"{st.session_state.get('chunks_cache_key', '')}:display"
    if st.session_state.get("chunk_display_cache_key") == display_cache_key:
        chunk_display_data = st.session_state.get("chunk_display_data", [])
    else:
        chunk_display_data = prepare_chunk_display_data(
            chunks=chunks,
            source_text=source_text,
            calculate_overlap=True,
        )
        st.session_state["chunk_display_cache_key"] = display_cache_key
        st.session_state["chunk_display_data"] = chunk_display_data

    view_mode = ui.tabs(
        options=["Visual View", "Raw JSON"],
        default_value="Visual View",
        key=WidgetKeys.CHUNKS_VIEW_TAB,
    )

    # Render based on selected view

    if view_mode == "Visual View":
        with st.container(border=True):
            total_chunks = len(chunk_display_data)
            if total_chunks > _DEFAULT_CHUNKS_PER_PAGE:
                col_size, col_page = st.columns([1, 2], vertical_alignment="center")
                with col_size:
                    page_size = st.selectbox(
                        "Chunks per page",
                        options=[10, 20, 50, 100],
                        index=1,
                        key="chunks_page_size",
                    )
                total_pages = max(1, (total_chunks + page_size - 1) // page_size)
                with col_page:
                    st.session_state["chunks_page"] = max(
                        1,
                        min(st.session_state.get("chunks_page", 1), total_pages),
                    )
                    page = st.number_input(
                        "Page",
                        min_value=1,
                        max_value=total_pages,
                        key="chunks_page",
                    )
                start = (page - 1) * page_size
                end = start + page_size
                st.caption(f"Showing chunks {start + 1}-{min(end, total_chunks)} of {total_chunks}")
                visible_chunk_data = chunk_display_data[start:end]
            else:
                visible_chunk_data = chunk_display_data

            render_chunk_cards(
                chunk_display_data=visible_chunk_data,
                show_overlap=True,
                display_mode="continuous",
                render_format=output_format,
            )
    else:  # Raw JSON
        # Convert chunks to serializable format for JSON display
        chunks_json = []
        for chunk in chunks:
            chunk_dict = {
                "text": chunk.text,
                "start_index": chunk.start_index,
                "end_index": chunk.end_index,
                "metadata": chunk.metadata,
            }
            chunks_json.append(chunk_dict)

        with st.container(border=True):
            st.json(chunks_json, expanded=True)
