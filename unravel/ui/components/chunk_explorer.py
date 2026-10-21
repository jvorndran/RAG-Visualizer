"""Local inspection of generated chunk text without changing the pipeline."""

import hashlib
import json
from typing import Any

import streamlit as st


def matching_chunk_indices(chunks: list[Any], query: str) -> list[int]:
    """Return original indices for a literal case-insensitive text search."""
    needle = query.strip().casefold()
    return [index for index, chunk in enumerate(chunks) if needle in chunk.text.casefold()]


def serialize_chunk_selection(
    chunks: list[Any], query: str, document_name: str | None
) -> str:
    """Export matching text and offsets; exclude arbitrary metadata and configuration."""
    indices = matching_chunk_indices(chunks, query)
    payload = {
        "schema_version": 1,
        "document": document_name,
        "search_text": query.strip(),
        "case_sensitive": False,
        "total_chunks": len(chunks),
        "count": len(indices),
        "chunks": [
            {
                "chunk_number": index + 1,
                "text": chunks[index].text,
                "start_index": chunks[index].start_index,
                "end_index": chunks[index].end_index,
            }
            for index in indices
        ],
    }
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n"


def render_chunk_explorer(
    chunks: list[Any], state_key: str, document_name: str | None = None
) -> None:
    """Inspect complete raw text, source offsets, and metadata for matching chunks."""
    with st.expander("Explore chunk text", expanded=False):
        query = st.text_input(
            "Search chunk text",
            key=f"chunk_explorer_search_{state_key}",
            help="Case-insensitive literal substring search across the generated raw chunk text.",
        )
        indices = matching_chunk_indices(chunks, query)
        st.caption(
            f"{len(indices)} of {len(chunks)} chunks match. Original chunk numbers are retained."
        )
        export_key = "chunk_explorer_prepared_export"
        prepared = st.session_state.get(export_key)
        if prepared and (prepared["generation"] != state_key or prepared["query"] != query):
            st.session_state.pop(export_key)
            prepared = None
        if not prepared and st.button(
            "Prepare JSON download",
            key=f"chunk_explorer_prepare_{state_key}",
            disabled=not indices,
        ):
            prepared = {
                "generation": state_key,
                "query": query,
                "data": serialize_chunk_selection(chunks, query, document_name),
            }
            st.session_state[export_key] = prepared
        if prepared:
            st.download_button(
                "Download matching chunks",
                data=prepared["data"],
                file_name="chunks.json",
                mime="application/json",
                key=f"chunk_explorer_download_{state_key}",
                help="Export full raw text, original chunk numbers, and source offsets as JSON.",
            )
        if not indices:
            st.info("No matching chunks. Clear or change the search to inspect other chunks.")
            return

        query_key = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
        selected = st.selectbox(
            "Inspect chunk",
            options=indices,
            format_func=lambda index: f"Chunk {index + 1} · {len(chunks[index].text):,} characters",
            key=f"chunk_explorer_selection_{state_key}_{query_key}",
        )
        chunk = chunks[selected]
        st.caption(f"Source offsets: {chunk.start_index}–{chunk.end_index}")
        st.code(chunk.text, language=None)
        st.json(chunk.metadata, expanded=False)
