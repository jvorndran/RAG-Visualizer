"""Local inspection of generated chunk text without changing the pipeline."""

import hashlib
from typing import Any

import streamlit as st


def matching_chunk_indices(chunks: list[Any], query: str) -> list[int]:
    """Return original indices for a literal case-insensitive text search."""
    needle = query.strip().casefold()
    return [index for index, chunk in enumerate(chunks) if needle in chunk.text.casefold()]


def render_chunk_explorer(chunks: list[Any], state_key: str) -> None:
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
