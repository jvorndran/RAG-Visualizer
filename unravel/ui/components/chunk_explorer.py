"""Local inspection of generated chunk text without changing the pipeline."""

import hashlib
import json
from collections import Counter
from typing import Any

import streamlit as st


def matching_chunk_indices(
    chunks: list[Any], query: str, min_characters: int = 0, max_characters: int | None = None,
    repeated_text_only: bool = False,
    sort_order: str = "Original order",
) -> list[int]:
    """Return original indices matching text, size, and optional exact-repeat filters."""
    needle = query.strip().casefold()
    text_counts = Counter(chunk.text for chunk in chunks) if repeated_text_only else None
    indices = [
        index for index, chunk in enumerate(chunks)
        if len(chunk.text) >= min_characters
        and (max_characters is None or len(chunk.text) <= max_characters)
        and needle in chunk.text.casefold()
        and (text_counts is None or text_counts[chunk.text] > 1)
    ]
    if sort_order == "Shortest first":
        indices.sort(key=lambda index: len(chunks[index].text))
    elif sort_order == "Longest first":
        indices.sort(key=lambda index: -len(chunks[index].text))
    return indices


def serialize_chunk_selection(
    chunks: list[Any], query: str, document_name: str | None,
    min_characters: int = 0, max_characters: int | None = None,
    repeated_text_only: bool = False,
    sort_order: str = "Original order",
) -> str:
    """Export matching text and offsets; exclude arbitrary metadata and configuration."""
    indices = matching_chunk_indices(
        chunks, query, min_characters, max_characters, repeated_text_only, sort_order
    )
    payload = {
        "schema_version": 1,
        "document": document_name,
        "search_text": query.strip(),
        "case_sensitive": False,
        "min_characters": min_characters,
        "max_characters": max_characters,
        "repeated_text_only": repeated_text_only,
        "sort_order": sort_order,
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
        minimum_column, maximum_column = st.columns(2)
        with minimum_column:
            min_characters = st.number_input(
                "Minimum characters", min_value=0, value=0, step=1,
                key=f"chunk_explorer_min_characters_{state_key}",
                help="Inclusive raw-text characters, including whitespace; not a token count.",
            ) or 0
        with maximum_column:
            max_characters = st.number_input(
                "Maximum characters", min_value=0, value=None, step=1,
                placeholder="No limit", key=f"chunk_explorer_max_characters_{state_key}",
                help="Inclusive raw-text character count. Leave empty for no upper bound.",
            )
        repeated_text_only = st.checkbox(
            "Repeated text only",
            key=f"chunk_explorer_repeated_text_{state_key}",
            help=(
                "Include every occurrence of text found in multiple generated chunks. "
                "Raw text must match exactly, including case, whitespace, and empty text."
            ),
        )
        sort_order = st.radio(
            "Order matching chunks",
            options=["Original order", "Shortest first", "Longest first"],
            horizontal=True,
            key=f"chunk_explorer_sort_{state_key}",
            help="Sort by raw character count; ties retain original chunk order.",
        )
        export_key = "chunk_explorer_prepared_export"
        prepared = st.session_state.get(export_key)
        if prepared and (
            prepared["generation"] != state_key or prepared["query"] != query
            or prepared.get("min_characters") != min_characters
            or prepared.get("max_characters") != max_characters
            or prepared.get("repeated_text_only", False) != repeated_text_only
            or prepared.get("sort_order", "Original order") != sort_order
        ):
            st.session_state.pop(export_key)
            prepared = None
        if max_characters is not None and min_characters > max_characters:
            st.error("Minimum characters must not exceed maximum characters.")
            return
        indices = matching_chunk_indices(
            chunks, query, min_characters, max_characters, repeated_text_only, sort_order
        )
        st.caption(
            f"{len(indices)} of {len(chunks)} chunks match. Original chunk numbers are retained."
        )
        if not prepared and st.button(
            "Prepare JSON download",
            key=f"chunk_explorer_prepare_{state_key}",
            disabled=not indices,
        ):
            prepared = {
                "generation": state_key,
                "query": query,
                "min_characters": min_characters,
                "max_characters": max_characters,
                "repeated_text_only": repeated_text_only,
                "sort_order": sort_order,
                "data": serialize_chunk_selection(
                    chunks, query, document_name, min_characters, max_characters,
                    repeated_text_only, sort_order
                ),
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
            st.info("No matching chunks. Clear or change the filters to inspect other chunks.")
            return

        selection = json.dumps([
            query, min_characters, max_characters, repeated_text_only, sort_order
        ])
        query_key = hashlib.sha256(selection.encode("utf-8")).hexdigest()[:16]
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
