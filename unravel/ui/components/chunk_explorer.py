"""Local inspection of generated chunk text without changing the pipeline."""

import hashlib
import json
from collections import Counter
from typing import Any

import streamlit as st


def chunk_element_types(chunk: Any) -> set[str]:
    """Read available element labels, including chunks containing multiple types."""
    value = getattr(chunk, "metadata", {}).get("element_type")
    if isinstance(value, str):
        return {value} if value else set()
    if isinstance(value, (list, tuple, set)):
        return {label for label in value if isinstance(label, str) and label}
    return set()


def chunk_sections(chunk: Any) -> set[str]:
    """Read reported section headings, with legacy Markdown headers as a fallback."""
    metadata = getattr(chunk, "metadata", {})
    value = metadata.get("section_hierarchy", [metadata.get(f"Header {i}") for i in range(1, 7)])
    if isinstance(value, str):
        return {value} if value else set()
    if isinstance(value, (list, tuple, set)):
        return {heading for heading in value if isinstance(heading, str) and heading}
    return set()


def matching_chunk_indices(
    chunks: list[Any], query: str, min_characters: int = 0, max_characters: int | None = None,
    repeated_text_only: bool = False,
    sort_order: str = "Original order",
    element_types: list[str] | None = None,
    sections: list[str] | None = None,
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
        and (not element_types or chunk_element_types(chunk).intersection(element_types))
        and (not sections or chunk_sections(chunk).intersection(sections))
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
    element_types: list[str] | None = None,
    sections: list[str] | None = None,
) -> str:
    """Export matching text and offsets; exclude arbitrary metadata and configuration."""
    indices = matching_chunk_indices(
        chunks, query, min_characters, max_characters, repeated_text_only, sort_order,
        element_types, sections
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
        "element_types": element_types or [],
        "sections": sections or [],
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
        element_types = st.multiselect(
            "Element types",
            options=sorted({label for chunk in chunks for label in chunk_element_types(chunk)}),
            key=f"chunk_explorer_element_types_{state_key}",
            help=(
                "Match any selected metadata label, such as table or code. "
                "Leave empty to include all chunks, including those without element metadata."
            ),
        )
        sections = st.multiselect(
            "Sections",
            options=sorted({heading for chunk in chunks for heading in chunk_sections(chunk)}),
            key=f"chunk_explorer_sections_{state_key}",
            help=(
                "Match any selected heading in the reported section hierarchy or legacy "
                "Markdown headers. Leave empty to include chunks without section metadata."
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
            or prepared.get("element_types", []) != element_types
            or prepared.get("sections", []) != sections
        ):
            st.session_state.pop(export_key)
            prepared = None
        if max_characters is not None and min_characters > max_characters:
            st.error("Minimum characters must not exceed maximum characters.")
            return
        indices = matching_chunk_indices(
            chunks, query, min_characters, max_characters, repeated_text_only, sort_order,
            element_types, sections
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
                "element_types": element_types,
                "sections": sections,
                "data": serialize_chunk_selection(
                    chunks, query, document_name, min_characters, max_characters,
                    repeated_text_only, sort_order, element_types, sections
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
            query, min_characters, max_characters, repeated_text_only, sort_order,
            element_types, sections
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
