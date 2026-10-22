"""Size diagnostics and a raw-text inspector for generated chunks."""

from collections.abc import Sequence
from statistics import median

import streamlit as st

from unravel.services.chunk_diagnostics import get_chunk_size_rows
from unravel.services.chunking import Chunk


def render_chunk_diagnostics(chunks: Sequence[Chunk]) -> None:
    with st.expander("Chunk diagnostics", expanded=False):
        if not chunks:
            st.info("No chunks to inspect.")
            return

        rows = get_chunk_size_rows(chunks)
        sizes = [row["characters"] for row in rows]
        count, middle, largest, duplicates = st.columns(4)
        count.metric("Chunks", len(rows))
        middle.metric("Median characters", f"{median(sizes):g}")
        largest.metric("Largest chunk", max(sizes))
        duplicates.metric("Exact repeats", sum(row["duplicate_of"] is not None for row in rows))
        st.caption(
            f"{sizes.count(0)} empty chunks. Character counts include whitespace and HTML markup; "
            "they are not token counts. Repeats match raw text exactly, including empty text. "
            "Source indexes are the positions reported by the chunker."
        )

        mode = st.selectbox(
            "Inspect by character count",
            ["All", "At least", "At most"],
            key="chunk_diagnostics_size",
        )
        visible_rows = rows
        if mode != "All":
            threshold = st.number_input(
                "Character threshold", min_value=0, value=100, step=1,
                key="chunk_diagnostics_threshold",
            )
            visible_rows = [
                row for row in rows
                if (row["characters"] >= threshold if mode == "At least"
                    else row["characters"] <= threshold)
            ]

        if not visible_rows:
            st.info("No chunks match this character threshold.")
            return

        st.dataframe(visible_rows, hide_index=True)
        indices = [row["chunk"] - 1 for row in visible_rows]
        selection_key = "chunk_diagnostics_chunk"
        if st.session_state.get(selection_key) not in indices:
            st.session_state[selection_key] = indices[0]
        index = st.selectbox(
            "Inspect a chunk",
            indices,
            format_func=lambda i: f"Chunk {i + 1} · {sizes[i]} characters",
            key=selection_key,
        )
        st.code(chunks[index].text, language=None)
        st.json(chunks[index].metadata, expanded=False)
