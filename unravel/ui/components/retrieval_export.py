"""Download retrieval evidence without issuing another provider request."""

from typing import Any

import streamlit as st

from unravel.services.retrieval.comparison import RetrievalSnapshot
from unravel.services.retrieval.export import (
    serialize_retrieval_experiment,
    serialize_visible_context,
)
from unravel.ui.components.retrieval_comparison import BASELINE_KEY


def render_context_text_export(ranked_results: list[tuple[int, Any]]) -> None:
    """Download the visible selection with raw text and original retrieval identities."""
    if not ranked_results:
        return
    document = "\n\n".join(
        f"Rank {rank} | Chunk index {result.index}\n{result.text}"
        for rank, result in ranked_results
    ) + "\n"
    st.download_button(
        label="Download visible context as text",
        data=document,
        file_name="retrieved-context.txt",
        mime="text/plain",
        key="retrieved_context_text_download",
        help="Full raw text of the cards matching the search and rank window, "
        "with original ranks and chunk indices. Excludes scores and arbitrary metadata.",
    )


def render_context_json_export(
    query: str, ranked_results: list[tuple[int, Any]], filters: dict[str, Any]
) -> None:
    """Download the visible selection with allowlisted scores and inspection filters."""
    if not ranked_results:
        return
    st.download_button(
        label="Download visible context as JSON",
        data=serialize_visible_context(query, ranked_results, filters),
        file_name="retrieved-context.json",
        mime="application/json",
        key="retrieved_context_json_download",
        help="Selected cards in displayed order, with full raw text, original ranks, "
        "chunk indices, scores, executed query and active inspection filters. "
        "Non-finite scores are exported as null.",
    )


def render_retrieval_export(current: RetrievalSnapshot) -> None:
    """Use the executed run, including an empty result and any pinned baseline."""
    if not current.query and not current.chunks:
        return
    st.download_button(
        label="Download retrieval experiment",
        data=serialize_retrieval_experiment(current, st.session_state.get(BASELINE_KEY)),
        file_name="retrieval-experiment.json",
        mime="application/json",
        key="retrieval_experiment_download",
        help="Includes full retrieved text, executed settings and any pinned baseline. "
        "Non-finite measurements are exported as null.",
    )
