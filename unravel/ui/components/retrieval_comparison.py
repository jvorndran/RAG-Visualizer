"""Pinned retrieval baseline controls for the query inspector."""

import streamlit as st

from unravel.services.retrieval.comparison import RetrievalSnapshot, compare_retrieval

BASELINE_KEY = "retrieval_comparison_baseline"


def render_retrieval_comparison(current: RetrievalSnapshot) -> None:
    """Compare already retrieved context without rerunning retrieval or generation."""
    if not current.chunks and BASELINE_KEY not in st.session_state:
        return
    with st.expander("Compare retrieval runs"):
        pin_column, clear_column = st.columns(2)
        with pin_column:
            if st.button(
                "Pin current run as baseline",
                key="retrieval_comparison_pin",
                disabled=not current.chunks,
            ):
                st.session_state[BASELINE_KEY] = current
        with clear_column:
            if st.button(
                "Clear baseline",
                key="retrieval_comparison_clear",
                disabled=BASELINE_KEY not in st.session_state,
            ):
                st.session_state.pop(BASELINE_KEY, None)

        baseline = st.session_state.get(BASELINE_KEY)
        if baseline is None:
            st.caption(
                "Pin a run, then change retrieval settings or the query "
                "and click Ask to compare context."
            )
            return

        st.caption(f"Baseline query: {baseline.query}")
        st.caption(f"Current query: {current.query}")
        st.caption(
            f"Baseline: {baseline.strategy}, top k {baseline.top_k}, "
            f"minimum score {baseline.threshold}, "
            f"reranker {baseline.reranker or 'off'}. "
            f"Current: {current.strategy}, top k {current.top_k}, "
            f"minimum score {current.threshold}, "
            f"reranker {current.reranker or 'off'}."
        )
        rows = compare_retrieval(baseline, current)
        retained = sum(row["Change"] == "Retained" for row in rows)
        added = sum(row["Change"] == "Added" for row in rows)
        removed = sum(row["Change"] == "Removed" for row in rows)
        st.caption(f"{retained} retained · {added} added · {removed} removed")
        st.dataframe(rows, hide_index=True, use_container_width=True)
        st.caption(
            "Positive rank changes mean a chunk moved up. Scores retain each run's scale; "
            "different retrievers or rerankers may produce incomparable scores. "
            "Chunk identity uses index and full text, "
            "so changed text appears as added and removed. "
            "The baseline lasts for this browser session."
        )
