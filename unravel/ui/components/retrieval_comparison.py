"""Pinned retrieval baseline controls for the query inspector."""

import streamlit as st

from unravel.services.retrieval.comparison import RetrievalSnapshot, compare_retrieval
from unravel.services.retrieval.export import serialize_visible_comparison

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
        change = st.selectbox(
            "Show changes",
            options=[
                "All changes", "Added", "Removed", "Retained",
                "Moved up", "Moved down", "Same rank",
            ],
            key="retrieval_comparison_change",
            help="Rank filters apply only to chunks present in both runs.",
        )
        min_movement = st.number_input(
            "Minimum rank movement", min_value=0, value=0, step=1,
            key="retrieval_comparison_min_movement",
            help="Inclusive absolute change in rank positions. Zero includes all changes; "
            "positive values include only chunks present in both runs. "
            "Filters this view and its text download; snapshots stay complete.",
        )
        search = st.text_input(
            "Find in comparison text",
            key="retrieval_comparison_search",
            help="Search full chunk text, including removed baseline context. "
            "Filters only this inspection view; snapshots and experiment downloads stay complete.",
        ).strip().casefold()
        order = st.radio(
            "Order comparison chunks",
            options=[
                "Current retrieval order", "Baseline rank", "Largest rank movement",
                "Shortest text first", "Longest text first",
            ],
            key="retrieval_comparison_order",
            horizontal=True,
            help="Length uses full chunk text, including whitespace and removed baseline text. "
            "Chunks without a baseline rank or rank change sort last in those modes. "
            "Ties keep their current retrieval order.",
        )
        visible = []
        for row in rows:
            if row["Current rank"] is not None:
                chunk = current.chunks[row["Current rank"] - 1]
            else:
                chunk = baseline.chunks[row["Baseline rank"] - 1]
            rank_change = row["Rank change"]
            if rank_change is None:
                movement = None
            elif rank_change > 0:
                movement = "Moved up"
            elif rank_change < 0:
                movement = "Moved down"
            else:
                movement = "Same rank"
            if (
                change in ("All changes", row["Change"], movement)
                and (not search or search in chunk.text.casefold())
                and (
                    min_movement == 0
                    or (rank_change is not None and abs(rank_change) >= min_movement)
                )
            ):
                visible.append((row, chunk))

        if order == "Baseline rank":
            visible.sort(key=lambda item: (
                item[0]["Baseline rank"] is None, item[0]["Baseline rank"] or 0
            ))
        elif order == "Largest rank movement":
            visible.sort(key=lambda item: (
                item[0]["Rank change"] is None, -abs(item[0]["Rank change"] or 0)
            ))
        elif order in ("Shortest text first", "Longest text first"):
            visible.sort(
                key=lambda item: len(item[1].text), reverse=order == "Longest text first"
            )

        st.caption(f"Showing {len(visible)} of {len(rows)} comparison chunks.")
        if visible:
            visible_rows = [row for row, _chunk in visible]
            st.dataframe(visible_rows, hide_index=True, use_container_width=True)
            document = (
                f"Baseline query: {baseline.query}\nCurrent query: {current.query}\n\n"
                + "\n\n---\n\n".join(
                    f"Chunk index {row['Chunk']} | {row['Change']}\n"
                    f"Baseline rank {row['Baseline rank'] or '—'} | "
                    f"Current rank {row['Current rank'] or '—'}\n\n{chunk.text}"
                    for row, chunk in visible
                )
            )
            st.download_button(
                label="Download visible comparison as JSON",
                data=serialize_visible_comparison(
                    baseline.query, current.query, visible,
                    {"change": change, "min_rank_movement": min_movement,
                     "search": search, "order": order},
                ),
                file_name="retrieval-comparison.json",
                mime="application/json",
                key="retrieval_comparison_json_download",
                help="Selected rows in this view's order, with full raw text, original ranks "
                "and scores. Includes removed baseline context and active filters.",
            )
            st.download_button(
                label="Download visible comparison as text",
                data=document,
                file_name="retrieval-comparison.txt",
                mime="text/plain; charset=utf-8",
                key="retrieval_comparison_text_download",
                help="Full raw text in this view's filtered order, with change types, "
                "original chunk indices and ranks. Includes removed baseline context.",
            )
            selected = st.selectbox(
                "Inspect full chunk text",
                options=range(len(visible)),
                format_func=lambda position: (
                    f"Chunk {visible_rows[position]['Chunk']} · {visible_rows[position]['Change']}"
                    f" · baseline rank {visible_rows[position]['Baseline rank'] or '—'}"
                    f" · current rank {visible_rows[position]['Current rank'] or '—'}"
                ),
                key="retrieval_comparison_inspect",
            )
            st.text(visible[selected][1].text)
        else:
            st.info(
                "No comparison chunks match. Clear the search, lower the minimum rank "
                "movement, or show all changes."
            )
        st.caption(
            "Positive rank changes mean a chunk moved up. Scores retain each run's scale; "
            "different retrievers or rerankers may produce incomparable scores. "
            "Chunk identity uses index and full text, "
            "so changed text appears as added and removed. "
            "The baseline lasts for this browser session."
        )
