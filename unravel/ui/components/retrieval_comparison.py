"""Pinned retrieval baseline controls for the query inspector."""

import streamlit as st

from unravel.services.retrieval.comparison import (
    RetrievalSnapshot,
    compare_retrieval,
    context_budget,
    context_redundancy,
    context_size_profile,
    phrase_coverage,
    rank_overlap_profile,
)
from unravel.services.retrieval.export import (
    serialize_context_budget,
    serialize_context_redundancy,
    serialize_context_size_profile,
    serialize_rank_overlap_profile,
    serialize_visible_comparison,
)

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
        if st.checkbox(
            "Show phrase coverage comparison",
            key="retrieval_comparison_phrases",
            help="Count chunks containing each phrase across both complete retrieval runs.",
        ):
            phrases = st.text_area(
                "Phrases to compare (one per line)",
                key="retrieval_comparison_phrase_text",
                max_chars=4000,
                help="Literal substrings, ignoring case. Blank lines and duplicate phrases "
                "are ignored; the first spelling is displayed.",
            )
            st.caption(
                "Each matching chunk counts once per phrase, even if the phrase repeats. "
                "Matches stay within a chunk and preserve punctuation and Unicode form. "
                "Chunk change is current minus baseline. Both complete snapshots are used, "
                "independent of the visible filters. This measures literal text presence, "
                "not semantic coverage or answer quality."
            )
            coverage = phrase_coverage(baseline, current, phrases)
            if coverage:
                st.dataframe(coverage, hide_index=True, use_container_width=True)
        rows = compare_retrieval(baseline, current)
        retained = sum(row["Change"] == "Retained" for row in rows)
        added = sum(row["Change"] == "Added" for row in rows)
        removed = sum(row["Change"] == "Removed" for row in rows)
        st.caption(f"{retained} retained · {added} added · {removed} removed")
        if st.checkbox(
            "Show context redundancy comparison",
            key="retrieval_comparison_redundancy",
            help="Measure exact raw text repeated across chunk indices in both complete runs.",
        ):
            st.caption(
                "Repeated chunks are copies after the first occurrence of identical full text, "
                "regardless of chunk index or score. Counts preserve case, whitespace and Unicode "
                "form. Repeated characters count only those excess copies. Partial overlaps and "
                "semantic similarity are not measured. Both complete snapshots are used; "
                "comparison filters, retrieval and generated answers stay unchanged."
            )
            redundancy_rows = []
            for label, run in (("Baseline", baseline), ("Current", current)):
                result = context_redundancy(run)
                redundancy_rows.append({
                    "Run": label,
                    "Chunks": result["chunks"],
                    "Unique texts": result["unique_texts"],
                    "Repeated chunks": result["repeated_chunks"],
                    "Total characters": result["total_characters"],
                    "Repeated characters": result["repeated_characters"],
                })
            st.dataframe(redundancy_rows, hide_index=True, use_container_width=True)
            st.download_button(
                label="Download context redundancy as JSON",
                data=serialize_context_redundancy(baseline, current),
                file_name="retrieval-context-redundancy.json",
                mime="application/json",
                key="retrieval_comparison_redundancy_download",
                help="Complete-run repetition measurements with query labels and strategies. "
                "Excludes raw context and scores; independent of comparison filters.",
            )
        if st.checkbox(
            "Show context budget comparison",
            key="retrieval_comparison_budget",
            help="Simulate a whole-chunk rank prefix within a raw character budget.",
        ):
            budget = st.number_input(
                "Character budget", min_value=0, value=1000, step=1,
                key="retrieval_comparison_budget_chars",
            )
            st.caption(
                "Includes whole chunks in retrieval order, stopping at the first chunk "
                "that would exceed the budget. Zero permits only leading empty chunks. "
                "Counts raw Unicode characters, including whitespace; excludes separators "
                "and prompt instructions. This is not a token limit. Both complete runs "
                "are used regardless of comparison filters. Retrieval and answers stay unchanged."
            )
            budget_rows = []
            for label, run in (("Baseline", baseline), ("Current", current)):
                result = context_budget(run, budget)
                budget_rows.append({
                    "Run": label,
                    "Included chunks": result["included_chunks"],
                    "Excluded chunks": result["excluded_chunks"],
                    "Used characters": result["used_characters"],
                    "Remaining characters": result["remaining_characters"],
                    "Next chunk characters": result["next_chunk_characters"],
                })
            st.dataframe(budget_rows, hide_index=True, use_container_width=True)
            st.download_button(
                label="Download context budget as JSON",
                data=serialize_context_budget(baseline, current, budget),
                file_name="retrieval-context-budget.json",
                mime="application/json",
                key="retrieval_comparison_budget_download",
                help="Complete-run budget measurements with both query labels and strategies. "
                "Excludes chunk text and scores; independent of comparison filters.",
            )
        if st.checkbox(
            "Show context size profile",
            key="retrieval_comparison_context_size",
            help="Compare cumulative raw chunk text at each rank depth in both complete runs.",
        ):
            st.caption(
                "Sizes count raw Unicode characters and UTF-8 bytes, including whitespace. "
                "Repeated text in different chunks counts each time. A shorter run contributes "
                "its available context. Character change is current minus baseline. "
                "These are text sizes, not token counts or complete prompt sizes; separators "
                "and prompt instructions are excluded. "
                "Comparison filters do not affect this profile."
            )
            st.dataframe(
                context_size_profile(baseline, current),
                hide_index=True, use_container_width=True,
            )
            st.download_button(
                label="Download context size profile as JSON",
                data=serialize_context_size_profile(baseline, current),
                file_name="retrieval-context-size.json",
                mime="application/json",
                key="retrieval_comparison_context_size_download",
                help="Full-run size measurements with both query labels and retrieval strategies.",
            )
        if st.checkbox(
            "Show rank overlap profile",
            key="retrieval_comparison_overlap",
            help="Compare the first N chunks from each complete run at every depth. "
            "Independent of the comparison filters below.",
        ):
            st.caption(
                "Overlap is shared chunks divided by the union at each depth (Jaccard). "
                "A shorter run contributes all its available chunks. Identity uses index "
                "and full text; scores do not affect overlap."
            )
            st.dataframe(
                rank_overlap_profile(baseline, current),
                hide_index=True, use_container_width=True,
            )
            st.download_button(
                label="Download rank overlap profile as JSON",
                data=serialize_rank_overlap_profile(baseline, current),
                file_name="retrieval-rank-overlap.json",
                mime="application/json",
                key="retrieval_comparison_overlap_download",
                help="Complete rank-prefix overlap with query labels and retrieval strategies. "
                "Independent of comparison filters; excludes chunk text and scores.",
            )
        change = st.selectbox(
            "Show changes",
            options=[
                "All changes", "Added", "Removed", "Retained",
                "Moved up", "Moved down", "Same rank",
            ],
            key="retrieval_comparison_change",
            help="Rank filters apply only to chunks present in both runs.",
        )
        top_ranks = st.number_input(
            "Top ranks in either run", min_value=0, value=0, step=1,
            key="retrieval_comparison_top_ranks",
            help="Keep chunks ranked at or above this position in either run. "
            "Zero includes every rank. Applies to the table, inspector and visible downloads; "
            "full snapshots and overall change counts stay complete.",
        )
        min_movement = st.number_input(
            "Minimum rank movement", min_value=0, value=0, step=1,
            key="retrieval_comparison_min_movement",
            help="Inclusive absolute change in rank positions. Zero includes all changes; "
            "positive values include only chunks present in both runs. "
            "Filters this view and its text download; snapshots stay complete.",
        )
        length_columns = st.columns(2)
        with length_columns[0]:
            min_chars = st.number_input(
                "Minimum characters", min_value=0, value=0, step=1,
                key="retrieval_comparison_min_chars",
                help="Inclusive raw Unicode character count, including whitespace. "
                "Applies to visible rows, inspection and comparison downloads.",
            )
        with length_columns[1]:
            max_chars = st.number_input(
                "Maximum characters", min_value=0, value=0, step=1,
                key="retrieval_comparison_max_chars",
                help="Inclusive raw Unicode character count. Zero has no upper bound. "
                "Removed chunks use their full baseline text; snapshots stay complete.",
            )
        if max_chars and min_chars > max_chars:
            st.warning(
                "Maximum characters must be at least Minimum characters, or zero for no limit."
            )
        search = st.text_input(
            "Find in comparison text",
            key="retrieval_comparison_search",
            help="Search full chunk text, including removed baseline context. "
            "Filters only this inspection view; snapshots and experiment downloads stay complete.",
        ).strip().casefold()
        exclude_text = st.text_input(
            "Exclude comparison text",
            key="retrieval_comparison_exclude",
            help="Hide chunks containing this literal text, ignoring case. "
            "Includes removed baseline context; combines with Find in comparison text. "
            "Applies to inspection and visible downloads only.",
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
                and (not exclude_text or exclude_text not in chunk.text.casefold())
                and len(chunk.text) >= min_chars
                and (max_chars == 0 or len(chunk.text) <= max_chars)
                and (
                    top_ranks == 0
                    or any(
                        rank is not None and rank <= top_ranks
                        for rank in (row["Baseline rank"], row["Current rank"])
                    )
                )
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
                    {"change": change, "top_ranks": top_ranks, "min_rank_movement": min_movement,
                     "search": search, "exclude_text": exclude_text, "order": order,
                     "min_characters": min_chars, "max_characters": max_chars},
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
                "No comparison chunks match. Clear the search and exclusion, "
                "widen the character bounds, "
                "lower the minimum rank movement, increase the top-rank window "
                "(zero shows all ranks), "
                "or show all changes."
            )
        st.caption(
            "Positive rank changes mean a chunk moved up. Scores retain each run's scale; "
            "different retrievers or rerankers may produce incomparable scores. "
            "Chunk identity uses index and full text, "
            "so changed text appears as added and removed. "
            "The baseline lasts for this browser session."
        )
