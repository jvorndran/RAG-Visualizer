"""Whole-chunk rank prefixes within a raw Unicode character budget."""

import pytest

from unravel.services.retrieval.comparison import RankedChunk, RetrievalSnapshot, context_budget


def snapshot(texts):
    return RetrievalSnapshot(
        "query", "DenseRetriever", None, None, None,
        tuple(RankedChunk(i, text, 0.1, i + 1) for i, text in enumerate(texts)),
    )


@pytest.mark.parametrize("budget,count,used,next_size", [
    (0, 1, 0, 3), (2, 1, 0, 3), (3, 2, 3, 6), (9, 4, 9, None),
])
def test_budget_keeps_prefix_including_empty_chunks(budget, count, used, next_size):
    run = snapshot(["", "é\n🙂", "123456", ""])
    assert context_budget(run, budget) == {
        "included_chunks": count, "excluded_chunks": 4 - count,
        "used_characters": used, "remaining_characters": budget - used,
        "next_chunk_characters": next_size,
    }
    assert run == snapshot(["", "é\n🙂", "123456", ""])


def test_budget_never_skips_a_large_chunk_or_truncates_text():
    assert context_budget(snapshot(["large", "x"]), 1)["included_chunks"] == 0
    assert context_budget(snapshot([]), 12) == {
        "included_chunks": 0, "excluded_chunks": 0, "used_characters": 0,
        "remaining_characters": 12, "next_chunk_characters": None,
    }
    assert context_budget(snapshot(["ab", "ab"]), 4)["used_characters"] == 4


@pytest.mark.parametrize("budget", [-1, 1.5, True, "2", None])
def test_invalid_character_budget_fails(budget):
    with pytest.raises(ValueError, match="non-negative integer"):
        context_budget(snapshot([]), budget)
