"""Exact raw text repetition is distinct from index-based chunk identity."""

from unravel.services.retrieval.comparison import (
    RankedChunk,
    RetrievalSnapshot,
    context_redundancy,
)


def snapshot(texts):
    return RetrievalSnapshot("q", "DenseRetriever", None, None, None, tuple(
        RankedChunk(i, text, 0.1, i + 1) for i, text in enumerate(texts)
    ))


def test_repeated_text_across_indices_counts_excess_copies_and_characters():
    run = snapshot(["é\n🙂", "é\n🙂", "x", "é\n🙂", "x"])
    assert context_redundancy(run) == {
        "chunks": 5, "unique_texts": 2, "repeated_chunks": 3,
        "total_characters": 11, "repeated_characters": 7,
    }
    assert run == snapshot(["é\n🙂", "é\n🙂", "x", "é\n🙂", "x"])


def test_text_equality_preserves_case_whitespace_and_unicode_normalization():
    assert context_redundancy(snapshot(["A", "a", "a ", "é", "e\u0301"])) == {
        "chunks": 5, "unique_texts": 5, "repeated_chunks": 0,
        "total_characters": 7, "repeated_characters": 0,
    }


def test_empty_text_and_empty_runs_have_defined_counts():
    assert context_redundancy(snapshot(["", "", " "])) == {
        "chunks": 3, "unique_texts": 2, "repeated_chunks": 1,
        "total_characters": 1, "repeated_characters": 0,
    }
    assert context_redundancy(snapshot([])) == {
        "chunks": 0, "unique_texts": 0, "repeated_chunks": 0,
        "total_characters": 0, "repeated_characters": 0,
    }
