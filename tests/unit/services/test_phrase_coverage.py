"""Literal phrase coverage of complete immutable retrieval snapshots."""
from types import SimpleNamespace

from unravel.services.retrieval.comparison import phrase_coverage, snapshot_retrieval


def snapshot(texts):
    return snapshot_retrieval("query", [
        SimpleNamespace(index=i, text=text, score=0.5) for i, text in enumerate(texts)
    ], {})


def test_counts_matching_chunks_once_and_keeps_first_phrase_spelling():
    baseline = snapshot(["Straße STRASSE", "[a].*", "Straße", ""])
    current = snapshot(["strasse", "strasse", "[a]xx", ""])
    assert phrase_coverage(baseline, current, "  Straße  \n STRASSE\n[a].*\nmissing\n\n") == [
        {"Phrase": "Straße", "Baseline chunks": 2, "Current chunks": 2, "Chunk change": 0},
        {"Phrase": "[a].*", "Baseline chunks": 1, "Current chunks": 0, "Chunk change": -1},
        {"Phrase": "missing", "Baseline chunks": 0, "Current chunks": 0, "Chunk change": 0},
    ]


def test_does_not_join_chunks_or_normalize_unicode_and_handles_empty_runs():
    baseline = snapshot(["alpha", "beta", "é"])
    empty = snapshot([])
    assert phrase_coverage(baseline, empty, "alpha beta\né\né") == [
        {"Phrase": "alpha beta", "Baseline chunks": 0, "Current chunks": 0, "Chunk change": 0},
        {"Phrase": "é", "Baseline chunks": 0, "Current chunks": 0, "Chunk change": 0},
        {"Phrase": "é", "Baseline chunks": 1, "Current chunks": 0, "Chunk change": -1},
    ]
    assert phrase_coverage(empty, baseline, "alpha")[0]["Chunk change"] == 1
    assert phrase_coverage(empty, empty, "x")[0]["Current chunks"] == 0
    assert phrase_coverage(baseline, baseline, " \n\t") == []
