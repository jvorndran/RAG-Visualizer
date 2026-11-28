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
        {"Phrase": "Straße", "Baseline chunks": 2, "Current chunks": 2, "Chunk change": 0,
         "Baseline first rank": 1, "Current first rank": 1, "First rank change": 0},
        {"Phrase": "[a].*", "Baseline chunks": 1, "Current chunks": 0, "Chunk change": -1,
         "Baseline first rank": 2, "Current first rank": None, "First rank change": None},
        {"Phrase": "missing", "Baseline chunks": 0, "Current chunks": 0, "Chunk change": 0,
         "Baseline first rank": None, "Current first rank": None, "First rank change": None},
    ]


def test_does_not_join_chunks_or_normalize_unicode_and_handles_empty_runs():
    baseline = snapshot(["alpha", "beta", "é"])
    empty = snapshot([])
    assert phrase_coverage(baseline, empty, "alpha beta\né\né") == [
        {"Phrase": "alpha beta", "Baseline chunks": 0, "Current chunks": 0, "Chunk change": 0,
         "Baseline first rank": None, "Current first rank": None, "First rank change": None},
        {"Phrase": "é", "Baseline chunks": 0, "Current chunks": 0, "Chunk change": 0,
         "Baseline first rank": None, "Current first rank": None, "First rank change": None},
        {"Phrase": "é", "Baseline chunks": 1, "Current chunks": 0, "Chunk change": -1,
         "Baseline first rank": 3, "Current first rank": None, "First rank change": None},
    ]
    assert phrase_coverage(empty, baseline, "alpha")[0]["Chunk change"] == 1
    assert phrase_coverage(empty, empty, "x")[0]["Current chunks"] == 0
    assert phrase_coverage(baseline, baseline, " \n\t") == []


def test_first_match_ranks_distinguish_earlier_later_unchanged_and_absent_phrases():
    baseline = snapshot(["later", "earlier", "same", "earlier earlier", "lost"])
    current = snapshot(["earlier EARLIER", "later", "same", "new"])
    rows = phrase_coverage(baseline, current, "earlier\nlater\nsame\nlost\nnew\nabsent")
    assert [
        (row["Baseline first rank"], row["Current first rank"], row["First rank change"])
        for row in rows
    ] == [(2, 1, 1), (1, 2, -1), (3, 3, 0), (5, None, None),
          (None, 4, None), (None, None, None)]
    assert [row["Baseline chunks"] for row in rows] == [2, 1, 1, 1, 0, 0]
    assert [row["Current chunks"] for row in rows] == [1, 1, 1, 0, 1, 0]


def test_first_rank_uses_deduplicated_retrieval_order_and_literal_casefold_matching():
    baseline = snapshot_retrieval("baseline", [
        SimpleNamespace(index=90, text="filler", score=0.1),
        SimpleNamespace(index=90, text="filler", score=100),
        SimpleNamespace(index=0, text="Straße [a].*", score=0.0),
        SimpleNamespace(index=1, text="STRASSE [a].*", score=0.9),
    ], {})
    current = snapshot(["strasse [a]xx", "[a].*", "strasse"])
    before = (baseline.chunks, current.chunks)
    rows = phrase_coverage(baseline, current, "  STRASSE  \n Straße\n[a].*\n[a]xx")
    assert [row["Phrase"] for row in rows] == ["STRASSE", "[a].*", "[a]xx"]
    assert [row["Baseline first rank"] for row in rows] == [2, 2, None]
    assert [row["Current first rank"] for row in rows] == [1, 2, 1]
    assert [row["First rank change"] for row in rows] == [1, 0, None]
    assert (baseline.chunks, current.chunks) == before


def test_first_match_ranks_are_missing_for_empty_runs_and_cross_chunk_phrases():
    baseline = snapshot(["alpha", "beta", "é"])
    empty = snapshot([])
    rows = phrase_coverage(baseline, empty, "alpha beta\né\né")
    assert [row["Baseline first rank"] for row in rows] == [None, None, 3]
    assert all(row["Current first rank"] is None for row in rows)
    assert all(row["First rank change"] is None for row in rows)
    assert phrase_coverage(empty, baseline, "é")[0]["Current first rank"] == 3
    assert phrase_coverage(empty, empty, "x")[0]["First rank change"] is None
