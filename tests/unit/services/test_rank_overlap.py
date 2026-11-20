"""Rank-prefix overlap uses chunk identity, never score scales or view filters."""

from types import SimpleNamespace

from unravel.services.retrieval.comparison import rank_overlap_profile, snapshot_retrieval


def snapshot(records):
    return snapshot_retrieval("query", [
        SimpleNamespace(index=index, text=text, score=score)
        for index, text, score in records
    ], {})


def test_crossing_ranks_and_unequal_lengths():
    before = snapshot([(0, "a", 0.9), (1, "b", 0.8), (2, "c", 0.7)])
    after = snapshot([(1, "b", 99), (0, "a", -10)])
    assert rank_overlap_profile(before, after) == [
        {"Depth": 1, "Baseline chunks": 1, "Current chunks": 1,
         "Shared chunks": 0, "Union chunks": 2, "Overlap (%)": 0.0},
        {"Depth": 2, "Baseline chunks": 2, "Current chunks": 2,
         "Shared chunks": 2, "Union chunks": 2, "Overlap (%)": 100.0},
        {"Depth": 3, "Baseline chunks": 3, "Current chunks": 2,
         "Shared chunks": 2, "Union chunks": 3, "Overlap (%)": 66.7},
    ]


def test_identity_deduplication_and_changed_text():
    before = snapshot([(0, "a", 1), (0, "a", 2), (1, "a", 3)])
    after = snapshot([(0, "changed", 1), (1, "a", 2)])
    profile = rank_overlap_profile(before, after)
    assert len(profile) == 2
    assert profile[-1]["Shared chunks"] == 1
    assert profile[-1]["Overlap (%)"] == 33.3
    assert len(before.chunks) == 2
    assert before.chunks[0].score == 1


def test_empty_runs_are_zero_overlap_and_both_empty_have_no_depths():
    empty = snapshot([])
    full = snapshot([(0, "a", 0)])
    assert rank_overlap_profile(empty, empty) == []
    for before, after in [(empty, full), (full, empty)]:
        assert rank_overlap_profile(before, after)[0]["Overlap (%)"] == 0.0
        assert rank_overlap_profile(before, after)[0]["Shared chunks"] == 0
