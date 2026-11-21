"""Raw context sizes use full text at each rank, independent of retrieval scores."""

from unravel.services.retrieval.comparison import (
    RankedChunk,
    RetrievalSnapshot,
    context_size_profile,
)


def snapshot(texts):
    return RetrievalSnapshot(
        "query", "DenseRetriever", None, None, None,
        tuple(RankedChunk(i, text, 0.1, i + 1) for i, text in enumerate(texts)),
    )


def test_profile_counts_raw_unicode_whitespace_and_repeated_text():
    baseline = snapshot(["é\n", "🙂", "é\n"])
    current = snapshot(["", " a "])
    assert context_size_profile(baseline, current) == [
        {"Depth": 1, "Baseline characters": 2, "Current characters": 0,
         "Character change": -2, "Baseline UTF-8 bytes": 3, "Current UTF-8 bytes": 0},
        {"Depth": 2, "Baseline characters": 3, "Current characters": 3,
         "Character change": 0, "Baseline UTF-8 bytes": 7, "Current UTF-8 bytes": 3},
        {"Depth": 3, "Baseline characters": 5, "Current characters": 3,
         "Character change": -2, "Baseline UTF-8 bytes": 10, "Current UTF-8 bytes": 3},
    ]


def test_empty_runs_and_current_only_context():
    empty = snapshot([])
    assert context_size_profile(empty, empty) == []
    assert context_size_profile(empty, snapshot(["abc"])) == [
        {"Depth": 1, "Baseline characters": 0, "Current characters": 3,
         "Character change": 3, "Baseline UTF-8 bytes": 0, "Current UTF-8 bytes": 3},
    ]
    assert context_size_profile(snapshot(["abc"]), empty)[0]["Character change"] == -3


def test_profiles_leave_snapshots_unchanged_and_count_identical_text_per_chunk():
    baseline = snapshot(["same", "same"])
    current = snapshot(["same"])
    assert context_size_profile(baseline, current)[-1]["Baseline characters"] == 8
    assert baseline == snapshot(["same", "same"])
    assert current == snapshot(["same"])
