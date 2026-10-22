from unravel.services.chunk_diagnostics import get_chunk_size_rows
from unravel.services.chunking import Chunk


def test_rows_use_raw_text_and_keep_original_positions():
    chunks = [
        Chunk("<p>café \n text</p>", 7, 25, {"size": 999}),
        Chunk("", -1, -1, {}),
        Chunk("  ", 40, 42, {}),
    ]
    assert get_chunk_size_rows(chunks) == [
        {"chunk": 1, "characters": 18, "words": 2, "duplicate_of": None,
         "start_index": 7, "end_index": 25},
        {"chunk": 2, "characters": 0, "words": 0, "duplicate_of": None,
         "start_index": -1, "end_index": -1},
        {"chunk": 3, "characters": 2, "words": 0, "duplicate_of": None,
         "start_index": 40, "end_index": 42},
    ]
    assert chunks[0].metadata == {"size": 999}


def test_duplicates_reference_first_exact_raw_text_and_include_empty_chunks():
    texts = ["same", "other", "same", "Same", "same ", "", "", "same"]
    chunks = [Chunk(text, i, i + len(text), {}) for i, text in enumerate(texts)]
    assert [row["duplicate_of"] for row in get_chunk_size_rows(chunks)] == [
        None, None, 1, None, None, None, 6, 1,
    ]
    assert get_chunk_size_rows([]) == []
