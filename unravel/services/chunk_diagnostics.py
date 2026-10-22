"""Local diagnostics over raw generated chunk text."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import Sequence

    from unravel.services.chunking import Chunk


class ChunkSizeRow(TypedDict):
    chunk: int
    characters: int
    words: int
    duplicate_of: int | None
    start_index: int
    end_index: int


def get_chunk_size_rows(chunks: Sequence[Chunk]) -> list[ChunkSizeRow]:
    """Keep original ordering and identify exact repeats without changing chunks."""
    first_seen: dict[str, int] = {}
    rows: list[ChunkSizeRow] = []
    for number, chunk in enumerate(chunks, start=1):
        rows.append({
            "chunk": number,
            "characters": len(chunk.text),
            "words": len(chunk.text.split()),
            "duplicate_of": first_seen.get(chunk.text),
            "start_index": chunk.start_index,
            "end_index": chunk.end_index,
        })
        first_seen.setdefault(chunk.text, number)
    return rows
