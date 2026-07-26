from collections.abc import Generator, Iterator
from typing import cast

import polars as pl

from core.chunk_manager import ChunkManager
from core.types import Chunk, Schema

SCHEMA = Schema(columns={"id": pl.Int64(), "name": pl.String()})


class FakeReader:
    """Minimal stand-in for the Reader protocol (core/reader.py, added in
    M3): open(), schema(), read_batches(rows_per_batch), close(). Exists
    only to drive ChunkManager in these tests."""

    def __init__(self, batches: list[pl.DataFrame], schema: Schema) -> None:
        self._batches = batches
        self._schema = schema
        self.opened = False
        self.closed = False
        self.open_count = 0
        self.close_count = 0
        self.requested_rows_per_batch: int | None = None

    def open(self) -> None:
        self.opened = True
        self.open_count += 1

    def schema(self) -> Schema:
        return self._schema

    def read_batches(self, rows_per_batch: int) -> Iterator[pl.DataFrame]:
        self.requested_rows_per_batch = rows_per_batch
        yield from self._batches

    def close(self) -> None:
        self.closed = True
        self.close_count += 1


def _make_batches(sizes: list[int]) -> list[pl.DataFrame]:
    batches = []
    start = 0
    for size in sizes:
        ids = list(range(start, start + size))
        batches.append(pl.DataFrame({"id": ids, "name": [f"row-{i}" for i in ids]}))
        start += size
    return batches


def test_stream_assigns_sequential_index_and_marks_final_chunk() -> None:
    reader = FakeReader(_make_batches([2, 3, 1]), SCHEMA)
    manager = ChunkManager(reader, rows_per_chunk=2)

    chunks = list(manager.stream())

    assert [chunk.index for chunk in chunks] == [0, 1, 2]
    assert [chunk.is_last for chunk in chunks] == [False, False, True]


def test_stream_preserves_all_rows_without_loss_or_duplication() -> None:
    batches = _make_batches([2, 3, 1])
    reader = FakeReader(batches, SCHEMA)
    manager = ChunkManager(reader, rows_per_chunk=2)

    chunks = list(manager.stream())
    combined = pl.concat([chunk.data for chunk in chunks])
    expected = pl.concat(batches)

    assert combined.equals(expected)


def test_stream_attaches_reader_schema_to_every_chunk() -> None:
    reader = FakeReader(_make_batches([1, 1]), SCHEMA)
    manager = ChunkManager(reader, rows_per_chunk=1)

    chunks = list(manager.stream())

    assert all(chunk.source_schema == SCHEMA for chunk in chunks)


def test_stream_opens_once_and_forwards_rows_per_chunk() -> None:
    reader = FakeReader(_make_batches([1]), SCHEMA)
    manager = ChunkManager(reader, rows_per_chunk=7)

    list(manager.stream())

    assert reader.open_count == 1
    assert reader.requested_rows_per_batch == 7


def test_stream_closes_reader_after_full_consumption() -> None:
    reader = FakeReader(_make_batches([1, 1]), SCHEMA)
    manager = ChunkManager(reader, rows_per_chunk=1)

    list(manager.stream())

    assert reader.closed is True
    assert reader.close_count == 1


def test_stream_closes_reader_when_iteration_is_abandoned() -> None:
    reader = FakeReader(_make_batches([1, 1, 1]), SCHEMA)
    manager = ChunkManager(reader, rows_per_chunk=1)

    generator = cast(Generator[Chunk, None, None], manager.stream())
    next(generator)  # consume one chunk, leave the rest unread
    generator.close()  # simulate the caller abandoning iteration

    assert reader.closed is True
    assert reader.close_count == 1


def test_stream_closes_reader_even_when_no_batches_are_produced() -> None:
    reader = FakeReader([], SCHEMA)
    manager = ChunkManager(reader, rows_per_chunk=10)

    chunks = list(manager.stream())

    assert chunks == []
    assert reader.opened is True
    assert reader.closed is True
