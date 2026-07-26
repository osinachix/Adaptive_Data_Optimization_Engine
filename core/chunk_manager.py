from collections.abc import Iterator

import polars as pl

from core.reader import Reader
from core.types import Chunk, Schema


class ChunkManager:
    """Turns a Reader's batches into a stream of Chunk objects with correct
    index and is_last flags. The single source of chunk boundaries; readers
    never emit Chunk objects directly."""

    def __init__(self, reader: Reader, rows_per_chunk: int) -> None:
        """reader: the Reader to stream from.
        rows_per_chunk: maximum rows per yielded Chunk, passed through to
        the reader's read_batches().
        """
        self._reader = reader
        self._rows_per_chunk = rows_per_chunk

    def stream(self) -> Iterator[Chunk]:
        """Open the reader, yield Chunk objects until exhausted, close the
        reader even if iteration is abandoned (use try/finally)."""
        self._reader.open()
        try:
            schema: Schema = self._reader.schema()
            batches: Iterator[pl.DataFrame] = self._reader.read_batches(
                self._rows_per_chunk
            )

            try:
                previous = next(batches)
            except StopIteration:
                return

            index = 0
            for current in batches:
                yield Chunk(
                    data=previous, index=index, is_last=False, source_schema=schema
                )
                previous = current
                index += 1

            yield Chunk(data=previous, index=index, is_last=True, source_schema=schema)
        finally:
            self._reader.close()
