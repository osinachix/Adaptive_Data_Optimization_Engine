from collections.abc import Iterator
from typing import Protocol

import polars as pl

from core.types import Schema


class Reader(Protocol):
    """Streams a dataset as a sequence of row batches. Format-specific
    implementations live in core/readers/. Never reads the whole file into
    memory."""

    def open(self) -> None:
        """Acquire the file handle and read enough to establish the schema."""

    def schema(self) -> Schema:
        """Return the dataset schema. Valid only after open()."""

    def read_batches(self, rows_per_batch: int) -> Iterator[pl.DataFrame]:
        """Yield row batches of at most rows_per_batch rows, streaming."""

    def close(self) -> None:
        """Release the file handle. Idempotent."""
