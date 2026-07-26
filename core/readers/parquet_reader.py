from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from core.types import Schema


class ParquetReader:
    """Streams a Parquet file as row batches, reading row group by row
    group via PyArrow so the full file is never materialized at once.

    open() reads only the Parquet footer (schema and row-group metadata),
    not the column data; read_batches() streams rows through PyArrow's
    ParquetFile.iter_batches(), which walks row groups incrementally.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._parquet_file: pq.ParquetFile | None = None
        self._schema: Schema | None = None

    def open(self) -> None:
        """Validate the file exists and establish the schema from the
        Parquet footer."""
        if not self._path.is_file():
            raise FileNotFoundError(f"Parquet file not found: {self._path}")
        self._schema = Schema(columns=dict(pl.read_parquet_schema(self._path)))
        self._parquet_file = pq.ParquetFile(self._path)

    def schema(self) -> Schema:
        """Return the dataset schema. Valid only after open()."""
        if self._schema is None:
            raise RuntimeError("ParquetReader.schema() called before open()")
        return self._schema

    def read_batches(self, rows_per_batch: int) -> Iterator[pl.DataFrame]:
        """Yield row batches of at most rows_per_batch rows, streaming
        row group by row group."""
        if self._parquet_file is None:
            raise RuntimeError("ParquetReader.read_batches() called before open()")
        for record_batch in self._parquet_file.iter_batches(batch_size=rows_per_batch):
            yield pl.DataFrame(record_batch)

    def close(self) -> None:
        """Release the file handle. Idempotent."""
        self._parquet_file = None
        self._schema = None
