from collections.abc import Iterator
from pathlib import Path

import polars as pl

from core.types import Schema


class JSONReader:
    """Streams a newline-delimited JSON (NDJSON) file as row batches using
    Polars in streaming mode.

    Only NDJSON (one JSON object per line) is supported, not a single JSON
    array: NDJSON is the only JSON variant Polars can scan lazily, since a
    JSON array has to be parsed as a whole before any row can be yielded.
    This also mirrors what ExportFormat.JSON writes (see
    engine/exporter.py's write_ndjson), so round-tripping an ADOE JSON
    export back through this reader works without special-casing.

    open() establishes the schema from Polars' bounded schema-inference
    sample, not the whole file; read_batches() streams rows through
    Polars' streaming query engine, so the file is never materialized in
    full at once.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lazy_frame: pl.LazyFrame | None = None
        self._schema: Schema | None = None

    def open(self) -> None:
        """Validate the file exists and establish the schema."""
        if not self._path.is_file():
            raise FileNotFoundError(f"JSON file not found: {self._path}")
        self._lazy_frame = pl.scan_ndjson(self._path)
        self._schema = Schema(columns=dict(self._lazy_frame.collect_schema()))

    def schema(self) -> Schema:
        """Return the dataset schema. Valid only after open()."""
        if self._schema is None:
            raise RuntimeError("JSONReader.schema() called before open()")
        return self._schema

    def read_batches(self, rows_per_batch: int) -> Iterator[pl.DataFrame]:
        """Yield row batches of at most rows_per_batch rows, streaming."""
        if self._lazy_frame is None:
            raise RuntimeError("JSONReader.read_batches() called before open()")
        yield from self._lazy_frame.collect_batches(
            chunk_size=rows_per_batch, engine="streaming"
        )

    def close(self) -> None:
        """Release the file handle. Idempotent."""
        self._lazy_frame = None
        self._schema = None
