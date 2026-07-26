import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import IO, cast

import polars as pl
import pyarrow.parquet as pq
import zstandard

from core.mode import OptimizationMode
from core.types import Chunk
from engine.validator import ValidationResult, Validator

# When compress=True, Parquet uses Zstandard (consistently better than
# PyArrow's default, Snappy, for the same lossless guarantee). When
# compress=False it uses no codec at all, so compress consistently means
# the same thing - "apply Zstandard to the output" - across every export
# format instead of Parquet always compressing regardless of the flag.
_PARQUET_COMPRESSION = "zstd"


class ExportFormat(StrEnum):
    CSV = "csv"
    PARQUET = "parquet"
    JSON = "json"

    @classmethod
    def from_suffix(cls, path: str | Path) -> "ExportFormat | None":
        """Infer the export format from path's suffix, for any caller that
        needs to turn a filename into a format (the CLI's --out, the GUI's
        uploaded filename). A trailing .zst (Zstandard-compressed CSV/JSON)
        is stripped first, so foo.csv.zst is still recognized as csv; .jsonl
        is treated the same as .json, since both are the newline-delimited
        JSON this engine reads and writes (see JSONReader). Returns None
        for an unrecognized suffix rather than raising, so callers can
        decide how to report that themselves."""
        resolved = Path(path)
        if resolved.suffix == ".zst":
            resolved = resolved.with_suffix("")
        suffix = resolved.suffix.lower().lstrip(".")
        if suffix == "jsonl":
            suffix = "json"
        try:
            return cls(suffix)
        except ValueError:
            return None


class ExportError(Exception):
    """Raised when export is refused because validation failed (invariant
    I3). Carries the ValidationResult that caused the refusal, so callers
    can still build a report (e.g. via ReportGenerator.finalize()) even
    though nothing was written. Nothing is written to the requested output
    path when this is raised: any in-progress temporary output is removed
    first."""

    def __init__(self, validation: ValidationResult) -> None:
        super().__init__(
            "export refused: validation failed - " + "; ".join(validation.reasons)
        )
        self.validation = validation


class Exporter:
    """Streams an optimized Chunk sequence to disk, validating it against
    the original stream as it goes.

    Invariant I3 requires both that nothing is written until validation
    passes, and (invariant I1) that the full dataset is never held in
    memory - which for a large streamed dataset are only simultaneously
    satisfiable by writing incrementally and gating on the *destination*,
    not on buffering. So export() writes each optimized chunk to a
    temporary file as it arrives, interleaved with validating that chunk
    against the corresponding original chunk. Only once the Validator has
    seen every chunk and reports passed=True is the temporary file moved
    to the real output path (an atomic rename on the same filesystem). If
    it reports passed=False - or if anything else goes wrong mid-export -
    the temporary file is deleted and ExportError is raised with the
    validator's reasons. The real output path therefore never shows
    partial or invalid content, without ever buffering the whole dataset
    to guarantee that.
    """

    def export(
        self,
        original_chunks: Iterator[Chunk],
        optimized_chunks: Iterator[Chunk],
        mode: OptimizationMode,
        output_path: str | Path,
        export_format: ExportFormat,
        compress: bool = False,
    ) -> ValidationResult:
        """Validate and write optimized_chunks (paired index-for-index
        with original_chunks) to output_path in export_format. Returns the
        ValidationResult on success. Raises ExportError - writing nothing
        to output_path - if validation fails or export is interrupted.

        compress applies uniformly across formats: for CSV/JSON, the file
        at output_path is itself a Zstandard-compressed stream (the caller
        decides the filename, e.g. appending .zst; Exporter just writes
        whatever bytes result to exactly the path it's given); for
        Parquet, it selects the Zstandard codec instead of no codec at
        the Parquet-writer level. Either way, compress=False means the
        output bytes are genuinely uncompressed - there is no format
        where compression is forced on regardless of the flag. For
        CSV/JSON, compression happens after validation has already
        confirmed the data is correct, on the final bytes only - it never
        touches the Chunk/DataFrame representation the Validator compares,
        so it cannot affect correctness.
        """
        output_path = Path(output_path)
        temp_path = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
        validator = Validator(mode)

        try:
            with self._open_writer(temp_path, export_format, compress) as write_chunk:
                for original_chunk, optimized_chunk in zip(
                    original_chunks, optimized_chunks, strict=True
                ):
                    write_chunk(optimized_chunk.data)
                    validator.observe(original_chunk, optimized_chunk)

            validation = validator.finalize()
            if not validation.passed:
                raise ExportError(validation)

            temp_path.replace(output_path)  # atomic on the same filesystem
            return validation
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise

    @contextmanager
    def _open_writer(
        self, path: Path, export_format: ExportFormat, compress: bool
    ) -> Iterator[Callable[[pl.DataFrame], None]]:
        """Yield a callable that appends one chunk's rows to path, in the
        requested format. The file at path is only ever a temporary
        working file; callers are responsible for committing or discarding
        it."""
        if export_format is ExportFormat.CSV:
            with self._open_raw_or_compressed(path, compress) as handle:
                state = {"wrote_header": False}

                def write_csv_chunk(data: pl.DataFrame) -> None:
                    data.write_csv(handle, include_header=not state["wrote_header"])
                    state["wrote_header"] = True

                yield write_csv_chunk

        elif export_format is ExportFormat.JSON:
            with self._open_raw_or_compressed(path, compress) as handle:

                def write_json_chunk(data: pl.DataFrame) -> None:
                    data.write_ndjson(handle)

                yield write_json_chunk

        elif export_format is ExportFormat.PARQUET:
            writer_box: list[pq.ParquetWriter] = []
            parquet_compression = _PARQUET_COMPRESSION if compress else None
            try:

                def write_parquet_chunk(data: pl.DataFrame) -> None:
                    table = data.to_arrow()
                    if not writer_box:
                        writer_box.append(
                            pq.ParquetWriter(
                                path, table.schema, compression=parquet_compression
                            )
                        )
                    writer_box[0].write_table(table)

                yield write_parquet_chunk
            finally:
                if writer_box:
                    writer_box[0].close()

        else:
            raise ValueError(f"unsupported export format: {export_format!r}")

    @contextmanager
    def _open_raw_or_compressed(
        self, path: Path, compress: bool
    ) -> Iterator[IO[bytes]]:
        """Open path for writing, streaming through a Zstandard compressor
        first if compress is True. Chunks are still written incrementally
        either way - the compressor streams, it doesn't buffer the whole
        file - so this doesn't reintroduce the full-dataset-in-memory
        problem invariant I1 forbids."""
        with open(path, "wb") as raw_handle:
            if not compress:
                yield raw_handle
                return
            compressor = zstandard.ZstdCompressor()
            with compressor.stream_writer(raw_handle) as compressed_handle:
                yield cast(IO[bytes], compressed_handle)
