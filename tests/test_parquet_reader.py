from pathlib import Path

import polars as pl
import pytest

from core.chunk_manager import ChunkManager
from core.readers.parquet_reader import ParquetReader
from core.types import Schema

EXPECTED_SCHEMA = Schema(columns={"id": pl.Int64(), "name": pl.String()})


def _write_parquet(path: Path, rows: int, row_group_size: int = 1_000) -> pl.DataFrame:
    df = pl.DataFrame({"id": range(rows), "name": [f"row-{i}" for i in range(rows)]})
    df.write_parquet(path, row_group_size=row_group_size)
    return df


def test_schema_raises_before_open(tmp_path: Path) -> None:
    reader = ParquetReader(tmp_path / "data.parquet")

    with pytest.raises(RuntimeError):
        reader.schema()


def test_read_batches_raises_before_open(tmp_path: Path) -> None:
    reader = ParquetReader(tmp_path / "data.parquet")

    with pytest.raises(RuntimeError):
        next(reader.read_batches(10))


def test_open_raises_for_missing_file(tmp_path: Path) -> None:
    reader = ParquetReader(tmp_path / "missing.parquet")

    with pytest.raises(FileNotFoundError):
        reader.open()


def test_open_establishes_schema(tmp_path: Path) -> None:
    parquet_path = tmp_path / "data.parquet"
    _write_parquet(parquet_path, rows=5)
    reader = ParquetReader(parquet_path)

    reader.open()

    assert reader.schema() == EXPECTED_SCHEMA


def test_read_batches_respects_rows_per_batch(tmp_path: Path) -> None:
    parquet_path = tmp_path / "data.parquet"
    _write_parquet(parquet_path, rows=10_000, row_group_size=2_000)
    reader = ParquetReader(parquet_path)
    reader.open()

    sizes = [batch.height for batch in reader.read_batches(3_000)]

    assert sizes == [3_000, 3_000, 3_000, 1_000]


def test_read_batches_preserves_all_rows_without_loss_or_duplication(
    tmp_path: Path,
) -> None:
    parquet_path = tmp_path / "data.parquet"
    expected = _write_parquet(parquet_path, rows=2_500, row_group_size=400)
    reader = ParquetReader(parquet_path)
    reader.open()

    combined = pl.concat(list(reader.read_batches(700)))

    assert combined.equals(expected)


def test_read_batches_yields_incrementally_without_exhausting_source(
    tmp_path: Path,
) -> None:
    parquet_path = tmp_path / "data.parquet"
    _write_parquet(parquet_path, rows=50_000, row_group_size=5_000)
    reader = ParquetReader(parquet_path)
    reader.open()

    batches = reader.read_batches(1_000)
    first = next(batches)

    assert first.height == 1_000


def test_close_is_idempotent_and_resets_state(tmp_path: Path) -> None:
    parquet_path = tmp_path / "data.parquet"
    _write_parquet(parquet_path, rows=5)
    reader = ParquetReader(parquet_path)
    reader.open()

    reader.close()
    reader.close()  # must not raise

    with pytest.raises(RuntimeError):
        reader.schema()


def test_corrupt_file_raises_an_actionable_error_not_a_silent_crash(
    tmp_path: Path,
) -> None:
    """Parquet is a strongly-typed binary format, so there is no ragged-row
    equivalent to a malformed CSV line; the analogous failure mode is a
    corrupt or truncated file. open() must fail with a specific, actionable
    exception rather than a silent or opaque crash."""
    parquet_path = tmp_path / "corrupt.parquet"
    parquet_path.write_bytes(b"this is not a real parquet file" * 5)
    reader = ParquetReader(parquet_path)

    with pytest.raises(pl.exceptions.ComputeError, match="[Pp]arquet"):
        reader.open()


def test_read_batches_streams_a_large_parquet_file_with_exact_row_count(
    tmp_path: Path,
) -> None:
    """Feed a deliberately large Parquet file through small batches and
    confirm the row count reassembles exactly, without ever holding the
    full dataset as a single DataFrame (rows are summed batch by batch as
    they stream)."""
    parquet_path = tmp_path / "large.parquet"
    total_rows = 200_000
    _write_parquet(parquet_path, rows=total_rows, row_group_size=10_000)
    reader = ParquetReader(parquet_path)
    reader.open()

    row_count = 0
    batch_count = 0
    for batch in reader.read_batches(500):
        row_count += batch.height
        batch_count += 1

    assert row_count == total_rows
    assert batch_count == total_rows // 500


def test_chunk_manager_streams_parquet_reader_end_to_end(tmp_path: Path) -> None:
    parquet_path = tmp_path / "data.parquet"
    expected = _write_parquet(parquet_path, rows=2_200, row_group_size=300)
    manager = ChunkManager(ParquetReader(parquet_path), rows_per_chunk=500)

    chunks = list(manager.stream())

    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert [chunk.is_last for chunk in chunks] == [False] * (len(chunks) - 1) + [True]
    assert all(chunk.source_schema == EXPECTED_SCHEMA for chunk in chunks)

    combined = pl.concat([chunk.data for chunk in chunks])
    assert combined.equals(expected)
