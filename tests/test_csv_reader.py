from pathlib import Path

import polars as pl
import pytest

from core.chunk_manager import ChunkManager
from core.readers.csv_reader import CSVReader
from core.types import Schema

EXPECTED_SCHEMA = Schema(columns={"id": pl.Int64(), "name": pl.String()})


def _write_csv(path: Path, rows: int) -> pl.DataFrame:
    df = pl.DataFrame({"id": range(rows), "name": [f"row-{i}" for i in range(rows)]})
    df.write_csv(path)
    return df


def test_schema_raises_before_open(tmp_path: Path) -> None:
    reader = CSVReader(tmp_path / "data.csv")

    with pytest.raises(RuntimeError):
        reader.schema()


def test_read_batches_raises_before_open(tmp_path: Path) -> None:
    reader = CSVReader(tmp_path / "data.csv")

    with pytest.raises(RuntimeError):
        next(reader.read_batches(10))


def test_open_raises_for_missing_file(tmp_path: Path) -> None:
    reader = CSVReader(tmp_path / "missing.csv")

    with pytest.raises(FileNotFoundError):
        reader.open()


def test_open_establishes_schema(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path, rows=5)
    reader = CSVReader(csv_path)

    reader.open()

    assert reader.schema() == EXPECTED_SCHEMA


def test_read_batches_respects_rows_per_batch(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path, rows=10_000)
    reader = CSVReader(csv_path)
    reader.open()

    sizes = [batch.height for batch in reader.read_batches(3_000)]

    assert sizes == [3_000, 3_000, 3_000, 1_000]


def test_read_batches_preserves_all_rows_without_loss_or_duplication(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "data.csv"
    expected = _write_csv(csv_path, rows=2_500)
    reader = CSVReader(csv_path)
    reader.open()

    combined = pl.concat(list(reader.read_batches(700)))

    assert combined.equals(expected)


def test_read_batches_yields_incrementally_without_exhausting_source(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path, rows=50_000)
    reader = CSVReader(csv_path)
    reader.open()

    batches = reader.read_batches(1_000)
    first = next(batches)

    assert first.height == 1_000


def test_close_is_idempotent_and_resets_state(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path, rows=5)
    reader = CSVReader(csv_path)
    reader.open()

    reader.close()
    reader.close()  # must not raise

    with pytest.raises(RuntimeError):
        reader.schema()


def test_malformed_row_raises_an_actionable_error_not_a_silent_crash(
    tmp_path: Path,
) -> None:
    """A row with more fields than the header (a ragged CSV line) must fail
    the read with a specific, actionable exception, per the guide's "fail
    gracefully with actionable error messages" objective, not proceed
    silently or crash with an opaque error."""
    csv_path = tmp_path / "malformed.csv"
    csv_path.write_text("id,name,age\n1,alice,30\n2,bob,25,extra_field\n3,carol,40\n")
    reader = CSVReader(csv_path)
    reader.open()

    with pytest.raises(pl.exceptions.ComputeError, match="fields"):
        list(reader.read_batches(2))


def test_read_batches_streams_a_large_csv_with_exact_row_count(
    tmp_path: Path,
) -> None:
    """Feed a deliberately large CSV through small batches and confirm the
    row count reassembles exactly, without ever holding the full dataset as
    a single DataFrame (rows are summed batch by batch as they stream)."""
    csv_path = tmp_path / "large.csv"
    total_rows = 200_000
    _write_csv(csv_path, rows=total_rows)
    reader = CSVReader(csv_path)
    reader.open()

    row_count = 0
    batch_count = 0
    for batch in reader.read_batches(500):
        row_count += batch.height
        batch_count += 1

    assert row_count == total_rows
    assert batch_count == total_rows // 500


def test_chunk_manager_streams_csv_reader_end_to_end(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    expected = _write_csv(csv_path, rows=2_200)
    manager = ChunkManager(CSVReader(csv_path), rows_per_chunk=500)

    chunks = list(manager.stream())

    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert [chunk.is_last for chunk in chunks] == [False] * (len(chunks) - 1) + [True]
    assert all(chunk.source_schema == EXPECTED_SCHEMA for chunk in chunks)

    combined = pl.concat([chunk.data for chunk in chunks])
    assert combined.equals(expected)
