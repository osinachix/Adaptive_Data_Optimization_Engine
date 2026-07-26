from pathlib import Path

import openpyxl
import polars as pl
import pytest

from core.chunk_manager import ChunkManager
from core.readers.excel_reader import ExcelReader
from core.types import Schema

EXPECTED_SCHEMA = Schema(columns={"id": pl.Int64(), "name": pl.String()})


def _write_xlsx(path: Path, rows: int) -> pl.DataFrame:
    df = pl.DataFrame({"id": range(rows), "name": [f"row-{i}" for i in range(rows)]})
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(list(df.columns))
    for row in df.iter_rows():
        sheet.append(row)
    workbook.save(path)
    return df


def test_schema_raises_before_open(tmp_path: Path) -> None:
    reader = ExcelReader(tmp_path / "data.xlsx")

    with pytest.raises(RuntimeError):
        reader.schema()


def test_read_batches_raises_before_open(tmp_path: Path) -> None:
    reader = ExcelReader(tmp_path / "data.xlsx")

    with pytest.raises(RuntimeError):
        next(reader.read_batches(10))


def test_open_raises_for_missing_file(tmp_path: Path) -> None:
    reader = ExcelReader(tmp_path / "missing.xlsx")

    with pytest.raises(FileNotFoundError):
        reader.open()


def test_open_establishes_schema(tmp_path: Path) -> None:
    xlsx_path = tmp_path / "data.xlsx"
    _write_xlsx(xlsx_path, rows=5)
    reader = ExcelReader(xlsx_path)

    reader.open()

    assert reader.schema() == EXPECTED_SCHEMA


def test_read_batches_respects_rows_per_batch(tmp_path: Path) -> None:
    xlsx_path = tmp_path / "data.xlsx"
    _write_xlsx(xlsx_path, rows=2_500)
    reader = ExcelReader(xlsx_path)
    reader.open()

    sizes = [batch.height for batch in reader.read_batches(700)]

    assert sizes == [700, 700, 700, 400]


def test_read_batches_preserves_all_rows_without_loss_or_duplication(
    tmp_path: Path,
) -> None:
    xlsx_path = tmp_path / "data.xlsx"
    expected = _write_xlsx(xlsx_path, rows=2_500)
    reader = ExcelReader(xlsx_path)
    reader.open()

    combined = pl.concat(list(reader.read_batches(700)))

    assert combined.equals(expected)


def test_read_batches_beyond_the_schema_sample_still_uses_the_same_schema(
    tmp_path: Path,
) -> None:
    """Schema is inferred from only the first _SCHEMA_SAMPLE_ROWS rows;
    rows read afterward (well beyond that sample) must still come back
    cast to that same schema, not re-inferred per batch."""
    xlsx_path = tmp_path / "data.xlsx"
    expected = _write_xlsx(xlsx_path, rows=1_500)
    reader = ExcelReader(xlsx_path)
    reader.open()

    combined = pl.concat(list(reader.read_batches(200)))

    assert combined.schema == EXPECTED_SCHEMA.columns
    assert combined.equals(expected)


def test_open_falls_back_to_string_schema_for_a_header_only_sheet(
    tmp_path: Path,
) -> None:
    xlsx_path = tmp_path / "empty.xlsx"
    _write_xlsx(xlsx_path, rows=0)
    reader = ExcelReader(xlsx_path)

    reader.open()

    assert reader.schema() == Schema(columns={"id": pl.String(), "name": pl.String()})


def test_close_is_idempotent_and_resets_state(tmp_path: Path) -> None:
    xlsx_path = tmp_path / "data.xlsx"
    _write_xlsx(xlsx_path, rows=5)
    reader = ExcelReader(xlsx_path)
    reader.open()

    reader.close()
    reader.close()  # must not raise

    with pytest.raises(RuntimeError):
        reader.schema()


def test_chunk_manager_streams_excel_reader_end_to_end(tmp_path: Path) -> None:
    xlsx_path = tmp_path / "data.xlsx"
    expected = _write_xlsx(xlsx_path, rows=1_100)
    manager = ChunkManager(ExcelReader(xlsx_path), rows_per_chunk=500)

    chunks = list(manager.stream())

    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert [chunk.is_last for chunk in chunks] == [False] * (len(chunks) - 1) + [True]
    assert all(chunk.source_schema == EXPECTED_SCHEMA for chunk in chunks)

    combined = pl.concat([chunk.data for chunk in chunks])
    assert combined.equals(expected)
