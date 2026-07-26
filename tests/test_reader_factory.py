from pathlib import Path

import pytest

from core.readers.csv_reader import CSVReader
from core.readers.excel_reader import ExcelReader
from core.readers.factory import ReaderFactory
from core.readers.json_reader import JSONReader
from core.readers.parquet_reader import ParquetReader


def test_for_path_selects_csv_reader_for_csv_extension(tmp_path: Path) -> None:
    reader = ReaderFactory.for_path(tmp_path / "data.csv")

    assert isinstance(reader, CSVReader)


def test_for_path_selects_parquet_reader_for_parquet_extension(tmp_path: Path) -> None:
    reader = ReaderFactory.for_path(tmp_path / "data.parquet")

    assert isinstance(reader, ParquetReader)


def test_for_path_selects_json_reader_for_json_extension(tmp_path: Path) -> None:
    reader = ReaderFactory.for_path(tmp_path / "data.json")

    assert isinstance(reader, JSONReader)


def test_for_path_selects_json_reader_for_jsonl_extension(tmp_path: Path) -> None:
    reader = ReaderFactory.for_path(tmp_path / "data.jsonl")

    assert isinstance(reader, JSONReader)


def test_for_path_selects_excel_reader_for_xlsx_extension(tmp_path: Path) -> None:
    reader = ReaderFactory.for_path(tmp_path / "data.xlsx")

    assert isinstance(reader, ExcelReader)


def test_for_path_is_case_insensitive_on_extension(tmp_path: Path) -> None:
    reader = ReaderFactory.for_path(tmp_path / "DATA.CSV")

    assert isinstance(reader, CSVReader)


def test_for_path_raises_a_clear_error_for_an_unsupported_extension(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="no reader registered"):
        ReaderFactory.for_path(tmp_path / "data.txt")


def test_for_path_strips_a_trailing_zst_before_dispatching(tmp_path: Path) -> None:
    """adoe optimize --compress writes foo.csv.zst; ReaderFactory must
    still recognize it as CSV so `adoe validate`/`adoe profile` can read
    a compressed output back without special-casing."""
    reader = ReaderFactory.for_path(tmp_path / "data.csv.zst")

    assert isinstance(reader, CSVReader)


def test_for_path_constructs_the_reader_against_the_original_zst_path(
    tmp_path: Path,
) -> None:
    """The .zst suffix is only stripped to choose the reader class; the
    reader itself must still be pointed at the real (compressed) file."""
    zst_path = tmp_path / "data.csv.zst"

    reader = ReaderFactory.for_path(zst_path)

    assert isinstance(reader, CSVReader)
    assert reader._path == zst_path


def test_for_path_still_rejects_an_unsupported_format_under_a_zst_suffix(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="no reader registered"):
        ReaderFactory.for_path(tmp_path / "data.txt.zst")
