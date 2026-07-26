from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest
import zstandard

from core.mode import OptimizationMode
from core.types import Chunk, Schema
from engine.exporter import Exporter, ExportError, ExportFormat


def _chunk(data: pl.DataFrame, schema: Schema, index: int, is_last: bool) -> Chunk:
    return Chunk(data=data, index=index, is_last=is_last, source_schema=schema)


def _make_stream(id_batches: list[list[int]], schema: Schema) -> list[Chunk]:
    return [
        _chunk(
            pl.DataFrame({"id": ids}, schema={"id": pl.Int64}),
            schema,
            index=i,
            is_last=(i == len(id_batches) - 1),
        )
        for i, ids in enumerate(id_batches)
    ]


SCHEMA = Schema(columns={"id": pl.Int64()})


def test_export_csv_writes_the_file_when_validation_passes(tmp_path: Path) -> None:
    original = _make_stream([[1, 2], [3, 4]], SCHEMA)
    optimized = _make_stream([[1, 2], [3, 4]], SCHEMA)
    output_path = tmp_path / "out.csv"

    result = Exporter().export(
        iter(original),
        iter(optimized),
        OptimizationMode.LOSSLESS,
        output_path,
        ExportFormat.CSV,
    )

    assert result.passed is True
    assert output_path.exists()
    assert pl.read_csv(output_path)["id"].to_list() == [1, 2, 3, 4]


def test_export_parquet_writes_the_file_when_validation_passes(tmp_path: Path) -> None:
    original = _make_stream([[1, 2], [3, 4]], SCHEMA)
    optimized = _make_stream([[1, 2], [3, 4]], SCHEMA)
    output_path = tmp_path / "out.parquet"

    result = Exporter().export(
        iter(original),
        iter(optimized),
        OptimizationMode.LOSSLESS,
        output_path,
        ExportFormat.PARQUET,
    )

    assert result.passed is True
    assert output_path.exists()
    assert pl.read_parquet(output_path)["id"].to_list() == [1, 2, 3, 4]


def test_export_json_writes_the_file_when_validation_passes(tmp_path: Path) -> None:
    original = _make_stream([[1, 2], [3, 4]], SCHEMA)
    optimized = _make_stream([[1, 2], [3, 4]], SCHEMA)
    output_path = tmp_path / "out.json"

    result = Exporter().export(
        iter(original),
        iter(optimized),
        OptimizationMode.LOSSLESS,
        output_path,
        ExportFormat.JSON,
    )

    assert result.passed is True
    assert output_path.exists()
    assert pl.read_ndjson(output_path)["id"].to_list() == [1, 2, 3, 4]


def test_export_refuses_to_write_and_leaves_no_output_when_validation_fails(
    tmp_path: Path,
) -> None:
    """Invariant I3: a failed validation aborts export, reports why, and
    writes nothing - not even a partial or temporary file."""
    original = _make_stream([[1, 2], [3, 4]], SCHEMA)
    optimized = _make_stream([[1, 2], [999, 4]], SCHEMA)  # value changed
    output_path = tmp_path / "out.csv"

    with pytest.raises(ExportError, match="validation failed"):
        Exporter().export(
            iter(original),
            iter(optimized),
            OptimizationMode.LOSSLESS,
            output_path,
            ExportFormat.CSV,
        )

    assert not output_path.exists()
    assert list(tmp_path.iterdir()) == []  # no stray temp file either


def test_export_does_not_leave_a_temp_file_behind_after_success(
    tmp_path: Path,
) -> None:
    original = _make_stream([[1, 2]], SCHEMA)
    optimized = _make_stream([[1, 2]], SCHEMA)
    output_path = tmp_path / "out.csv"

    Exporter().export(
        iter(original),
        iter(optimized),
        OptimizationMode.LOSSLESS,
        output_path,
        ExportFormat.CSV,
    )

    assert list(tmp_path.iterdir()) == [output_path]


@pytest.mark.parametrize(
    ("export_format", "reader", "suffix"),
    [
        (ExportFormat.CSV, pl.read_csv, "csv"),
        (ExportFormat.PARQUET, pl.read_parquet, "parquet"),
        (ExportFormat.JSON, pl.read_ndjson, "json"),
    ],
)
def test_export_streams_multiple_chunks_correctly_for_each_format(
    tmp_path: Path,
    export_format: ExportFormat,
    reader: object,
    suffix: str,
) -> None:
    output_path = tmp_path / f"out.{suffix}"
    original = _make_stream([[1, 2], [3, 4], [5]], SCHEMA)
    optimized = _make_stream([[1, 2], [3, 4], [5]], SCHEMA)

    result = Exporter().export(
        iter(original),
        iter(optimized),
        OptimizationMode.LOSSLESS,
        output_path,
        export_format,
    )

    assert result.passed is True
    assert reader(output_path)["id"].to_list() == [1, 2, 3, 4, 5]  # type: ignore[operator]


def test_export_parquet_with_compress_uses_zstd_compression(tmp_path: Path) -> None:
    """Parquet previously left PyArrow's compression codec unconfigured
    (defaulting to Snappy); it now requests Zstandard explicitly when
    compress=True - verified here by reading the codec back out of the
    file's own metadata, not just checking the file is smaller."""
    original = _make_stream([[1, 2], [3, 4]], SCHEMA)
    optimized = _make_stream([[1, 2], [3, 4]], SCHEMA)
    output_path = tmp_path / "out.parquet"

    Exporter().export(
        iter(original),
        iter(optimized),
        OptimizationMode.LOSSLESS,
        output_path,
        ExportFormat.PARQUET,
        compress=True,
    )

    metadata = pq.ParquetFile(output_path).metadata
    assert metadata.row_group(0).column(0).compression == "ZSTD"


def test_export_parquet_without_compress_uses_no_codec(tmp_path: Path) -> None:
    """compress=False must mean genuinely zero compression for Parquet
    too, not just for CSV/JSON - otherwise there is no way to isolate the
    effect of the optimizers alone from the effect of compression."""
    original = _make_stream([[1, 2], [3, 4]], SCHEMA)
    optimized = _make_stream([[1, 2], [3, 4]], SCHEMA)
    output_path = tmp_path / "out.parquet"

    Exporter().export(
        iter(original),
        iter(optimized),
        OptimizationMode.LOSSLESS,
        output_path,
        ExportFormat.PARQUET,
    )

    metadata = pq.ParquetFile(output_path).metadata
    assert metadata.row_group(0).column(0).compression == "UNCOMPRESSED"


def test_export_csv_with_compress_produces_a_zstd_stream_that_decompresses_correctly(
    tmp_path: Path,
) -> None:
    original = _make_stream([[1, 2], [3, 4], [5]], SCHEMA)
    optimized = _make_stream([[1, 2], [3, 4], [5]], SCHEMA)
    output_path = tmp_path / "out.csv.zst"

    Exporter().export(
        iter(original),
        iter(optimized),
        OptimizationMode.LOSSLESS,
        output_path,
        ExportFormat.CSV,
        compress=True,
    )

    with (
        open(output_path, "rb") as raw,
        zstandard.ZstdDecompressor().stream_reader(raw) as reader,
    ):
        readback = pl.read_csv(reader)
    assert readback["id"].to_list() == [1, 2, 3, 4, 5]

    # The file's own magic bytes confirm it's a real Zstandard frame
    # (independent of any particular reader's behavior).
    assert output_path.read_bytes()[:4] == b"\x28\xb5\x2f\xfd"

    # Bonus, verified rather than assumed: Polars' own CSV reader
    # auto-detects Zstandard's magic bytes and transparently decompresses,
    # so plain pl.read_csv() also works directly on the compressed file -
    # no manual decompression required by downstream tooling.
    assert pl.read_csv(output_path)["id"].to_list() == [1, 2, 3, 4, 5]


def test_export_json_with_compress_produces_a_zstd_stream_that_decompresses_correctly(
    tmp_path: Path,
) -> None:
    original = _make_stream([[1, 2], [3, 4]], SCHEMA)
    optimized = _make_stream([[1, 2], [3, 4]], SCHEMA)
    output_path = tmp_path / "out.json.zst"

    Exporter().export(
        iter(original),
        iter(optimized),
        OptimizationMode.LOSSLESS,
        output_path,
        ExportFormat.JSON,
        compress=True,
    )

    with (
        open(output_path, "rb") as raw,
        zstandard.ZstdDecompressor().stream_reader(raw) as reader,
    ):
        readback = pl.read_ndjson(reader)
    assert readback["id"].to_list() == [1, 2, 3, 4]


def test_export_without_compress_produces_plain_uncompressed_output(
    tmp_path: Path,
) -> None:
    """compress defaults to False: existing behavior (plain text CSV) is
    unchanged unless a caller explicitly opts in."""
    original = _make_stream([[1, 2]], SCHEMA)
    optimized = _make_stream([[1, 2]], SCHEMA)
    output_path = tmp_path / "out.csv"

    Exporter().export(
        iter(original),
        iter(optimized),
        OptimizationMode.LOSSLESS,
        output_path,
        ExportFormat.CSV,
    )

    # Readable directly as plain CSV text - no decompression needed.
    assert pl.read_csv(output_path)["id"].to_list() == [1, 2]


def test_from_suffix_recognizes_each_supported_format(tmp_path: Path) -> None:
    assert ExportFormat.from_suffix(tmp_path / "out.csv") is ExportFormat.CSV
    assert ExportFormat.from_suffix(tmp_path / "out.parquet") is ExportFormat.PARQUET
    assert ExportFormat.from_suffix(tmp_path / "out.json") is ExportFormat.JSON


def test_from_suffix_treats_jsonl_the_same_as_json(tmp_path: Path) -> None:
    assert ExportFormat.from_suffix(tmp_path / "out.jsonl") is ExportFormat.JSON


def test_from_suffix_strips_a_trailing_zst_first(tmp_path: Path) -> None:
    assert ExportFormat.from_suffix(tmp_path / "out.csv.zst") is ExportFormat.CSV


def test_from_suffix_returns_none_for_an_unrecognized_extension(tmp_path: Path) -> None:
    assert ExportFormat.from_suffix(tmp_path / "out.xlsx") is None


def test_export_raises_when_original_and_optimized_streams_have_different_chunk_counts(
    tmp_path: Path,
) -> None:
    original = _make_stream([[1, 2], [3, 4]], SCHEMA)
    optimized = _make_stream([[1, 2]], SCHEMA)  # one chunk short
    output_path = tmp_path / "out.csv"

    with pytest.raises(ValueError):
        Exporter().export(
            iter(original),
            iter(optimized),
            OptimizationMode.LOSSLESS,
            output_path,
            ExportFormat.CSV,
        )

    assert not output_path.exists()
    assert list(tmp_path.iterdir()) == []
