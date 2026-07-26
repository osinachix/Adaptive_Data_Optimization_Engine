from pathlib import Path

import polars as pl
import pytest

from core.chunk_manager import ChunkManager
from core.mode import OptimizationMode
from core.readers.json_reader import JSONReader
from core.types import Chunk, Schema
from engine.exporter import ExportFormat, Exporter

EXPECTED_SCHEMA = Schema(columns={"id": pl.Int64(), "name": pl.String()})


def _write_ndjson(path: Path, rows: int) -> pl.DataFrame:
    df = pl.DataFrame({"id": range(rows), "name": [f"row-{i}" for i in range(rows)]})
    df.write_ndjson(path)
    return df


def test_schema_raises_before_open(tmp_path: Path) -> None:
    reader = JSONReader(tmp_path / "data.jsonl")

    with pytest.raises(RuntimeError):
        reader.schema()


def test_read_batches_raises_before_open(tmp_path: Path) -> None:
    reader = JSONReader(tmp_path / "data.jsonl")

    with pytest.raises(RuntimeError):
        next(reader.read_batches(10))


def test_open_raises_for_missing_file(tmp_path: Path) -> None:
    reader = JSONReader(tmp_path / "missing.jsonl")

    with pytest.raises(FileNotFoundError):
        reader.open()


def test_open_establishes_schema(tmp_path: Path) -> None:
    json_path = tmp_path / "data.jsonl"
    _write_ndjson(json_path, rows=5)
    reader = JSONReader(json_path)

    reader.open()

    assert reader.schema() == EXPECTED_SCHEMA


def test_read_batches_respects_rows_per_batch(tmp_path: Path) -> None:
    json_path = tmp_path / "data.jsonl"
    _write_ndjson(json_path, rows=10_000)
    reader = JSONReader(json_path)
    reader.open()

    sizes = [batch.height for batch in reader.read_batches(3_000)]

    assert sizes == [3_000, 3_000, 3_000, 1_000]


def test_read_batches_preserves_all_rows_without_loss_or_duplication(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "data.jsonl"
    expected = _write_ndjson(json_path, rows=2_500)
    reader = JSONReader(json_path)
    reader.open()

    combined = pl.concat(list(reader.read_batches(700)))

    assert combined.equals(expected)


def test_close_is_idempotent_and_resets_state(tmp_path: Path) -> None:
    json_path = tmp_path / "data.jsonl"
    _write_ndjson(json_path, rows=5)
    reader = JSONReader(json_path)
    reader.open()

    reader.close()
    reader.close()  # must not raise

    with pytest.raises(RuntimeError):
        reader.schema()


def test_chunk_manager_streams_json_reader_end_to_end(tmp_path: Path) -> None:
    json_path = tmp_path / "data.jsonl"
    expected = _write_ndjson(json_path, rows=2_200)
    manager = ChunkManager(JSONReader(json_path), rows_per_chunk=500)

    chunks = list(manager.stream())

    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert [chunk.is_last for chunk in chunks] == [False] * (len(chunks) - 1) + [True]
    assert all(chunk.source_schema == EXPECTED_SCHEMA for chunk in chunks)

    combined = pl.concat([chunk.data for chunk in chunks])
    assert combined.equals(expected)


def test_reads_back_adoe_exported_ndjson_round_trip(tmp_path: Path) -> None:
    """JSONReader must be able to read exactly what ExportFormat.JSON
    writes (write_ndjson), so an exported .json file can be re-profiled
    or re-validated without special-casing."""
    schema = Schema(columns={"id": pl.Int64()})
    chunk = Chunk(
        data=pl.DataFrame({"id": [1, 2, 3]}, schema={"id": pl.Int64}),
        index=0,
        is_last=True,
        source_schema=schema,
    )
    output_path = tmp_path / "out.json"
    Exporter().export(
        iter([chunk]), iter([chunk]), OptimizationMode.LOSSLESS, output_path,
        ExportFormat.JSON,
    )

    reader = JSONReader(output_path)
    reader.open()
    combined = pl.concat(list(reader.read_batches(100)))

    assert combined["id"].to_list() == [1, 2, 3]
