import polars as pl

from core.mode import OptimizationMode
from core.types import Chunk, Schema
from engine.validator import Validator


def _chunk(data: pl.DataFrame, schema: Schema, index: int = 0) -> Chunk:
    return Chunk(data=data, index=index, is_last=True, source_schema=schema)


def test_passes_for_identical_streams_in_lossless_mode() -> None:
    schema = Schema(columns={"id": pl.Int64()})
    data = pl.DataFrame({"id": [1, 2, 3]}, schema={"id": pl.Int64})

    validator = Validator(OptimizationMode.LOSSLESS)
    validator.observe(_chunk(data, schema), _chunk(data.clone(), schema))
    result = validator.finalize()

    assert result.passed is True
    assert result.reasons == []
    assert result.original_row_count == 3
    assert result.optimized_row_count == 3


def test_lossless_mode_passes_for_an_exact_downcast() -> None:
    schema = Schema(columns={"id": pl.Int64()})
    original_data = pl.DataFrame({"id": [1, 2, 3]}, schema={"id": pl.Int64})
    optimized_data = original_data.with_columns(pl.col("id").cast(pl.Int8))

    validator = Validator(OptimizationMode.LOSSLESS)
    validator.observe(_chunk(original_data, schema), _chunk(optimized_data, schema))
    result = validator.finalize()

    assert result.passed is True
    assert result.reasons == []


def test_lossless_mode_fails_when_a_value_actually_changes() -> None:
    schema = Schema(columns={"id": pl.Int64()})
    original_data = pl.DataFrame({"id": [1, 2, 3]}, schema={"id": pl.Int64})
    optimized_data = pl.DataFrame({"id": [1, 2, 999]}, schema={"id": pl.Int64})

    validator = Validator(OptimizationMode.LOSSLESS)
    validator.observe(_chunk(original_data, schema), _chunk(optimized_data, schema))
    result = validator.finalize()

    assert result.passed is False
    assert any("does not reconstruct exactly" in reason for reason in result.reasons)


def test_balanced_mode_ignores_value_changes_but_still_checks_row_count() -> None:
    schema = Schema(columns={"id": pl.Int64()})
    original_data = pl.DataFrame({"id": [1, 2, 3]}, schema={"id": pl.Int64})
    # Value changed - allowed under balanced mode, unlike lossless.
    optimized_data = pl.DataFrame({"id": [1, 2, 999]}, schema={"id": pl.Int64})

    validator = Validator(OptimizationMode.BALANCED)
    validator.observe(_chunk(original_data, schema), _chunk(optimized_data, schema))
    result = validator.finalize()

    assert result.passed is True
    assert result.reasons == []


def test_detects_column_mismatch() -> None:
    schema = Schema(columns={"id": pl.Int64(), "name": pl.String()})
    original_data = pl.DataFrame({"id": [1], "name": ["a"]})
    optimized_data = pl.DataFrame({"id": [1]})  # "name" dropped

    validator = Validator(OptimizationMode.LOSSLESS)
    validator.observe(_chunk(original_data, schema), _chunk(optimized_data, schema))
    result = validator.finalize()

    assert result.passed is False
    assert any("column mismatch" in reason for reason in result.reasons)


def test_detects_row_count_mismatch_within_a_chunk() -> None:
    schema = Schema(columns={"id": pl.Int64()})
    original_data = pl.DataFrame({"id": [1, 2, 3]}, schema={"id": pl.Int64})
    optimized_data = pl.DataFrame(
        {"id": [1, 2]}, schema={"id": pl.Int64}
    )  # row dropped

    validator = Validator(OptimizationMode.LOSSLESS)
    validator.observe(_chunk(original_data, schema), _chunk(optimized_data, schema))
    result = validator.finalize()

    assert result.passed is False
    assert any("row count mismatch" in reason for reason in result.reasons)


def test_detects_null_count_change_regardless_of_mode() -> None:
    schema = Schema(columns={"id": pl.Int64()})
    original_data = pl.DataFrame({"id": [1, None, 3]}, schema={"id": pl.Int64})
    optimized_data = pl.DataFrame(
        {"id": [1, 2, 3]}, schema={"id": pl.Int64}
    )  # null filled in

    # Aggressive mode: value changes are allowed, but a null silently
    # becoming non-null (or vice versa) is a correctness bug, not a
    # sanctioned precision trade-off, so this must still fail.
    validator = Validator(OptimizationMode.AGGRESSIVE)
    validator.observe(_chunk(original_data, schema), _chunk(optimized_data, schema))
    result = validator.finalize()

    assert result.passed is False
    assert any("null count changed" in reason for reason in result.reasons)


def test_accumulates_row_counts_across_multiple_chunks() -> None:
    schema = Schema(columns={"id": pl.Int64()})
    validator = Validator(OptimizationMode.LOSSLESS)

    for i in range(3):
        data = pl.DataFrame({"id": [i * 10, i * 10 + 1]}, schema={"id": pl.Int64})
        validator.observe(
            _chunk(data, schema, index=i), _chunk(data.clone(), schema, index=i)
        )

    result = validator.finalize()

    assert result.passed is True
    assert result.original_row_count == 6
    assert result.optimized_row_count == 6


def test_finalize_with_no_chunks_observed_passes_trivially() -> None:
    result = Validator(OptimizationMode.LOSSLESS).finalize()

    assert result.passed is True
    assert result.reasons == []
    assert result.original_row_count == 0
    assert result.optimized_row_count == 0
