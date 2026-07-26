"""Stress tests for large streaming workloads (M11). These exercise
multi-million-row datasets through the full pipeline and take much
longer than the rest of the suite, so they're skipped by default - see
tests/conftest.py. Run them explicitly with:

    poetry run pytest --run-stress tests/test_stress.py -v
"""

import gc
from pathlib import Path

import polars as pl
import pytest
from structlog.testing import capture_logs

from core.mode import OptimizationMode
from engine.exporter import ExportFormat
from engine.pipeline import run_optimize
from optimizers.dictionary_encoding import DictionaryEncodingOptimizer
from optimizers.numeric_downcast import NumericDowncastOptimizer
from plugins.registry import OptimizerRegistry


def _write_large_csv(path: Path, rows: int) -> None:
    regions = ["north", "south", "east", "west"]
    pl.DataFrame(
        {
            "id": list(range(rows)),
            "value": [i % 90 for i in range(rows)],
            "region": [regions[i % len(regions)] for i in range(rows)],
        },
        schema={"id": pl.Int64, "value": pl.Int64, "region": pl.String},
    ).write_csv(path)


def _registry() -> OptimizerRegistry:
    registry = OptimizerRegistry()
    registry.register(NumericDowncastOptimizer())
    registry.register(DictionaryEncodingOptimizer())
    return registry


def _optimize_and_get_peak_memory(
    input_path: Path, output_path: Path, rows_per_chunk: int
) -> tuple[int, bool, int]:
    """Run the real optimize pipeline and read back its own timing/memory
    measurement via structured logs, rather than wrapping a second
    tracemalloc session around it (run_optimize() already manages
    tracemalloc internally; nesting a second start/stop pair around it
    would stop tracing prematurely for whichever call unwinds first - see
    benchmarks/harness.py, which uses the same approach)."""
    with capture_logs() as logs:
        outcome = run_optimize(
            input_path,
            output_path,
            ExportFormat.PARQUET,
            OptimizationMode.LOSSLESS,
            rows_per_chunk,
            _registry(),
        )
    completed = next(entry for entry in logs if entry["event"] == "optimize.completed")
    return (
        int(completed["peak_memory_bytes"]),
        outcome.report.validation.passed,
        outcome.report.validation.original_row_count,
    )


@pytest.mark.stress
def test_large_streaming_workload_completes_and_validates_losslessly(
    tmp_path: Path,
) -> None:
    """A dataset far larger than anything used in the regular unit tests
    must still stream through the full pipeline correctly: profile, plan,
    optimize, validate, export - and reconstruct exactly in lossless
    mode."""
    rows = 1_000_000
    csv_path = tmp_path / "large.csv"
    output_path = tmp_path / "large.parquet"
    _write_large_csv(csv_path, rows)

    outcome = run_optimize(
        csv_path,
        output_path,
        ExportFormat.PARQUET,
        OptimizationMode.LOSSLESS,
        50_000,
        _registry(),
    )

    assert outcome.report.validation.passed is True
    assert outcome.report.validation.original_row_count == rows
    assert output_path.exists()

    # Cross-check independently of the tool's own validator: read both
    # files back and compare every column exactly.
    original = pl.read_csv(csv_path)
    optimized = pl.read_parquet(output_path)
    for column in original.columns:
        reconstructed = optimized[column].cast(original[column].dtype)
        assert original[column].equals(reconstructed), column


@pytest.mark.stress
def test_large_streaming_workload_memory_stays_bounded(tmp_path: Path) -> None:
    """Peak memory during a full optimize run must not scale
    proportionally with row count - the point of the streaming
    architecture (invariant I1), now proven at the whole-pipeline level
    (read + profile + plan + apply + validate + export), not just the
    profiler in isolation (see test_profiler.py's equivalent test)."""

    def run(rows: int) -> int:
        gc.collect()
        csv_path = tmp_path / f"data_{rows}.csv"
        output_path = tmp_path / f"data_{rows}.parquet"
        _write_large_csv(csv_path, rows)
        peak, passed, row_count = _optimize_and_get_peak_memory(
            csv_path, output_path, rows_per_chunk=50_000
        )
        assert passed is True
        assert row_count == rows
        return peak

    small_peak = run(20_000)
    large_peak = run(1_000_000)

    # A 50x increase in row count should not translate into anywhere near
    # a 50x increase in peak memory if the full pipeline is genuinely
    # bounded end to end.
    assert large_peak < small_peak * 10
