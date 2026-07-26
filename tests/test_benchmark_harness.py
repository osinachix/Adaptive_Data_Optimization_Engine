import dataclasses
import json
from pathlib import Path

from benchmarks.harness import run_benchmark


def test_run_benchmark_produces_sane_metrics_for_a_tiny_dataset(tmp_path: Path) -> None:
    """A fast smoke test with a tiny row count - not a substitute for
    actually running the harness (poetry run python benchmarks/harness.py)
    to record real small/medium/large baseline numbers, just a check that
    run_benchmark() itself works and reports sane values."""
    result = run_benchmark("smoke", rows=500, rows_per_chunk=100, dataset_dir=tmp_path)

    assert result.row_count == 500
    assert result.elapsed_seconds > 0
    assert result.peak_memory_bytes > 0
    assert result.input_file_bytes > 0
    assert result.output_file_bytes > 0
    assert result.rows_per_second > 0
    assert result.validation_passed is True
    # size_reduction_percent can be negative for a tiny file (Parquet's
    # own overhead can outweigh savings at this scale - see the dictionary
    # encoding optimizer's own docstring on fixed per-row overhead), so
    # this only checks the field is a real, finite number.
    assert isinstance(result.size_reduction_percent, float)


def test_benchmark_result_is_json_serializable(tmp_path: Path) -> None:
    """Confirms the dataclass that main() writes to baseline_results.json
    round-trips through json.dumps() cleanly (no non-serializable field
    types crept in)."""
    result = run_benchmark(
        "smoke_json", rows=200, rows_per_chunk=50, dataset_dir=tmp_path
    )

    serialized = json.dumps(dataclasses.asdict(result))
    reloaded = json.loads(serialized)

    assert reloaded["row_count"] == 200
    assert reloaded["validation_passed"] is True
