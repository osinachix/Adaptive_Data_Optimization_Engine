"""Benchmark harness (M11): runs the real optimize pipeline
(engine.pipeline.run_optimize - the same function the CLI calls, per
invariant I6) across small/medium/large synthetic datasets and measures
execution time, peak memory, dataset size reduction, throughput
(rows/sec), and validation success rate.

Run directly:

    poetry run python benchmarks/harness.py

Generated datasets live under benchmarks/datasets/ (gitignored, regenerated
on each run from a fixed seed for reproducibility). Results are printed as
a table and written to benchmarks/baseline_results.json.
"""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl
from structlog.testing import capture_logs

from core.mode import OptimizationMode
from engine.exporter import ExportFormat
from engine.pipeline import run_optimize
from optimizers.defaults import build_default_registry

_BENCHMARK_DIR = Path(__file__).parent
_DATASET_DIR = _BENCHMARK_DIR / "datasets"
_RESULTS_PATH = _BENCHMARK_DIR / "baseline_results.json"

# Fixed seed: datasets are reproducible byte-for-byte across runs, so
# results are comparable over time (regression detection).
_SEED = 20_260_101
_SIZES: dict[str, int] = {"small": 10_000, "medium": 200_000, "large": 1_000_000}
_ROWS_PER_CHUNK = 10_000


@dataclass(frozen=True)
class BenchmarkResult:
    """Metrics for one dataset-size run."""

    name: str
    row_count: int
    elapsed_seconds: float
    peak_memory_bytes: int
    input_file_bytes: int
    output_file_bytes: int
    size_reduction_percent: float
    rows_per_second: float
    validation_passed: bool


def _make_dataset(path: Path, rows: int, seed: int) -> None:
    """Generate a reproducible synthetic dataset shaped like a typical
    tabular export: a sequential id, small-range integers, and
    low-cardinality strings - the same shape numeric_downcast and
    dictionary_encoding are designed around."""
    rng = random.Random(seed)
    regions = ["north", "south", "east", "west"]
    statuses = ["pending", "shipped", "delivered", "cancelled"]
    pl.DataFrame(
        {
            "order_id": list(range(1, rows + 1)),
            "customer_age": [rng.randint(18, 90) for _ in range(rows)],
            "quantity": [rng.randint(1, 20) for _ in range(rows)],
            "region": [rng.choice(regions) for _ in range(rows)],
            "status": [rng.choice(statuses) for _ in range(rows)],
        },
        schema={
            "order_id": pl.Int64,
            "customer_age": pl.Int64,
            "quantity": pl.Int64,
            "region": pl.String,
            "status": pl.String,
        },
    ).write_csv(path)


def run_benchmark(
    name: str,
    rows: int,
    rows_per_chunk: int = _ROWS_PER_CHUNK,
    dataset_dir: Path | None = None,
) -> BenchmarkResult:
    """Generate a dataset of the given size and run it through the real
    optimize pipeline, measuring the benchmark metrics. Timing and peak
    memory are read from the pipeline's own structured log output (via
    capture_logs()) rather than wrapped in a second tracemalloc session:
    run_optimize() already manages tracemalloc internally, and nesting a
    second start/stop pair around it would stop tracing prematurely for
    whichever call unwinds first - this reuses the same measurement
    run_optimize() logs for its own "optimize.completed" event instead of
    conflicting with it.

    dataset_dir defaults to benchmarks/datasets/; tests pass tmp_path so
    smoke-testing this function doesn't leave stray files next to the
    real recorded baseline's datasets.
    """
    resolved_dataset_dir = dataset_dir if dataset_dir is not None else _DATASET_DIR
    resolved_dataset_dir.mkdir(parents=True, exist_ok=True)
    input_path = resolved_dataset_dir / f"{name}.csv"
    output_path = resolved_dataset_dir / f"{name}.parquet"
    _make_dataset(input_path, rows, seed=_SEED)

    with capture_logs() as logs:
        outcome = run_optimize(
            input_path,
            output_path,
            ExportFormat.PARQUET,
            OptimizationMode.LOSSLESS,
            rows_per_chunk,
            build_default_registry(),
        )

    completed = next(entry for entry in logs if entry["event"] == "optimize.completed")
    input_bytes = input_path.stat().st_size
    output_bytes = output_path.stat().st_size
    row_count = outcome.report.validation.original_row_count
    elapsed_seconds = float(completed["elapsed_seconds"])

    return BenchmarkResult(
        name=name,
        row_count=row_count,
        elapsed_seconds=elapsed_seconds,
        peak_memory_bytes=int(completed["peak_memory_bytes"]),
        input_file_bytes=input_bytes,
        output_file_bytes=output_bytes,
        size_reduction_percent=round((1 - output_bytes / input_bytes) * 100, 2),
        rows_per_second=(
            round(row_count / elapsed_seconds, 1)
            if elapsed_seconds > 0
            else float("inf")
        ),
        validation_passed=outcome.report.validation.passed,
    )


def run_all(sizes: dict[str, int] | None = None) -> list[BenchmarkResult]:
    """Run the benchmark suite across all configured dataset sizes."""
    resolved_sizes = sizes if sizes is not None else _SIZES
    return [run_benchmark(name, rows) for name, rows in resolved_sizes.items()]


def _print_report(results: list[BenchmarkResult]) -> None:
    passed = sum(1 for result in results if result.validation_passed)
    header = (
        f"{'name':<8} {'rows':>10} {'time(s)':>9} {'peak MB':>9} "
        f"{'rows/s':>10} {'size cut':>8}  validated"
    )
    print(header)
    for result in results:
        print(
            f"{result.name:<8} {result.row_count:>10} "
            f"{result.elapsed_seconds:>9.2f} "
            f"{result.peak_memory_bytes / 1e6:>9.2f} "
            f"{result.rows_per_second:>10.0f} "
            f"{result.size_reduction_percent:>7.1f}%  "
            f"{'PASS' if result.validation_passed else 'FAIL'}"
        )
    rate = passed / len(results) * 100 if results else 0.0
    print(f"\nValidation success rate: {passed}/{len(results)} ({rate:.0f}%)")


def main() -> None:
    results = run_all()
    _print_report(results)
    _RESULTS_PATH.write_text(
        json.dumps([asdict(result) for result in results], indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nBaseline results written to {_RESULTS_PATH}")


if __name__ == "__main__":
    main()
