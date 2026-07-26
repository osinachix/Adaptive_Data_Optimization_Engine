import ast
from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest
import structlog
from structlog.testing import capture_logs

import engine.pipeline as pipeline_module
from core.mode import OptimizationMode
from engine.exporter import ExportFormat
from engine.pipeline import run_optimize, run_profile, run_validate
from optimizers.numeric_downcast import NumericDowncastOptimizer
from plugins.registry import OptimizerRegistry


@pytest.fixture(autouse=True)
def _reset_structlog_config() -> Iterator[None]:
    """Other test modules (e.g. test_cli.py) call configure_logging(),
    which sets process-global structlog state with
    cache_logger_on_first_use=True. That leaks into these tests
    regardless of file/collection order and breaks capture_logs() (logs
    go to the real PrintLogger instead of being captured). Reset to
    structlog's own defaults before and after every test here so these
    tests behave the same regardless of what ran before them."""
    structlog.reset_defaults()
    yield
    structlog.reset_defaults()


def _write_csv(path: Path, rows: int) -> list[int]:
    ages = [i % 90 for i in range(rows)]
    pl.DataFrame({"age": ages}, schema={"age": pl.Int64}).write_csv(path)
    return ages


# --- run_profile --------------------------------------------------------


def test_run_profile_produces_expected_stats(tmp_path: Path) -> None:
    csv_path = tmp_path / "input.csv"
    _write_csv(csv_path, 500)

    outcome = run_profile(csv_path, rows_per_chunk=100)

    assert outcome.row_count == 500
    assert outcome.dataset_profile.columns["age"].count == 500
    assert outcome.schema_analysis.columns["age"].value == "integer"
    assert "age" in outcome.optimization_profile.columns


def test_run_profile_raises_a_clear_error_for_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_profile(tmp_path / "missing.csv", rows_per_chunk=100)


# --- run_optimize --------------------------------------------------------


def test_run_optimize_produces_output_and_report(tmp_path: Path) -> None:
    csv_path = tmp_path / "input.csv"
    ages = _write_csv(csv_path, 1_000)
    output_path = tmp_path / "output.parquet"
    registry = OptimizerRegistry()
    registry.register(NumericDowncastOptimizer())

    outcome = run_optimize(
        csv_path,
        output_path,
        ExportFormat.PARQUET,
        OptimizationMode.LOSSLESS,
        200,
        registry,
    )

    assert output_path.exists()
    assert outcome.report.validation.passed is True
    assert len(outcome.plan.steps) == 1
    readback = pl.read_parquet(output_path)
    assert readback["age"].dtype == pl.Int8()
    assert readback["age"].to_list() == ages


def test_run_optimize_with_empty_registry_still_exports_data_unchanged(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "input.csv"
    ages = _write_csv(csv_path, 200)
    output_path = tmp_path / "output.csv"

    outcome = run_optimize(
        csv_path,
        output_path,
        ExportFormat.CSV,
        OptimizationMode.LOSSLESS,
        50,
        OptimizerRegistry(),
    )

    assert outcome.plan.steps == []
    assert outcome.report.validation.passed is True
    assert pl.read_csv(output_path)["age"].to_list() == ages


def test_run_optimize_populates_real_file_bytes_in_the_report(tmp_path: Path) -> None:
    csv_path = tmp_path / "input.csv"
    _write_csv(csv_path, 500)
    output_path = tmp_path / "output.parquet"

    outcome = run_optimize(
        csv_path,
        output_path,
        ExportFormat.PARQUET,
        OptimizationMode.LOSSLESS,
        100,
        OptimizerRegistry(),
    )

    assert outcome.report.input_file_bytes == csv_path.stat().st_size
    assert outcome.report.output_file_bytes == output_path.stat().st_size


def test_run_optimize_with_compress_writes_a_zstd_file_readable_end_to_end(
    tmp_path: Path,
) -> None:
    """Completes the round trip for the compression feature: optimize
    writes a Zstandard-compressed CSV, and run_validate (via
    ReaderFactory, which strips the .zst before dispatch) can read it
    straight back for independent re-validation - no special-casing
    needed by the caller."""
    csv_path = tmp_path / "input.csv"
    ages = _write_csv(csv_path, 500)
    output_path = tmp_path / "output.csv.zst"
    registry = OptimizerRegistry()
    registry.register(NumericDowncastOptimizer())

    outcome = run_optimize(
        csv_path,
        output_path,
        ExportFormat.CSV,
        OptimizationMode.LOSSLESS,
        100,
        registry,
        compress=True,
    )

    assert outcome.report.validation.passed is True
    assert output_path.read_bytes()[:4] == b"\x28\xb5\x2f\xfd"  # real zstd frame
    assert pl.read_csv(output_path)["age"].to_list() == ages  # Polars auto-decompresses

    revalidation = run_validate(csv_path, output_path, OptimizationMode.LOSSLESS, 100)
    assert revalidation.passed is True


# --- run_validate --------------------------------------------------------


def test_run_validate_passes_for_a_freshly_optimized_output(tmp_path: Path) -> None:
    csv_path = tmp_path / "input.csv"
    _write_csv(csv_path, 300)
    output_path = tmp_path / "output.parquet"
    registry = OptimizerRegistry()
    registry.register(NumericDowncastOptimizer())
    run_optimize(
        csv_path,
        output_path,
        ExportFormat.PARQUET,
        OptimizationMode.LOSSLESS,
        100,
        registry,
    )

    result = run_validate(csv_path, output_path, OptimizationMode.LOSSLESS, 100)

    assert result.passed is True


def test_run_validate_fails_for_mismatched_files(tmp_path: Path) -> None:
    csv_path = tmp_path / "input.csv"
    _write_csv(csv_path, 300)
    other_path = tmp_path / "other.csv"
    pl.DataFrame({"age": [1, 2, 3]}, schema={"age": pl.Int64}).write_csv(other_path)

    with pytest.raises(ValueError):  # zip(strict=True): chunk counts don't line up
        run_validate(csv_path, other_path, OptimizationMode.LOSSLESS, 100)


# --- structured logging ---------------------------------------------------


def test_run_profile_logs_execution_id_dataset_metadata_timing_and_memory(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "input.csv"
    _write_csv(csv_path, 200)

    with capture_logs() as logs:
        run_profile(csv_path, rows_per_chunk=50)

    events = {entry["event"]: entry for entry in logs}
    assert "profile.started" in events
    assert "profile.completed" in events

    started, completed = events["profile.started"], events["profile.completed"]
    assert started["execution_id"] == completed["execution_id"]
    assert completed["input_path"] == str(csv_path)
    assert completed["row_count"] == 200
    assert "elapsed_seconds" in completed
    assert "peak_memory_bytes" in completed


def test_run_optimize_logs_optimizer_activity(tmp_path: Path) -> None:
    csv_path = tmp_path / "input.csv"
    _write_csv(csv_path, 300)
    output_path = tmp_path / "output.parquet"
    registry = OptimizerRegistry()
    registry.register(NumericDowncastOptimizer())

    with capture_logs() as logs:
        run_optimize(
            csv_path,
            output_path,
            ExportFormat.PARQUET,
            OptimizationMode.LOSSLESS,
            100,
            registry,
        )

    events = [entry["event"] for entry in logs]
    assert "optimize.planned" in events
    assert "optimize.chunk_optimized" in events
    assert "optimize.completed" in events


def test_run_validate_logs_a_warning_when_validation_fails(tmp_path: Path) -> None:
    csv_path = tmp_path / "input.csv"
    _write_csv(csv_path, 100)
    other_path = tmp_path / "other.csv"
    pl.DataFrame({"age": [999] * 100}, schema={"age": pl.Int64}).write_csv(other_path)

    with capture_logs() as logs:
        result = run_validate(csv_path, other_path, OptimizationMode.LOSSLESS, 100)

    assert result.passed is False
    warnings = [entry for entry in logs if entry["log_level"] == "warning"]
    assert any(entry["event"] == "validate.failed_checks" for entry in warnings)


def test_run_profile_logs_an_error_on_failure(tmp_path: Path) -> None:
    with capture_logs() as logs, pytest.raises(FileNotFoundError):
        run_profile(tmp_path / "missing.csv", rows_per_chunk=50)

    errors = [entry for entry in logs if entry["log_level"] == "error"]
    assert any(entry["event"] == "profile.failed" for entry in errors)


# --- invariant I6: engine works with no CLI present ------------------------


def test_engine_pipeline_module_never_imports_cli_or_typer() -> None:
    """Invariant I6: the core engine must remain importable and runnable
    with no CLI present, so engine.pipeline - the module the CLI calls
    into - must never itself import the cli package or the Typer
    framework. Checked via the module's actual import statements (AST),
    not via a runtime sys.modules check, since the latter would be
    order-dependent on whatever other test files already imported in the
    same pytest process."""
    tree = ast.parse(Path(pipeline_module.__file__).read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])

    assert "cli" not in imported_roots
    assert "typer" not in imported_roots
