from pathlib import Path
from typing import Annotated

import polars as pl
import typer
from structlog.testing import capture_logs

from config.logging_setup import configure_logging
from config.settings import ADOESettings
from core.mode import OptimizationMode
from engine.exporter import ExportError, ExportFormat
from engine.pipeline import run_optimize, run_profile, run_validate
from engine.report_io import load_report, save_report
from optimizers.defaults import build_default_registry

app = typer.Typer(
    name="adoe",
    help="Adaptive Data Optimization Engine: profile, optimize, validate, "
    "benchmark, and report on structured datasets.",
)


def _resolve_settings(
    mode: OptimizationMode | None,
    rows_per_chunk: int | None,
    log_level: str | None,
) -> ADOESettings:
    """Build settings from explicit CLI flags (highest priority), falling
    through to environment variables, the adoe.toml config file, and
    finally the built-in defaults. ADOESettings() alone already resolves
    env/toml/defaults; a flag the user actually passed on the command
    line simply overrides that resolved value, so an omitted flag lets
    the lower-priority sources win."""
    resolved = ADOESettings()
    return ADOESettings(
        mode=mode if mode is not None else resolved.mode,
        rows_per_chunk=(
            rows_per_chunk if rows_per_chunk is not None else resolved.rows_per_chunk
        ),
        log_level=log_level if log_level is not None else resolved.log_level,
    )


def _fail(message: str) -> typer.Exit:
    typer.echo(f"Error: {message}", err=True)
    return typer.Exit(code=1)


# Errors raised by reading/streaming a dataset that should surface as a
# clean CLI message, not a raw traceback: a missing file, an unsupported
# format (ReaderFactory), or a malformed row (Polars raises this lazily,
# during streaming, not at open() time - a missing/unsupported file fails
# earlier and separately).
_DATA_ERRORS = (FileNotFoundError, ValueError, pl.exceptions.ComputeError)


ModeOption = Annotated[
    OptimizationMode | None,
    typer.Option(help="Optimization mode. Falls back to config/env if omitted."),
]
RowsPerChunkOption = Annotated[
    int | None,
    typer.Option(help="Rows streamed per chunk. Falls back to config/env if omitted."),
]
LogLevelOption = Annotated[
    str | None,
    typer.Option(help="Log level (DEBUG, INFO, WARNING, ERROR)."),
]


@app.command()
def profile(
    input_path: Annotated[Path, typer.Argument(help="Dataset to profile.")],
    rows_per_chunk: RowsPerChunkOption = None,
    log_level: LogLevelOption = None,
) -> None:
    """Profile a dataset without changing it."""
    settings = _resolve_settings(None, rows_per_chunk, log_level)
    configure_logging(settings.log_level)

    try:
        outcome = run_profile(input_path, settings.rows_per_chunk)
    except _DATA_ERRORS as exc:
        raise _fail(str(exc)) from exc

    typer.echo(f"Rows: {outcome.row_count}")
    for name, stats in outcome.dataset_profile.columns.items():
        kind = outcome.schema_analysis.columns[name]
        typer.echo(
            f"  {name} ({kind.value}, {stats.dtype}): "
            f"null_ratio={stats.null_ratio:.2%} "
            f"cardinality~{stats.cardinality_estimate} "
            f"duplicate%={stats.duplicate_percentage:.1f}"
        )


@app.command()
def optimize(
    input_path: Annotated[Path, typer.Argument(help="Dataset to optimize.")],
    out: Annotated[Path, typer.Option(help="Output file path.")],
    export_format: Annotated[
        ExportFormat | None,
        typer.Option("--format", help="Output format. Inferred from --out if omitted."),
    ] = None,
    mode: ModeOption = None,
    rows_per_chunk: RowsPerChunkOption = None,
    compress: Annotated[
        bool,
        typer.Option(
            "--compress",
            help=(
                "Zstandard-compress the output. For CSV/JSON this wraps the "
                "file in a Zstandard stream (appends .zst to the filename); "
                "for Parquet it selects the zstd codec instead of leaving "
                "the file uncompressed. Omitted, every format is written "
                'with zero compression - a genuine "optimize only" output.'
            ),
        ),
    ] = False,
    report_path: Annotated[
        Path | None,
        typer.Option("--report", help="Where to save the JSON execution report."),
    ] = None,
    log_level: LogLevelOption = None,
) -> None:
    """Optimize a dataset, lossless by default, and export the result."""
    settings = _resolve_settings(mode, rows_per_chunk, log_level)
    configure_logging(settings.log_level)

    resolved_format = export_format or ExportFormat.from_suffix(out)
    if resolved_format is None:
        raise _fail(
            f"cannot infer export format from '{out}'; pass --format explicitly"
        )

    # Parquet's compression is an internal codec choice (no outer file
    # wrapper needed), so only CSV/JSON get the .zst filename treatment.
    actual_out = out
    compressing_a_container_format = (
        compress and resolved_format is not ExportFormat.PARQUET
    )
    if compressing_a_container_format and out.suffix != ".zst":
        actual_out = out.with_name(out.name + ".zst")

    try:
        outcome = run_optimize(
            input_path,
            actual_out,
            resolved_format,
            settings.mode,
            settings.rows_per_chunk,
            build_default_registry(),
            compress,
        )
    except ExportError as exc:
        raise _fail(str(exc)) from exc
    except _DATA_ERRORS as exc:
        raise _fail(str(exc)) from exc

    report = outcome.report
    typer.echo(f"Wrote {actual_out} ({resolved_format.value}).")
    typer.echo(
        f"Bytes (optimized columns only): {report.total_bytes_before} -> "
        f"{report.total_bytes_after} "
        f"(validation {'passed' if report.validation.passed else 'FAILED'})"
    )
    if report.input_file_bytes is not None and report.output_file_bytes is not None:
        file_reduction = (
            1 - report.output_file_bytes / report.input_file_bytes
            if report.input_file_bytes
            else 0.0
        )
        typer.echo(
            f"File size:                      {report.input_file_bytes} -> "
            f"{report.output_file_bytes} ({file_reduction:.1%} smaller)"
        )
    if report_path is not None:
        save_report(report, report_path)
        typer.echo(f"Report saved to {report_path}.")


@app.command()
def validate(
    original_path: Annotated[Path, typer.Argument(help="The original input dataset.")],
    optimized_path: Annotated[Path, typer.Argument(help="The exported output.")],
    mode: ModeOption = None,
    rows_per_chunk: RowsPerChunkOption = None,
    log_level: LogLevelOption = None,
) -> None:
    """Validate an already-exported dataset against its original input."""
    settings = _resolve_settings(mode, rows_per_chunk, log_level)
    configure_logging(settings.log_level)

    try:
        result = run_validate(
            original_path, optimized_path, settings.mode, settings.rows_per_chunk
        )
    except _DATA_ERRORS as exc:
        raise _fail(str(exc)) from exc

    if not result.passed:
        typer.echo("Validation FAILED:", err=True)
        for reason in result.reasons:
            typer.echo(f"  - {reason}", err=True)
        raise typer.Exit(code=1)

    typer.echo(
        f"Validation passed ({result.original_row_count} rows, "
        f"mode={settings.mode.value})."
    )


@app.command()
def benchmark(
    input_path: Annotated[Path, typer.Argument(help="Dataset to benchmark.")],
    rows_per_chunk: RowsPerChunkOption = None,
    log_level: LogLevelOption = None,
) -> None:
    """Measure profiling throughput and peak memory for a dataset.

    This exercises the same streaming profiler used by `profile`/
    `optimize`; the dedicated performance/stress-test harness (comparing
    across datasets, regression-tracking results, etc.) is separate,
    later work.
    """
    settings = _resolve_settings(None, rows_per_chunk, log_level)
    configure_logging(settings.log_level)

    # run_profile() already manages its own tracemalloc session and logs
    # elapsed_seconds/peak_memory_bytes on "profile.completed"; wrapping it
    # in a second tracemalloc.start()/stop() here would stop tracing
    # prematurely (the inner stop() clears it before this code could read
    # it), which previously made this always report 0.00 MB regardless of
    # actual usage. Reading the pipeline's own measurement via
    # capture_logs() avoids nesting tracemalloc sessions entirely - the
    # same fix already applied in benchmarks/harness.py. capture_logs()
    # only swaps the processor chain, not the level filter though, so the
    # level is raised to DEBUG for the duration of this call - otherwise
    # the default WARNING level (or any --log-level above INFO) would
    # silently drop "profile.completed" before capture_logs() ever saw it.
    configure_logging("DEBUG")
    try:
        with capture_logs() as logs:
            outcome = run_profile(input_path, settings.rows_per_chunk)
    except _DATA_ERRORS as exc:
        raise _fail(str(exc)) from exc
    finally:
        configure_logging(settings.log_level)
    completed = next(entry for entry in logs if entry["event"] == "profile.completed")
    elapsed = float(completed["elapsed_seconds"])
    peak_memory_bytes = int(completed["peak_memory_bytes"])

    rows_per_second = outcome.row_count / elapsed if elapsed > 0 else float("inf")
    typer.echo(f"Rows:              {outcome.row_count}")
    typer.echo(f"Elapsed:           {elapsed:.3f}s")
    typer.echo(f"Throughput:        {rows_per_second:.0f} rows/s")
    typer.echo(f"Peak memory (CLI): {peak_memory_bytes / 1e6:.2f} MB")


@app.command()
def report(
    report_path: Annotated[Path, typer.Argument(help="A JSON report from --report.")],
) -> None:
    """Display a previously saved execution report."""
    try:
        loaded = load_report(report_path)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise _fail(f"could not load report: {exc}") from exc

    validation_word = "passed" if loaded.validation.passed else "FAILED"
    typer.echo(f"Mode:              {loaded.mode.value}")
    typer.echo(f"Validation:        {validation_word}")
    if not loaded.validation.passed:
        for reason in loaded.validation.reasons:
            typer.echo(f"  - {reason}")
    typer.echo(
        f"Bytes (optimized): {loaded.total_bytes_before} -> {loaded.total_bytes_after}"
    )
    if loaded.input_file_bytes is not None and loaded.output_file_bytes is not None:
        file_reduction = (
            1 - loaded.output_file_bytes / loaded.input_file_bytes
            if loaded.input_file_bytes
            else 0.0
        )
        typer.echo(
            f"File size:         {loaded.input_file_bytes} -> "
            f"{loaded.output_file_bytes} ({file_reduction:.1%} smaller)"
        )
    typer.echo(
        f"Results:           {len(loaded.results)} optimizer applications recorded"
    )
    for result in loaded.results:
        typer.echo(f"  - {result.detail} (lossless={result.lossless})")


if __name__ == "__main__":
    app()
