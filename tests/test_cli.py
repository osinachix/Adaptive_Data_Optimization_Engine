from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from cli.main import app

runner = CliRunner()


def _write_csv(path: Path, rows: int) -> list[int]:
    ages = [i % 90 for i in range(rows)]
    pl.DataFrame({"age": ages}, schema={"age": pl.Int64}).write_csv(path)
    return ages


def test_profile_command_prints_a_summary_and_exits_zero(tmp_path: Path) -> None:
    csv_path = tmp_path / "input.csv"
    _write_csv(csv_path, 200)

    result = runner.invoke(app, ["profile", str(csv_path), "--log-level", "WARNING"])

    assert result.exit_code == 0
    assert "age" in result.output
    assert "Rows: 200" in result.output


def test_profile_command_fails_cleanly_for_a_malformed_csv_row(tmp_path: Path) -> None:
    """Regression test: the profile/optimize/validate/benchmark commands
    originally caught only (FileNotFoundError, ValueError), but a
    malformed CSV row raises polars.exceptions.ComputeError lazily during
    streaming (not at open() time), which isn't a subclass of either -
    so it propagated as an unhandled exception with a raw traceback
    instead of the usual clean "Error: ..." message and exit code 1.
    Found while building examples/sample_data/missing_and_malformed.csv."""
    csv_path = tmp_path / "malformed.csv"
    csv_path.write_text("id,name\n1,alice\n2,bob,EXTRA_FIELD\n3,carol\n")

    result = runner.invoke(app, ["profile", str(csv_path), "--log-level", "WARNING"])

    assert result.exit_code == 1
    assert "Error:" in result.output
    # A clean, handled CLI failure (_fail() -> typer.Exit -> SystemExit),
    # not the raw ComputeError escaping uncaught to the caller.
    assert isinstance(result.exception, SystemExit)


def test_profile_command_fails_cleanly_for_a_missing_file(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["profile", str(tmp_path / "missing.csv"), "--log-level", "WARNING"]
    )

    assert result.exit_code == 1
    assert "Error:" in result.output


def test_optimize_command_writes_output_and_report(tmp_path: Path) -> None:
    csv_path = tmp_path / "input.csv"
    ages = _write_csv(csv_path, 500)
    output_path = tmp_path / "output.parquet"
    report_path = tmp_path / "report.json"

    result = runner.invoke(
        app,
        [
            "optimize",
            str(csv_path),
            "--out",
            str(output_path),
            "--report",
            str(report_path),
            "--log-level",
            "WARNING",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert report_path.exists()
    assert pl.read_parquet(output_path)["age"].to_list() == ages


def test_optimize_command_infers_format_from_output_extension(tmp_path: Path) -> None:
    csv_path = tmp_path / "input.csv"
    _write_csv(csv_path, 100)
    output_path = tmp_path / "output.csv"

    result = runner.invoke(
        app,
        [
            "optimize",
            str(csv_path),
            "--out",
            str(output_path),
            "--log-level",
            "WARNING",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()


def test_optimize_command_rejects_an_unrecognized_output_extension(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "input.csv"
    _write_csv(csv_path, 50)
    output_path = tmp_path / "output.xyz"

    result = runner.invoke(
        app,
        [
            "optimize",
            str(csv_path),
            "--out",
            str(output_path),
            "--log-level",
            "WARNING",
        ],
    )

    assert result.exit_code == 1
    assert not output_path.exists()


def test_optimize_command_with_compress_appends_zst_and_reports_file_size(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "input.csv"
    ages = _write_csv(csv_path, 500)
    requested_out = tmp_path / "output.csv"

    result = runner.invoke(
        app,
        [
            "optimize",
            str(csv_path),
            "--out",
            str(requested_out),
            "--compress",
            "--log-level",
            "WARNING",
        ],
    )

    actual_out = tmp_path / "output.csv.zst"
    assert result.exit_code == 0
    assert not requested_out.exists()  # the .zst-suffixed path was used instead
    assert actual_out.exists()
    assert str(actual_out) in result.output
    assert "File size:" in result.output
    assert pl.read_csv(actual_out)["age"].to_list() == ages


def test_optimize_command_with_compress_does_not_double_suffix_an_existing_zst_name(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "input.csv"
    _write_csv(csv_path, 100)
    out_path = tmp_path / "output.csv.zst"

    result = runner.invoke(
        app,
        [
            "optimize",
            str(csv_path),
            "--out",
            str(out_path),
            "--compress",
            "--log-level",
            "WARNING",
        ],
    )

    assert result.exit_code == 0
    assert out_path.exists()
    assert not (tmp_path / "output.csv.zst.zst").exists()


def test_optimize_command_with_compress_has_no_effect_on_parquet_filename(
    tmp_path: Path,
) -> None:
    """--compress selects Parquet's internal zstd codec (see
    engine/exporter.py), not an outer file wrapper, so it must not also
    rename the file or wrap it in an outer .zst container."""
    csv_path = tmp_path / "input.csv"
    _write_csv(csv_path, 100)
    out_path = tmp_path / "output.parquet"

    result = runner.invoke(
        app,
        [
            "optimize",
            str(csv_path),
            "--out",
            str(out_path),
            "--compress",
            "--log-level",
            "WARNING",
        ],
    )

    assert result.exit_code == 0
    assert out_path.exists()
    assert not (tmp_path / "output.parquet.zst").exists()


def test_validate_command_passes_after_optimize(tmp_path: Path) -> None:
    csv_path = tmp_path / "input.csv"
    _write_csv(csv_path, 300)
    output_path = tmp_path / "output.parquet"
    runner.invoke(
        app,
        [
            "optimize",
            str(csv_path),
            "--out",
            str(output_path),
            "--log-level",
            "WARNING",
        ],
    )

    result = runner.invoke(
        app, ["validate", str(csv_path), str(output_path), "--log-level", "WARNING"]
    )

    assert result.exit_code == 0
    assert "Validation passed" in result.output


def test_benchmark_command_reports_throughput(tmp_path: Path) -> None:
    csv_path = tmp_path / "input.csv"
    _write_csv(csv_path, 200)

    result = runner.invoke(app, ["benchmark", str(csv_path), "--log-level", "WARNING"])

    assert result.exit_code == 0
    assert "Throughput" in result.output


def test_benchmark_command_reports_nonzero_peak_memory(tmp_path: Path) -> None:
    """Regression test: benchmark wrapped run_profile() (which manages its
    own tracemalloc session) in a second, outer tracemalloc.start()/stop()
    pair. The inner stop() cleared tracing before the outer code read it,
    so "Peak memory (CLI)" always printed 0.00 MB regardless of actual
    usage. Fixed by reading elapsed_seconds/peak_memory_bytes from
    run_profile()'s own "profile.completed" log event via capture_logs(),
    the same fix already used in benchmarks/harness.py."""
    csv_path = tmp_path / "input.csv"
    _write_csv(csv_path, 5_000)

    result = runner.invoke(app, ["benchmark", str(csv_path), "--log-level", "WARNING"])

    assert result.exit_code == 0
    peak_line = next(
        line for line in result.output.splitlines() if "Peak memory" in line
    )
    peak_mb = float(peak_line.split(":")[1].replace("MB", "").strip())
    assert peak_mb > 0.0


def test_report_command_displays_a_saved_report(tmp_path: Path) -> None:
    csv_path = tmp_path / "input.csv"
    _write_csv(csv_path, 300)
    output_path = tmp_path / "output.parquet"
    report_path = tmp_path / "report.json"
    runner.invoke(
        app,
        [
            "optimize",
            str(csv_path),
            "--out",
            str(output_path),
            "--report",
            str(report_path),
            "--log-level",
            "WARNING",
        ],
    )

    result = runner.invoke(app, ["report", str(report_path)])

    assert result.exit_code == 0
    assert "Validation:" in result.output
    assert "passed" in result.output
    assert "optimizer applications recorded" in result.output


def test_report_command_fails_cleanly_for_a_missing_report(tmp_path: Path) -> None:
    result = runner.invoke(app, ["report", str(tmp_path / "missing.json")])

    assert result.exit_code == 1
    assert "Error:" in result.output
