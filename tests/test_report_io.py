import json
from pathlib import Path

from core.mode import OptimizationMode
from engine.report import ExecutionReport
from engine.report_io import load_report, save_report
from engine.validator import ValidationResult
from optimizers.base import OptimizationResult


def test_save_then_load_round_trips_an_execution_report(tmp_path: Path) -> None:
    report = ExecutionReport(
        results=[
            OptimizationResult(
                columns_changed=["age"],
                bytes_before=1000,
                bytes_after=200,
                lossless=True,
                detail="age: Int64 -> Int8",
            ),
            OptimizationResult(
                columns_changed=["region"],
                bytes_before=500,
                bytes_after=300,
                lossless=True,
                detail="region: String -> Categorical",
            ),
        ],
        validation=ValidationResult(
            passed=True, reasons=[], original_row_count=1000, optimized_row_count=1000
        ),
        mode=OptimizationMode.LOSSLESS,
        total_bytes_before=1500,
        total_bytes_after=500,
    )
    report_path = tmp_path / "report.json"

    save_report(report, report_path)
    loaded = load_report(report_path)

    assert loaded == report


def test_save_then_load_round_trips_real_file_bytes(tmp_path: Path) -> None:
    report = ExecutionReport(
        results=[],
        validation=ValidationResult(
            passed=True, reasons=[], original_row_count=100, optimized_row_count=100
        ),
        mode=OptimizationMode.LOSSLESS,
        total_bytes_before=0,
        total_bytes_after=0,
        input_file_bytes=1_000_000,
        output_file_bytes=400_000,
    )
    report_path = tmp_path / "report.json"

    save_report(report, report_path)
    loaded = load_report(report_path)

    assert loaded == report
    assert loaded.input_file_bytes == 1_000_000
    assert loaded.output_file_bytes == 400_000


def test_load_report_saved_before_file_bytes_existed_defaults_to_none(
    tmp_path: Path,
) -> None:
    """Simulates a report JSON saved by an older version of ADOE, before
    input_file_bytes/output_file_bytes were added - loading it must not
    fail just because those keys are absent."""
    old_format_report = {
        "results": [],
        "validation": {
            "passed": True,
            "reasons": [],
            "original_row_count": 5,
            "optimized_row_count": 5,
        },
        "mode": "lossless",
        "total_bytes_before": 0,
        "total_bytes_after": 0,
        # no input_file_bytes / output_file_bytes keys at all
    }
    report_path = tmp_path / "old_report.json"
    report_path.write_text(json.dumps(old_format_report), encoding="utf-8")

    loaded = load_report(report_path)

    assert loaded.input_file_bytes is None
    assert loaded.output_file_bytes is None
    assert loaded.validation.passed is True


def test_save_then_load_round_trips_a_failed_validation(tmp_path: Path) -> None:
    report = ExecutionReport(
        results=[],
        validation=ValidationResult(
            passed=False,
            reasons=["chunk 0: column 'age' does not reconstruct exactly"],
            original_row_count=10,
            optimized_row_count=10,
        ),
        mode=OptimizationMode.LOSSLESS,
        total_bytes_before=0,
        total_bytes_after=0,
    )
    report_path = tmp_path / "report.json"

    save_report(report, report_path)
    loaded = load_report(report_path)

    assert loaded.validation.passed is False
    assert loaded.validation.reasons == report.validation.reasons
