import json
from dataclasses import asdict
from pathlib import Path

from core.mode import OptimizationMode
from engine.report import ExecutionReport
from engine.validator import ValidationResult
from optimizers.base import OptimizationResult


def save_report(report: ExecutionReport, path: str | Path) -> None:
    """Write an ExecutionReport to path as JSON, so a later `adoe report`
    invocation can load and display it without re-running the pipeline."""
    Path(path).write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")


def load_report(path: str | Path) -> ExecutionReport:
    """Load an ExecutionReport previously written by save_report(). Reports
    saved before input_file_bytes/output_file_bytes existed simply lack
    those keys; .get() makes loading them not a bug (invariant I4 is about
    accounting for what an optimizer changed, not about permanently
    rejecting older report files)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validation_data = data["validation"]
    return ExecutionReport(
        results=[OptimizationResult(**result_data) for result_data in data["results"]],
        validation=ValidationResult(
            passed=validation_data["passed"],
            reasons=list(validation_data["reasons"]),
            original_row_count=validation_data["original_row_count"],
            optimized_row_count=validation_data["optimized_row_count"],
        ),
        mode=OptimizationMode(data["mode"]),
        total_bytes_before=data["total_bytes_before"],
        total_bytes_after=data["total_bytes_after"],
        input_file_bytes=data.get("input_file_bytes"),
        output_file_bytes=data.get("output_file_bytes"),
    )
