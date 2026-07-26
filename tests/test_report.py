from core.mode import OptimizationMode
from engine.report import ReportGenerator
from engine.validator import ValidationResult
from optimizers.base import OptimizationResult


def test_record_accumulates_results_and_finalize_sums_bytes() -> None:
    generator = ReportGenerator(OptimizationMode.LOSSLESS)
    generator.record(
        OptimizationResult(
            columns_changed=["a"],
            bytes_before=100,
            bytes_after=50,
            lossless=True,
            detail="a: Int64 -> Int8",
        )
    )
    generator.record(
        OptimizationResult(
            columns_changed=["b"],
            bytes_before=200,
            bytes_after=150,
            lossless=True,
            detail="b: String -> Categorical",
        )
    )
    validation = ValidationResult(
        passed=True, reasons=[], original_row_count=10, optimized_row_count=10
    )

    report = generator.finalize(validation)

    assert len(report.results) == 2
    assert report.total_bytes_before == 300
    assert report.total_bytes_after == 200
    assert report.validation is validation
    assert report.mode is OptimizationMode.LOSSLESS


def test_finalize_without_file_bytes_defaults_to_none() -> None:
    """Callers that don't measure real file sizes (e.g. existing tests
    constructed before this feature) still work unchanged."""
    generator = ReportGenerator(OptimizationMode.LOSSLESS)
    validation = ValidationResult(
        passed=True, reasons=[], original_row_count=1, optimized_row_count=1
    )

    report = generator.finalize(validation)

    assert report.input_file_bytes is None
    assert report.output_file_bytes is None


def test_finalize_records_real_file_bytes_when_given() -> None:
    generator = ReportGenerator(OptimizationMode.LOSSLESS)
    validation = ValidationResult(
        passed=True, reasons=[], original_row_count=1, optimized_row_count=1
    )

    report = generator.finalize(
        validation, input_file_bytes=1_000_000, output_file_bytes=400_000
    )

    assert report.input_file_bytes == 1_000_000
    assert report.output_file_bytes == 400_000


def test_finalize_with_no_results_recorded() -> None:
    generator = ReportGenerator(OptimizationMode.BALANCED)
    validation = ValidationResult(
        passed=True, reasons=[], original_row_count=0, optimized_row_count=0
    )

    report = generator.finalize(validation)

    assert report.results == []
    assert report.total_bytes_before == 0
    assert report.total_bytes_after == 0


def test_finalize_includes_a_failed_validation_verbatim() -> None:
    generator = ReportGenerator(OptimizationMode.LOSSLESS)
    validation = ValidationResult(
        passed=False,
        reasons=[
            "chunk 0: column 'age' does not reconstruct exactly under lossless mode"
        ],
        original_row_count=10,
        optimized_row_count=10,
    )

    report = generator.finalize(validation)

    assert report.validation.passed is False
    assert report.validation.reasons == validation.reasons
