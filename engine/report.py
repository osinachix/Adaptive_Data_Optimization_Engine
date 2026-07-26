from dataclasses import dataclass, field

from core.mode import OptimizationMode
from engine.validator import ValidationResult
from optimizers.base import OptimizationResult


@dataclass(frozen=True)
class ExecutionReport:
    """The complete record of one optimization run: every OptimizationResult
    an optimizer recorded, the validator's verdict, and summary totals. A
    transformation the report cannot account for is a bug (invariant I4);
    nothing here is inferred after the fact - it is exactly what
    ReportGenerator.record() was told, plus the ValidationResult it was
    given.

    total_bytes_before/after cover only the columns optimizers actually
    touched (numeric_downcast, dictionary_encoding, etc.) - they say
    nothing about format-level effects like Parquet's own column
    compression or an exported CSV/JSON being Zstandard-compressed as a
    container. input_file_bytes/output_file_bytes are the real on-disk
    sizes, when the caller measured them (None otherwise, e.g. for older
    saved reports) - added after finding that a real dataset's file shrank
    ~47% in a way total_bytes_before/after didn't capture or explain at
    all: that gap is exactly what invariant I4 exists to prevent.
    """

    results: list[OptimizationResult]
    validation: ValidationResult
    mode: OptimizationMode
    total_bytes_before: int
    total_bytes_after: int
    input_file_bytes: int | None = None
    output_file_bytes: int | None = None


@dataclass
class ReportGenerator:
    """Accumulates OptimizationResult records as chunks are processed and
    produces the final ExecutionReport once validation has run. Holds only
    the results list (proportional to work done - chunks x optimizers
    applied - not to row count) and never touches chunk data itself."""

    mode: OptimizationMode
    _results: list[OptimizationResult] = field(default_factory=list)

    def record(self, result: OptimizationResult) -> None:
        """Add one optimizer's result for one chunk to the report."""
        self._results.append(result)

    def finalize(
        self,
        validation: ValidationResult,
        input_file_bytes: int | None = None,
        output_file_bytes: int | None = None,
    ) -> ExecutionReport:
        """Produce the final report, given the validator's verdict and
        (optionally) the real on-disk input/output sizes."""
        return ExecutionReport(
            results=list(self._results),
            validation=validation,
            mode=self.mode,
            total_bytes_before=sum(result.bytes_before for result in self._results),
            total_bytes_after=sum(result.bytes_after for result in self._results),
            input_file_bytes=input_file_bytes,
            output_file_bytes=output_file_bytes,
        )
