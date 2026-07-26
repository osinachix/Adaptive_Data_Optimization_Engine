from dataclasses import dataclass, field

from core.mode import OptimizationMode
from core.types import Chunk


@dataclass(frozen=True)
class ValidationResult:
    """The outcome of validating an optimized Chunk stream against the
    original stream, for the selected mode. Exporters must refuse to write
    to their real destination unless passed is True (invariant I3)."""

    passed: bool
    reasons: list[str]  # empty when passed; one entry per failed check otherwise
    original_row_count: int
    optimized_row_count: int


@dataclass
class Validator:
    """Validates an optimized Chunk stream against the original stream,
    incrementally, one chunk pair at a time: schema (column names
    unchanged), row count, column integrity (null counts unchanged), and -
    only in lossless mode - an exact value-level reconstruction check.

    Holds only bounded running state (row counters, the expected column
    list, and a list of failure reasons); it never retains row data beyond
    the current chunk pair passed to observe().
    """

    mode: OptimizationMode
    _original_row_count: int = 0
    _optimized_row_count: int = 0
    _expected_columns: list[str] | None = None
    _reasons: list[str] = field(default_factory=list)

    def observe(self, original_chunk: Chunk, optimized_chunk: Chunk) -> None:
        """Fold one (original, optimized) chunk pair into the running
        validation state. Call once per chunk, in stream order."""
        if self._expected_columns is None:
            self._expected_columns = list(original_chunk.source_schema.columns)

        self._original_row_count += len(original_chunk.data)
        self._optimized_row_count += len(optimized_chunk.data)

        actual_columns = list(optimized_chunk.data.columns)
        if actual_columns != self._expected_columns:
            self._reasons.append(
                f"chunk {optimized_chunk.index}: column mismatch - expected "
                f"{self._expected_columns}, got {actual_columns}"
            )
            return  # columns don't line up; skip the per-column checks below

        if len(original_chunk.data) != len(optimized_chunk.data):
            self._reasons.append(
                f"chunk {optimized_chunk.index}: row count mismatch within "
                f"chunk ({len(original_chunk.data)} -> {len(optimized_chunk.data)})"
            )
            return  # rows don't line up; skip the per-column checks below

        for column in self._expected_columns:
            original_series = original_chunk.data[column]
            optimized_series = optimized_chunk.data[column]

            if original_series.null_count() != optimized_series.null_count():
                self._reasons.append(
                    f"chunk {optimized_chunk.index}: column '{column}' null "
                    f"count changed ({original_series.null_count()} -> "
                    f"{optimized_series.null_count()})"
                )

            if self.mode is OptimizationMode.LOSSLESS:
                reconstructed = optimized_series.cast(original_series.dtype)
                if not original_series.equals(reconstructed):
                    self._reasons.append(
                        f"chunk {optimized_chunk.index}: column '{column}' "
                        "does not reconstruct exactly under lossless mode"
                    )

    def finalize(self) -> ValidationResult:
        """Derive the final pass/fail verdict once both streams are
        exhausted."""
        if self._original_row_count != self._optimized_row_count:
            self._reasons.append(
                "total row count mismatch: "
                f"{self._original_row_count} -> {self._optimized_row_count}"
            )
        return ValidationResult(
            passed=len(self._reasons) == 0,
            reasons=list(self._reasons),
            original_row_count=self._original_row_count,
            optimized_row_count=self._optimized_row_count,
        )
