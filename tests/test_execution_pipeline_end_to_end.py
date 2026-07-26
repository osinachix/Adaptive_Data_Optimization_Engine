from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest

from core.chunk_manager import ChunkManager
from core.mode import OptimizationMode
from core.optimization_profile import OptimizationProfile, build_optimization_profile
from core.profiler import Profiler
from core.readers.csv_reader import CSVReader
from core.schema_analyzer import SchemaAnalyzer
from core.types import Chunk
from engine.exporter import Exporter, ExportError, ExportFormat
from engine.planner import Planner
from engine.report import ReportGenerator
from optimizers.base import OptimizationDecision, OptimizationResult
from optimizers.numeric_downcast import NumericDowncastOptimizer
from plugins.registry import OptimizerRegistry


def test_full_pipeline_profiles_plans_optimizes_validates_exports_and_reports(
    tmp_path: Path,
) -> None:
    """Wires every M1-M9 piece together end to end: read -> profile ->
    plan -> apply -> validate + export -> report. Matches the guide's
    acceptance criterion: every completed optimization produces both an
    output dataset and a validation report."""
    csv_path = tmp_path / "input.csv"
    ages = [i % 90 for i in range(1_800)]
    pl.DataFrame({"age": ages}, schema={"age": pl.Int64}).write_csv(csv_path)

    original_chunks = list(
        ChunkManager(CSVReader(csv_path), rows_per_chunk=500).stream()
    )
    schema = original_chunks[0].source_schema

    profiler = Profiler(schema)
    for chunk in original_chunks:
        profiler.process(chunk)
    optimization_profile = build_optimization_profile(
        profiler.finalize(), SchemaAnalyzer().analyze(schema)
    )

    registry = OptimizerRegistry()
    registry.register(NumericDowncastOptimizer())
    plan = Planner().plan(optimization_profile, registry)
    assert len(plan.steps) == 1
    assert plan.steps[0].decision.columns == ["age"]

    optimizer = registry.get(plan.steps[0].optimizer_name)
    assert optimizer is not None
    decision = plan.steps[0].decision

    report_generator = ReportGenerator(OptimizationMode.LOSSLESS)
    optimized_chunks = []
    for chunk in original_chunks:
        new_chunk, result = optimizer.apply(chunk, decision)
        report_generator.record(result)
        optimized_chunks.append(new_chunk)

    output_path = tmp_path / "output.parquet"
    validation = Exporter().export(
        iter(original_chunks),
        iter(optimized_chunks),
        OptimizationMode.LOSSLESS,
        output_path,
        ExportFormat.PARQUET,
    )
    report = report_generator.finalize(validation)

    assert validation.passed is True
    assert output_path.exists()
    readback = pl.read_parquet(output_path)
    assert readback["age"].dtype == pl.Int8()
    assert readback["age"].to_list() == ages

    assert report.validation.passed is True
    assert report.mode is OptimizationMode.LOSSLESS
    assert report.total_bytes_before > report.total_bytes_after
    assert len(report.results) == len(original_chunks)
    assert all(result.lossless for result in report.results)


@dataclass
class _CorruptingOptimizer:
    """A fake optimizer that silently alters values while still claiming
    to be lossless - simulating a bug. The Validator must catch this
    independently of what the optimizer reports about itself; it is not
    supposed to trust is_lossless()/OptimizationResult.lossless blindly."""

    name: str = "corrupting"

    def evaluate(self, profile: OptimizationProfile) -> OptimizationDecision:
        raise NotImplementedError("not exercised by this test")

    def apply(
        self, chunk: Chunk, decision: OptimizationDecision
    ) -> tuple[Chunk, OptimizationResult]:
        corrupted_data = chunk.data.with_columns(pl.col("age") + 1)
        new_chunk = Chunk(
            data=corrupted_data,
            index=chunk.index,
            is_last=chunk.is_last,
            source_schema=chunk.source_schema,
        )
        result = OptimizationResult(
            columns_changed=["age"],
            bytes_before=100,
            bytes_after=100,
            lossless=True,  # a lie - this is exactly what the validator must catch
            detail="age: incremented by 1 (deliberately corrupted for this test)",
        )
        return new_chunk, result

    def is_lossless(self, decision: OptimizationDecision) -> bool:
        return True


def test_full_pipeline_corrupted_result_fails_validation_but_report_accounts_for_all(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "input.csv"
    ages = [i % 90 for i in range(300)]
    pl.DataFrame({"age": ages}, schema={"age": pl.Int64}).write_csv(csv_path)

    original_chunks = list(
        ChunkManager(CSVReader(csv_path), rows_per_chunk=100).stream()
    )
    decision = OptimizationDecision(
        applicable=True,
        columns=["age"],
        rationale="fake corruption for testing",
        estimated_saving_bytes=1,
    )
    optimizer = _CorruptingOptimizer()

    report_generator = ReportGenerator(OptimizationMode.LOSSLESS)
    optimized_chunks = []
    for chunk in original_chunks:
        new_chunk, result = optimizer.apply(chunk, decision)
        report_generator.record(result)
        optimized_chunks.append(new_chunk)

    output_path = tmp_path / "output.csv"
    with pytest.raises(ExportError) as exc_info:
        Exporter().export(
            iter(original_chunks),
            iter(optimized_chunks),
            OptimizationMode.LOSSLESS,
            output_path,
            ExportFormat.CSV,
        )

    # Invariant I3: nothing appears at the output path, and no stray
    # temp file is left behind either - only the original input remains.
    assert not output_path.exists()
    assert list(tmp_path.iterdir()) == [csv_path]

    # The report must still account for every result the (buggy)
    # optimizer produced - one per chunk - even though export failed;
    # its job is to record what happened, not to hide it on failure.
    report = report_generator.finalize(exc_info.value.validation)

    assert report.validation.passed is False
    assert any(
        "does not reconstruct exactly" in reason for reason in report.validation.reasons
    )
    assert len(report.results) == len(original_chunks)
    assert all(result.lossless for result in report.results)  # the lie, recorded as-is
