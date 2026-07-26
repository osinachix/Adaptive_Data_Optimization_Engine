import time
import tracemalloc
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import tee
from pathlib import Path

import structlog

from core.chunk_manager import ChunkManager
from core.mode import OptimizationMode
from core.optimization_profile import OptimizationProfile, build_optimization_profile
from core.profiler import DatasetProfile, Profiler
from core.readers.factory import ReaderFactory
from core.schema_analyzer import SchemaAnalysis, SchemaAnalyzer
from core.types import Chunk, Schema
from engine.exporter import Exporter, ExportFormat
from engine.planner import ExecutionPlan, Planner
from engine.report import ExecutionReport, ReportGenerator
from engine.validator import ValidationResult, Validator
from plugins.registry import OptimizerRegistry

logger = structlog.get_logger()


@dataclass(frozen=True)
class ProfileOutcome:
    """The result of profiling a dataset: its schema, raw statistics,
    type classification, and the combined OptimizationProfile a planner
    can consume."""

    schema: Schema
    dataset_profile: DatasetProfile
    schema_analysis: SchemaAnalysis
    optimization_profile: OptimizationProfile
    row_count: int


@dataclass(frozen=True)
class OptimizeOutcome:
    """The result of a full optimize run: the plan that was executed and
    the final execution report (which itself carries the validation
    outcome)."""

    plan: ExecutionPlan
    report: ExecutionReport


def _profile_chunks(chunks: Iterator[Chunk]) -> tuple[Schema, DatasetProfile, int]:
    """Fold a Chunk stream into a DatasetProfile. Shared by run_profile
    and run_optimize so the profiling algorithm exists in exactly one
    place; carries no logging or timing of its own; callers own that."""
    profiler: Profiler | None = None
    schema: Schema | None = None
    row_count = 0
    for chunk in chunks:
        if profiler is None:
            schema = chunk.source_schema
            profiler = Profiler(schema)
        profiler.process(chunk)
        row_count += len(chunk.data)

    if profiler is None or schema is None:
        raise ValueError("no data read: the input produced zero chunks")
    return schema, profiler.finalize(), row_count


def run_profile(input_path: str | Path, rows_per_chunk: int) -> ProfileOutcome:
    """Stream input_path once and produce its profile. Never holds more
    than the current chunk in memory. This is also the engine's first of
    two passes when called from run_optimize (a second, independent read
    is required to apply and export a plan, since profiling deliberately
    retains no rows)."""
    input_path = Path(input_path)
    execution_id = str(uuid.uuid4())
    log = logger.bind(execution_id=execution_id, input_path=str(input_path))
    log.info("profile.started", rows_per_chunk=rows_per_chunk)

    start = time.perf_counter()
    tracemalloc.start()
    try:
        reader = ReaderFactory.for_path(input_path)
        manager = ChunkManager(reader, rows_per_chunk)
        schema, dataset_profile, row_count = _profile_chunks(manager.stream())

        log.info("profile.schema_established", columns=list(schema.columns.keys()))
        schema_analysis = SchemaAnalyzer().analyze(schema)
        optimization_profile = build_optimization_profile(
            dataset_profile, schema_analysis
        )
    except Exception:
        log.exception("profile.failed")
        raise
    finally:
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    log.info(
        "profile.completed",
        row_count=row_count,
        elapsed_seconds=round(time.perf_counter() - start, 3),
        peak_memory_bytes=peak_memory_bytes,
    )
    return ProfileOutcome(
        schema=schema,
        dataset_profile=dataset_profile,
        schema_analysis=schema_analysis,
        optimization_profile=optimization_profile,
        row_count=row_count,
    )


def run_optimize(
    input_path: str | Path,
    output_path: str | Path,
    export_format: ExportFormat,
    mode: OptimizationMode,
    rows_per_chunk: int,
    registry: OptimizerRegistry,
    compress: bool = False,
) -> OptimizeOutcome:
    """Profile input_path, plan against registry, then re-read
    input_path a second time to apply the plan chunk by chunk, validating
    and exporting as it streams. The two chunk streams Exporter needs
    (original and optimized) are derived from that single second read via
    itertools.tee, so the file is read exactly twice in total (once to
    profile, once to optimize+validate+export), not three times.

    compress is passed straight through to Exporter.export() (Zstandard
    container compression for CSV/JSON; Parquet always compresses
    regardless - see engine/exporter.py). The report's
    input_file_bytes/output_file_bytes reflect the real bytes on disk
    after that, whatever format or compression was used - not just the
    columns optimizers directly touched (see ExecutionReport's docstring)."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    execution_id = str(uuid.uuid4())
    log = logger.bind(execution_id=execution_id, input_path=str(input_path))
    log.info(
        "optimize.started",
        output_path=str(output_path),
        mode=str(mode),
        export_format=str(export_format),
        rows_per_chunk=rows_per_chunk,
        compress=compress,
    )

    start = time.perf_counter()
    tracemalloc.start()
    try:
        profile_reader = ReaderFactory.for_path(input_path)
        profile_manager = ChunkManager(profile_reader, rows_per_chunk)
        schema, dataset_profile, _ = _profile_chunks(profile_manager.stream())
        schema_analysis = SchemaAnalyzer().analyze(schema)
        optimization_profile = build_optimization_profile(
            dataset_profile, schema_analysis
        )

        plan = Planner().plan(optimization_profile, registry)
        log.info(
            "optimize.planned",
            steps=[
                {
                    "optimizer": step.optimizer_name,
                    "columns": step.decision.columns,
                    "estimated_saving_bytes": step.decision.estimated_saving_bytes,
                }
                for step in plan.steps
            ],
        )

        reader = ReaderFactory.for_path(input_path)
        manager = ChunkManager(reader, rows_per_chunk)
        original_chunks, chunks_to_optimize = tee(manager.stream(), 2)

        report_generator = ReportGenerator(mode)

        def _apply_plan(chunks: Iterator[Chunk]) -> Iterator[Chunk]:
            for chunk in chunks:
                current = chunk
                for step in plan.steps:
                    optimizer = registry.get(step.optimizer_name)
                    if optimizer is None:
                        continue
                    current, result = optimizer.apply(current, step.decision)
                    report_generator.record(result)
                    log.info(
                        "optimize.chunk_optimized",
                        chunk_index=chunk.index,
                        optimizer=step.optimizer_name,
                        detail=result.detail,
                    )
                yield current

        optimized_chunks = _apply_plan(chunks_to_optimize)

        validation = Exporter().export(
            original_chunks,
            optimized_chunks,
            mode,
            output_path,
            export_format,
            compress,
        )
        if not validation.passed:
            log.warning("optimize.validation_failed", reasons=validation.reasons)
        report = report_generator.finalize(
            validation,
            input_file_bytes=input_path.stat().st_size,
            output_file_bytes=output_path.stat().st_size,
        )
    except Exception:
        log.exception("optimize.failed")
        raise
    finally:
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    log.info(
        "optimize.completed",
        validation_passed=report.validation.passed,
        total_bytes_before=report.total_bytes_before,
        total_bytes_after=report.total_bytes_after,
        input_file_bytes=report.input_file_bytes,
        output_file_bytes=report.output_file_bytes,
        elapsed_seconds=round(time.perf_counter() - start, 3),
        peak_memory_bytes=peak_memory_bytes,
    )
    return OptimizeOutcome(plan=plan, report=report)


def run_validate(
    original_path: str | Path,
    optimized_path: str | Path,
    mode: OptimizationMode,
    rows_per_chunk: int,
) -> ValidationResult:
    """Validate an already-exported dataset against its original input,
    independent of any optimize run - for re-checking an existing output
    file. Both files are streamed once each, chunk pair by chunk pair;
    this requires matching total row counts (chunk boundaries then align,
    since both readers honor rows_per_chunk identically)."""
    original_path = Path(original_path)
    optimized_path = Path(optimized_path)
    execution_id = str(uuid.uuid4())
    log = logger.bind(
        execution_id=execution_id,
        original_path=str(original_path),
        optimized_path=str(optimized_path),
    )
    log.info("validate.started", mode=str(mode))

    start = time.perf_counter()
    try:
        original_stream = ChunkManager(
            ReaderFactory.for_path(original_path), rows_per_chunk
        ).stream()
        optimized_stream = ChunkManager(
            ReaderFactory.for_path(optimized_path), rows_per_chunk
        ).stream()

        validator = Validator(mode)
        for original_chunk, optimized_chunk in zip(
            original_stream, optimized_stream, strict=True
        ):
            validator.observe(original_chunk, optimized_chunk)
        result = validator.finalize()
    except Exception:
        log.exception("validate.failed")
        raise

    if not result.passed:
        log.warning("validate.failed_checks", reasons=result.reasons)
    log.info(
        "validate.completed",
        passed=result.passed,
        elapsed_seconds=round(time.perf_counter() - start, 3),
    )
    return result
