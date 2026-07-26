from dataclasses import dataclass

import polars as pl

from core.chunk_manager import ChunkManager
from core.optimization_profile import ColumnProfile, OptimizationProfile
from core.profiler import ColumnStats, Profiler
from core.schema_analyzer import ColumnKind, SchemaAnalyzer
from core.types import Chunk, Schema
from engine.planner import Planner
from optimizers.base import OptimizationDecision, OptimizationResult
from optimizers.numeric_downcast import NumericDowncastOptimizer
from plugins.registry import OptimizerRegistry


def _int_stats(
    dtype: pl.DataType, minimum: int, maximum: int, count: int = 1000
) -> ColumnStats:
    return ColumnStats(
        dtype=dtype,
        count=count,
        null_ratio=0.0,
        minimum=minimum,
        maximum=maximum,
        cardinality_estimate=min(count, maximum - minimum + 1),
        duplicate_percentage=0.0,
        average_string_length=None,
        distribution_summary={},
    )


def _profile(**columns: ColumnProfile) -> OptimizationProfile:
    return OptimizationProfile(columns=columns)


def _int_column(
    name: str, dtype: pl.DataType, minimum: int, maximum: int
) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        kind=ColumnKind.INTEGER,
        stats=_int_stats(dtype, minimum, maximum),
    )


# --- evaluate() ------------------------------------------------------------


def test_evaluate_proposes_downcast_when_range_fits_a_smaller_dtype() -> None:
    profile = _profile(age=_int_column("age", pl.Int64(), 0, 90))

    decision = NumericDowncastOptimizer().evaluate(profile)

    assert decision.applicable is True
    assert decision.columns == ["age"]
    assert decision.estimated_saving_bytes == (8 - 1) * 1000  # Int64 -> Int8


def test_evaluate_boundary_value_exactly_at_int8_max_still_fits_int8() -> None:
    profile = _profile(age=_int_column("age", pl.Int64(), -128, 127))

    decision = NumericDowncastOptimizer().evaluate(profile)

    assert decision.applicable is True
    assert decision.columns == ["age"]
    assert decision.estimated_saving_bytes == (8 - 1) * 1000


def test_evaluate_value_just_outside_int8_range_is_not_downcast_to_int8() -> None:
    """One past Int8's max (128) must not be selected for Int8 - proposing
    that would be lossy. It should still fit Int16 instead."""
    profile = _profile(age=_int_column("age", pl.Int64(), -128, 128))

    decision = NumericDowncastOptimizer().evaluate(profile)

    assert decision.applicable is True
    assert decision.columns == ["age"]
    assert decision.estimated_saving_bytes == (8 - 2) * 1000  # Int64 -> Int16, not Int8


def test_evaluate_excludes_columns_already_at_the_smallest_fitting_dtype() -> None:
    profile = _profile(flag=_int_column("flag", pl.Int8(), -1, 1))

    decision = NumericDowncastOptimizer().evaluate(profile)

    assert decision.applicable is False
    assert decision.columns == []
    assert decision.estimated_saving_bytes == 0


def test_evaluate_excludes_non_integer_columns() -> None:
    float_stats = ColumnStats(
        dtype=pl.Float64(),
        count=1000,
        null_ratio=0.0,
        minimum=0.0,
        maximum=1.0,
        cardinality_estimate=1000,
        duplicate_percentage=0.0,
        average_string_length=None,
        distribution_summary={},
    )
    profile = _profile(
        ratio=ColumnProfile(name="ratio", kind=ColumnKind.FLOAT, stats=float_stats)
    )

    decision = NumericDowncastOptimizer().evaluate(profile)

    assert decision.applicable is False
    assert decision.columns == []


def test_evaluate_skips_a_column_with_no_observed_values() -> None:
    stats = ColumnStats(
        dtype=pl.Int64(),
        count=5,
        null_ratio=1.0,
        minimum=None,
        maximum=None,
        cardinality_estimate=0,
        duplicate_percentage=0.0,
        average_string_length=None,
        distribution_summary={},
    )
    profile = _profile(
        empty=ColumnProfile(name="empty", kind=ColumnKind.INTEGER, stats=stats)
    )

    decision = NumericDowncastOptimizer().evaluate(profile)

    assert decision.applicable is False


def test_evaluate_handles_multiple_eligible_columns_and_sums_savings() -> None:
    profile = _profile(
        age=_int_column("age", pl.Int64(), 0, 90),  # -> Int8, saves 7/value
        year=_int_column("year", pl.Int64(), 1900, 2100),  # -> Int16, saves 6/value
    )

    decision = NumericDowncastOptimizer().evaluate(profile)

    assert decision.applicable is True
    assert set(decision.columns) == {"age", "year"}
    assert decision.estimated_saving_bytes == 7 * 1000 + 6 * 1000


def test_evaluate_is_deterministic_given_the_same_profile() -> None:
    profile = _profile(age=_int_column("age", pl.Int64(), 0, 90))
    optimizer = NumericDowncastOptimizer()

    first = optimizer.evaluate(profile)
    second = optimizer.evaluate(profile)

    assert first == second


# --- is_lossless() ----------------------------------------------------------


def test_is_lossless_is_always_true_for_this_optimizer() -> None:
    optimizer = NumericDowncastOptimizer()
    applicable_decision = optimizer.evaluate(
        _profile(age=_int_column("age", pl.Int64(), 0, 90))
    )
    inapplicable_decision = optimizer.evaluate(
        _profile(flag=_int_column("flag", pl.Int8(), -1, 1))
    )

    assert optimizer.is_lossless(applicable_decision) is True
    assert optimizer.is_lossless(inapplicable_decision) is True


# --- apply() -----------------------------------------------------------------


def _chunk(data: pl.DataFrame, schema: Schema) -> Chunk:
    return Chunk(data=data, index=0, is_last=True, source_schema=schema)


def test_apply_casts_decided_columns_to_the_target_dtype() -> None:
    schema = Schema(columns={"age": pl.Int64()})
    profile = _profile(age=_int_column("age", pl.Int64(), 0, 90))
    optimizer = NumericDowncastOptimizer()
    decision = optimizer.evaluate(profile)

    chunk = _chunk(
        pl.DataFrame({"age": [10, 20, 90]}, schema={"age": pl.Int64}), schema
    )
    new_chunk, result = optimizer.apply(chunk, decision)

    assert new_chunk.data["age"].dtype == pl.Int8()
    assert result.columns_changed == ["age"]
    assert result.lossless is True


def test_apply_round_trip_preserves_exact_values() -> None:
    """Downcast then cast back up: values must survive exactly, proving
    the transformation is genuinely lossless, not just labeled as such."""
    schema = Schema(columns={"age": pl.Int64()})
    original_values = [0, 10, 90, -5]
    profile = _profile(age=_int_column("age", pl.Int64(), -5, 90))
    optimizer = NumericDowncastOptimizer()
    decision = optimizer.evaluate(profile)

    chunk = _chunk(
        pl.DataFrame({"age": original_values}, schema={"age": pl.Int64}), schema
    )
    new_chunk, _ = optimizer.apply(chunk, decision)

    round_tripped = new_chunk.data["age"].cast(pl.Int64).to_list()
    assert round_tripped == original_values


def test_apply_report_content_is_correct() -> None:
    schema = Schema(columns={"age": pl.Int64()})
    profile = _profile(age=_int_column("age", pl.Int64(), 0, 90))
    optimizer = NumericDowncastOptimizer()
    decision = optimizer.evaluate(profile)

    # Values must stay within the profile's declared range [0, 90]; the
    # profile (not this chunk) is what evaluate() used to pick Int8.
    chunk = _chunk(
        pl.DataFrame({"age": [i % 91 for i in range(1000)]}, schema={"age": pl.Int64}),
        schema,
    )
    _, result = optimizer.apply(chunk, decision)

    assert result.columns_changed == ["age"]
    assert result.bytes_before == 8 * 1000
    assert result.bytes_after == 1 * 1000
    assert result.lossless is True
    assert "age" in result.detail
    assert "Int64" in result.detail
    assert "Int8" in result.detail


def test_apply_with_an_inapplicable_decision_leaves_the_chunk_unchanged() -> None:
    schema = Schema(columns={"flag": pl.Int8()})
    profile = _profile(flag=_int_column("flag", pl.Int8(), -1, 1))
    optimizer = NumericDowncastOptimizer()
    decision = optimizer.evaluate(profile)
    assert decision.applicable is False

    chunk = _chunk(pl.DataFrame({"flag": [1, -1, 0]}, schema={"flag": pl.Int8}), schema)
    new_chunk, result = optimizer.apply(chunk, decision)

    assert new_chunk is chunk
    assert result.columns_changed == []


def test_apply_only_touches_columns_named_in_the_decision() -> None:
    schema = Schema(columns={"age": pl.Int64(), "id": pl.Int64()})
    # Only "age" fits a smaller dtype; "id" spans the full Int64 range.
    profile = _profile(
        age=_int_column("age", pl.Int64(), 0, 90),
        id=_int_column("id", pl.Int64(), 0, 9_223_372_036_854_775_807),
    )
    optimizer = NumericDowncastOptimizer()
    decision = optimizer.evaluate(profile)
    assert decision.columns == ["age"]

    chunk = _chunk(
        pl.DataFrame(
            {"age": [10, 20, 30], "id": [1, 2, 3]},
            schema={"age": pl.Int64, "id": pl.Int64},
        ),
        schema,
    )
    new_chunk, _ = optimizer.apply(chunk, decision)

    assert new_chunk.data["age"].dtype == pl.Int8()
    assert new_chunk.data["id"].dtype == pl.Int64()


# --- end to end: profiler -> schema analyzer -> planner input -> optimizer --


def test_numeric_downcast_end_to_end_via_profiler_and_chunk_manager() -> None:
    schema = Schema(columns={"age": pl.Int64()})

    class _ListReader:
        def __init__(self, batches: list[pl.DataFrame]) -> None:
            self._batches = batches

        def open(self) -> None:
            pass

        def schema(self) -> Schema:
            return schema

        def read_batches(self, rows_per_batch: int):  # type: ignore[no-untyped-def]
            yield from self._batches

        def close(self) -> None:
            pass

    batches = [
        pl.DataFrame({"age": [1, 2, 3]}, schema={"age": pl.Int64}),
        pl.DataFrame({"age": [4, 5, 90]}, schema={"age": pl.Int64}),
    ]
    manager = ChunkManager(_ListReader(batches), rows_per_chunk=3)

    profiler = Profiler(schema)
    chunks = list(manager.stream())
    for chunk in chunks:
        profiler.process(chunk)
    dataset_profile = profiler.finalize()
    schema_analysis = SchemaAnalyzer().analyze(schema)

    from core.optimization_profile import build_optimization_profile

    optimization_profile = build_optimization_profile(dataset_profile, schema_analysis)

    optimizer = NumericDowncastOptimizer()
    decision = optimizer.evaluate(optimization_profile)
    assert decision.applicable is True
    assert decision.columns == ["age"]

    results = []
    optimized_chunks = []
    for chunk in chunks:
        new_chunk, result = optimizer.apply(chunk, decision)
        optimized_chunks.append(new_chunk)
        results.append(result)

    assert all(c.data["age"].dtype == pl.Int8() for c in optimized_chunks)
    combined = pl.concat([c.data["age"].cast(pl.Int64) for c in optimized_chunks])
    assert combined.to_list() == [1, 2, 3, 4, 5, 90]
    assert all(r.lossless for r in results)


# --- registry + planner integration -----------------------------------------


def test_registering_numeric_downcast_lets_the_planner_pick_it_up() -> None:
    registry = OptimizerRegistry()
    registry.register(NumericDowncastOptimizer())
    profile = _profile(age=_int_column("age", pl.Int64(), 0, 90))

    plan = Planner().plan(profile, registry)

    assert len(plan.steps) == 1
    assert plan.steps[0].optimizer_name == "numeric_downcast"
    assert plan.steps[0].decision.applicable is True
    assert plan.steps[0].decision.columns == ["age"]
    assert plan.steps[0].decision.estimated_saving_bytes == (8 - 1) * 1000


def test_planner_excludes_numeric_downcast_when_not_applicable() -> None:
    registry = OptimizerRegistry()
    registry.register(NumericDowncastOptimizer())
    # Already at the smallest fitting dtype: nothing for it to do.
    profile = _profile(flag=_int_column("flag", pl.Int8(), -1, 1))

    plan = Planner().plan(profile, registry)

    assert plan.steps == []


def test_planner_orders_numeric_downcast_alongside_other_optimizers() -> None:
    @dataclass
    class _BiggerSavingFakeOptimizer:
        name: str

        def evaluate(self, profile: OptimizationProfile) -> OptimizationDecision:
            return OptimizationDecision(
                applicable=True,
                columns=["age"],
                rationale="fake, bigger saving than numeric_downcast",
                estimated_saving_bytes=999_999,
            )

        def apply(
            self, chunk: Chunk, decision: OptimizationDecision
        ) -> tuple[Chunk, OptimizationResult]:
            raise NotImplementedError("not exercised by this test")

        def is_lossless(self, decision: OptimizationDecision) -> bool:
            raise NotImplementedError("not exercised by this test")

    registry = OptimizerRegistry()
    registry.register(NumericDowncastOptimizer())
    registry.register(_BiggerSavingFakeOptimizer(name="bigger_saver"))
    profile = _profile(age=_int_column("age", pl.Int64(), 0, 90))

    plan = Planner().plan(profile, registry)

    assert [step.optimizer_name for step in plan.steps] == [
        "bigger_saver",
        "numeric_downcast",
    ]
