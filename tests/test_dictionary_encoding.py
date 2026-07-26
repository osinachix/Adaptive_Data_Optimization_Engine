from dataclasses import dataclass

import polars as pl

from core.chunk_manager import ChunkManager
from core.optimization_profile import ColumnProfile, OptimizationProfile
from core.profiler import ColumnStats, Profiler
from core.schema_analyzer import ColumnKind, SchemaAnalyzer
from core.types import Chunk, Schema
from engine.planner import Planner
from optimizers.base import OptimizationDecision, OptimizationResult
from optimizers.dictionary_encoding import DictionaryEncodingOptimizer
from plugins.registry import OptimizerRegistry


def _string_stats(
    count: int, cardinality_estimate: int, average_string_length: float
) -> ColumnStats:
    return ColumnStats(
        dtype=pl.String(),
        count=count,
        null_ratio=0.0,
        minimum="a",
        maximum="z",
        cardinality_estimate=cardinality_estimate,
        duplicate_percentage=(1 - cardinality_estimate / count) * 100 if count else 0.0,
        average_string_length=average_string_length,
        distribution_summary={},
    )


def _profile(**columns: ColumnProfile) -> OptimizationProfile:
    return OptimizationProfile(columns=columns)


def _string_column(
    name: str, count: int, cardinality_estimate: int, average_string_length: float = 5.0
) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        kind=ColumnKind.STRING,
        stats=_string_stats(count, cardinality_estimate, average_string_length),
    )


# --- evaluate() --------------------------------------------------------------


def test_evaluate_proposes_encoding_for_a_low_cardinality_column() -> None:
    # avg_len=5, count=1000, cardinality=5: current=5000, new=5*5+4*1000=4025
    profile = _profile(region=_string_column("region", 1000, 5))

    decision = DictionaryEncodingOptimizer().evaluate(profile)

    assert decision.applicable is True
    assert decision.columns == ["region"]
    assert decision.estimated_saving_bytes == 5000 - 4025


def test_evaluate_boundary_zero_saving_is_not_selected() -> None:
    """avg_len=5, count=1000, cardinality=200 is the exact breakeven point:
    current = 5*1000 = 5000, new = 5*200 + 4*1000 = 5000, saving = 0. A
    decision that would save nothing must not be proposed."""
    profile = _profile(region=_string_column("region", 1000, 200))

    decision = DictionaryEncodingOptimizer().evaluate(profile)

    assert decision.applicable is False
    assert decision.columns == []


def test_evaluate_just_above_breakeven_is_selected() -> None:
    """One distinct value fewer than the breakeven point (199, not 200)
    tips the saving positive again: new = 5*199 + 4*1000 = 4995, saving =
    5."""
    profile = _profile(region=_string_column("region", 1000, 199))

    decision = DictionaryEncodingOptimizer().evaluate(profile)

    assert decision.applicable is True
    assert decision.columns == ["region"]
    assert decision.estimated_saving_bytes == 5


def test_evaluate_excludes_a_near_unique_column() -> None:
    # cardinality == count: no repetition at all, so encoding only adds
    # dictionary + code overhead with nothing to deduplicate.
    profile = _profile(id_text=_string_column("id_text", 1000, 1000))

    decision = DictionaryEncodingOptimizer().evaluate(profile)

    assert decision.applicable is False
    assert decision.columns == []
    assert decision.estimated_saving_bytes == 0


def test_evaluate_excludes_non_string_columns() -> None:
    int_stats = ColumnStats(
        dtype=pl.Int64(),
        count=1000,
        null_ratio=0.0,
        minimum=0,
        maximum=10,
        cardinality_estimate=5,
        duplicate_percentage=99.0,
        average_string_length=None,
        distribution_summary={},
    )
    profile = _profile(
        code=ColumnProfile(name="code", kind=ColumnKind.INTEGER, stats=int_stats)
    )

    decision = DictionaryEncodingOptimizer().evaluate(profile)

    assert decision.applicable is False
    assert decision.columns == []


def test_evaluate_skips_a_column_with_no_observed_values() -> None:
    stats = ColumnStats(
        dtype=pl.String(),
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
        empty=ColumnProfile(name="empty", kind=ColumnKind.STRING, stats=stats)
    )

    decision = DictionaryEncodingOptimizer().evaluate(profile)

    assert decision.applicable is False


def test_evaluate_handles_multiple_eligible_columns_and_sums_savings() -> None:
    profile = _profile(
        region=_string_column("region", 1000, 5),  # saving 975
        status=_string_column("status", 1000, 3),  # saving 985 (5000-4015)
    )

    decision = DictionaryEncodingOptimizer().evaluate(profile)

    assert decision.applicable is True
    assert set(decision.columns) == {"region", "status"}
    assert decision.estimated_saving_bytes == 975 + 985


def test_evaluate_is_deterministic_given_the_same_profile() -> None:
    profile = _profile(region=_string_column("region", 1000, 5))
    optimizer = DictionaryEncodingOptimizer()

    first = optimizer.evaluate(profile)
    second = optimizer.evaluate(profile)

    assert first == second


# --- is_lossless() -----------------------------------------------------------


def test_is_lossless_is_always_true_for_this_optimizer() -> None:
    optimizer = DictionaryEncodingOptimizer()
    applicable_decision = optimizer.evaluate(
        _profile(region=_string_column("region", 1000, 5))
    )
    inapplicable_decision = optimizer.evaluate(
        _profile(id_text=_string_column("id_text", 1000, 1000))
    )

    assert optimizer.is_lossless(applicable_decision) is True
    assert optimizer.is_lossless(inapplicable_decision) is True


# --- apply() -------------------------------------------------------------


def _chunk(data: pl.DataFrame, schema: Schema) -> Chunk:
    return Chunk(data=data, index=0, is_last=True, source_schema=schema)


def test_apply_casts_decided_columns_to_categorical() -> None:
    schema = Schema(columns={"region": pl.String()})
    # The profile (not the chunk row count) drives applicability; declare a
    # realistic count so the fixed per-row code overhead doesn't dominate.
    profile = _profile(region=_string_column("region", 1000, 2))
    optimizer = DictionaryEncodingOptimizer()
    decision = optimizer.evaluate(profile)

    chunk = _chunk(
        pl.DataFrame({"region": ["north", "south", "north", "north", "south"]}), schema
    )
    new_chunk, result = optimizer.apply(chunk, decision)

    assert new_chunk.data["region"].dtype == pl.Categorical()
    assert result.columns_changed == ["region"]
    assert result.lossless is True


def test_apply_round_trip_preserves_exact_values_including_nulls_and_empty() -> None:
    """Encode then decode back to String: values must survive exactly,
    proving the transformation is genuinely lossless, not just labeled as
    such."""
    schema = Schema(columns={"region": pl.String()})
    original_values = ["north", "south", None, "north", "", "south"]
    profile = _profile(region=_string_column("region", len(original_values), 3))
    optimizer = DictionaryEncodingOptimizer()
    decision = optimizer.evaluate(profile)

    chunk = _chunk(pl.DataFrame({"region": original_values}), schema)
    new_chunk, _ = optimizer.apply(chunk, decision)

    round_tripped = new_chunk.data["region"].cast(pl.String).to_list()
    assert round_tripped == original_values


def test_apply_report_content_is_correct() -> None:
    schema = Schema(columns={"region": pl.String()})
    categories = ["north", "south", "east", "west", "central"]
    values = [categories[i % len(categories)] for i in range(1000)]
    profile = _profile(
        region=_string_column("region", 1000, 5, average_string_length=5.4)
    )
    optimizer = DictionaryEncodingOptimizer()
    decision = optimizer.evaluate(profile)
    assert decision.applicable is True

    chunk = _chunk(pl.DataFrame({"region": values}), schema)
    _, result = optimizer.apply(chunk, decision)

    assert result.columns_changed == ["region"]
    assert result.bytes_before > result.bytes_after
    assert result.lossless is True
    assert "region" in result.detail
    assert "Categorical" in result.detail


def test_apply_with_an_inapplicable_decision_leaves_the_chunk_unchanged() -> None:
    schema = Schema(columns={"id_text": pl.String()})
    profile = _profile(id_text=_string_column("id_text", 3, 3))
    optimizer = DictionaryEncodingOptimizer()
    decision = optimizer.evaluate(profile)
    assert decision.applicable is False

    chunk = _chunk(pl.DataFrame({"id_text": ["a1", "b2", "c3"]}), schema)
    new_chunk, result = optimizer.apply(chunk, decision)

    assert new_chunk is chunk
    assert result.columns_changed == []


def test_apply_only_touches_columns_named_in_the_decision() -> None:
    schema = Schema(columns={"region": pl.String(), "id_text": pl.String()})
    profile = _profile(
        region=_string_column("region", 1000, 2),
        id_text=_string_column("id_text", 1000, 1000),  # near-unique, not eligible
    )
    optimizer = DictionaryEncodingOptimizer()
    decision = optimizer.evaluate(profile)
    assert decision.columns == ["region"]

    chunk = _chunk(
        pl.DataFrame(
            {
                "region": ["north", "south", "north", "north", "south"],
                "id_text": ["a1", "b2", "c3", "d4", "e5"],
            }
        ),
        schema,
    )
    new_chunk, _ = optimizer.apply(chunk, decision)

    assert new_chunk.data["region"].dtype == pl.Categorical()
    assert new_chunk.data["id_text"].dtype == pl.String()


# --- end to end: profiler -> schema analyzer -> planner input -> optimizer --


def test_dictionary_encoding_end_to_end_via_profiler_and_chunk_manager() -> None:
    schema = Schema(columns={"region": pl.String()})

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

    # Realistic scale: dictionary encoding's fixed per-row code overhead
    # only pays off once there are enough rows, so a handful of values
    # isn't representative (verified empirically while building this
    # optimizer - Polars' own Categorical showed no size benefit at 8
    # rows either).
    batch_1 = ["north" if i % 2 == 0 else "south" for i in range(300)]
    batch_2 = ["north" if i % 3 == 0 else "south" for i in range(300)]
    batches = [
        pl.DataFrame({"region": batch_1}),
        pl.DataFrame({"region": batch_2}),
    ]
    manager = ChunkManager(_ListReader(batches), rows_per_chunk=300)

    profiler = Profiler(schema)
    chunks = list(manager.stream())
    for chunk in chunks:
        profiler.process(chunk)
    dataset_profile = profiler.finalize()
    schema_analysis = SchemaAnalyzer().analyze(schema)

    from core.optimization_profile import build_optimization_profile

    optimization_profile = build_optimization_profile(dataset_profile, schema_analysis)

    optimizer = DictionaryEncodingOptimizer()
    decision = optimizer.evaluate(optimization_profile)
    assert decision.applicable is True
    assert decision.columns == ["region"]

    results = []
    optimized_chunks = []
    for chunk in chunks:
        new_chunk, result = optimizer.apply(chunk, decision)
        optimized_chunks.append(new_chunk)
        results.append(result)

    assert all(c.data["region"].dtype == pl.Categorical() for c in optimized_chunks)
    combined = pl.concat([c.data["region"].cast(pl.String) for c in optimized_chunks])
    assert combined.to_list() == batch_1 + batch_2
    assert all(r.lossless for r in results)


# --- registry + planner integration -----------------------------------------


def test_registering_dictionary_encoding_lets_the_planner_pick_it_up() -> None:
    registry = OptimizerRegistry()
    registry.register(DictionaryEncodingOptimizer())
    profile = _profile(region=_string_column("region", 1000, 5))

    plan = Planner().plan(profile, registry)

    assert len(plan.steps) == 1
    assert plan.steps[0].optimizer_name == "dictionary_encoding"
    assert plan.steps[0].decision.applicable is True
    assert plan.steps[0].decision.columns == ["region"]


def test_planner_excludes_dictionary_encoding_when_not_applicable() -> None:
    registry = OptimizerRegistry()
    registry.register(DictionaryEncodingOptimizer())
    profile = _profile(id_text=_string_column("id_text", 1000, 1000))

    plan = Planner().plan(profile, registry)

    assert plan.steps == []


def test_planner_orders_dictionary_encoding_alongside_other_optimizers() -> None:
    @dataclass
    class _BiggerSavingFakeOptimizer:
        name: str

        def evaluate(self, profile: OptimizationProfile) -> OptimizationDecision:
            return OptimizationDecision(
                applicable=True,
                columns=["region"],
                rationale="fake, bigger saving than dictionary_encoding",
                estimated_saving_bytes=999_999,
            )

        def apply(
            self, chunk: Chunk, decision: OptimizationDecision
        ) -> tuple[Chunk, OptimizationResult]:
            raise NotImplementedError("not exercised by this test")

        def is_lossless(self, decision: OptimizationDecision) -> bool:
            raise NotImplementedError("not exercised by this test")

    registry = OptimizerRegistry()
    registry.register(DictionaryEncodingOptimizer())
    registry.register(_BiggerSavingFakeOptimizer(name="bigger_saver"))
    profile = _profile(region=_string_column("region", 1000, 5))

    plan = Planner().plan(profile, registry)

    assert [step.optimizer_name for step in plan.steps] == [
        "bigger_saver",
        "dictionary_encoding",
    ]
