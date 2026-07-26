import gc
import tracemalloc
from collections.abc import Iterator
from typing import cast

import polars as pl
import pytest

from core.profiler import ColumnAccumulator, Profiler
from core.types import Chunk, Schema

SCHEMA = Schema(columns={"id": pl.Int64(), "name": pl.String()})


def _make_chunks(
    total_rows: int, rows_per_chunk: int, schema: Schema = SCHEMA
) -> Iterator[Chunk]:
    num_chunks = total_rows // rows_per_chunk
    for i in range(num_chunks):
        start = i * rows_per_chunk
        data = pl.DataFrame(
            {
                "id": range(start, start + rows_per_chunk),
                "name": [f"row-{n}" for n in range(start, start + rows_per_chunk)],
            }
        )
        yield Chunk(
            data=data, index=i, is_last=(i == num_chunks - 1), source_schema=schema
        )


# --- ColumnAccumulator -------------------------------------------------


def test_count_and_null_ratio_accumulate_across_chunks() -> None:
    acc = ColumnAccumulator()

    acc.update(pl.Series("x", [1, 2, None, 4]))
    acc.update(pl.Series("x", [None, 6]))
    stats = acc.finalize()

    assert stats.count == 6
    assert stats.null_ratio == pytest.approx(2 / 6)


def test_min_max_track_across_chunks_and_ignore_nulls() -> None:
    acc = ColumnAccumulator()

    acc.update(pl.Series("x", [5, None, 2]))
    acc.update(pl.Series("x", [None, 9, 1]))
    stats = acc.finalize()

    assert stats.minimum == 1
    assert stats.maximum == 9


def test_dtype_is_recorded_from_the_first_update() -> None:
    acc = ColumnAccumulator()

    acc.update(pl.Series("x", [1, 2, 3]))

    assert acc.finalize().dtype == pl.Int64()


def test_cardinality_estimate_is_within_tolerance_for_a_known_distinct_count() -> None:
    acc = ColumnAccumulator()
    true_distinct = 5_000

    acc.update(pl.Series("x", list(range(true_distinct))))

    estimate = acc.finalize().cardinality_estimate

    assert estimate == pytest.approx(true_distinct, rel=0.2)


def test_cardinality_estimate_for_a_constant_column_is_one() -> None:
    acc = ColumnAccumulator()

    acc.update(pl.Series("x", ["same"] * 2_000))

    assert acc.finalize().cardinality_estimate == 1


def test_duplicate_percentage_reflects_known_duplication() -> None:
    acc = ColumnAccumulator()
    # 100 distinct values, each repeated 10x -> 90% duplicate.
    values = [v for v in range(100) for _ in range(10)]

    acc.update(pl.Series("x", values))

    assert acc.finalize().duplicate_percentage == pytest.approx(90.0, abs=5.0)


def test_average_string_length_is_computed_for_string_columns() -> None:
    acc = ColumnAccumulator()

    acc.update(pl.Series("x", ["ab", "abcd", None]))

    assert acc.finalize().average_string_length == pytest.approx(3.0)


def test_average_string_length_is_none_for_non_string_columns() -> None:
    acc = ColumnAccumulator()

    acc.update(pl.Series("x", [1, 2, 3]))

    assert acc.finalize().average_string_length is None


def test_distribution_summary_quantiles_are_reasonable() -> None:
    """distribution_summary is derived from the bounded reservoir sample,
    not the exact dataset (that's what minimum/maximum on ColumnStats are
    for), so its min/max are sampled extremes, not the true boundaries;
    with 200 samples out of 1000 they land close but are not guaranteed
    exact."""
    acc = ColumnAccumulator()

    acc.update(pl.Series("x", list(range(1, 1_001))))  # 1..1000

    summary = cast(dict[str, int], acc.finalize().distribution_summary)

    assert summary["min"] <= summary["p25"] <= summary["p50"]
    assert summary["p50"] <= summary["p75"] <= summary["max"]
    assert summary["min"] <= 50  # sampled min should land near the low end
    assert summary["max"] >= 950  # sampled max should land near the high end
    assert 400 <= summary["p50"] <= 600


def test_finalize_on_a_never_updated_accumulator_does_not_crash() -> None:
    acc = ColumnAccumulator()

    stats = acc.finalize()

    assert stats.count == 0
    assert stats.null_ratio == 0.0
    assert stats.cardinality_estimate == 0
    assert stats.distribution_summary == {}


def test_finalize_on_an_all_null_column() -> None:
    acc = ColumnAccumulator()

    acc.update(pl.Series("x", [None, None, None], dtype=pl.Int64))

    stats = acc.finalize()

    assert stats.count == 3
    assert stats.null_ratio == 1.0
    assert stats.cardinality_estimate == 0
    assert stats.minimum is None


def test_hll_registers_and_reservoir_stay_fixed_size_regardless_of_row_count() -> None:
    small = ColumnAccumulator()
    small.update(pl.Series("x", list(range(500))))

    large = ColumnAccumulator()
    large.update(pl.Series("x", list(range(50_000))))

    assert len(small._hll_registers) == len(large._hll_registers) == 4_096
    assert len(small._reservoir) == len(large._reservoir) == 200


# --- Profiler ------------------------------------------------------------


def test_profiler_produces_stats_for_every_schema_column() -> None:
    profiler = Profiler(SCHEMA)

    profile = profiler.profile(_make_chunks(1_000, 100))

    assert set(profile.columns.keys()) == {"id", "name"}
    assert profile.columns["id"].count == 1_000
    assert profile.columns["name"].count == 1_000


def test_profiler_process_then_finalize_matches_the_profile_convenience_method() -> (
    None
):
    manual = Profiler(SCHEMA)
    for chunk in _make_chunks(500, 100):
        manual.process(chunk)
    manual_result = manual.finalize()

    convenience_result = Profiler(SCHEMA).profile(_make_chunks(500, 100))

    assert manual_result.columns["id"].count == convenience_result.columns["id"].count
    assert (
        manual_result.columns["id"].minimum == convenience_result.columns["id"].minimum
    )
    assert (
        manual_result.columns["id"].maximum == convenience_result.columns["id"].maximum
    )


def test_streaming_profiler_memory_and_accumulator_state_are_bounded() -> None:
    """The critical streaming test, per CLAUDE.md section 5: "a test that
    feeds a large dataset in small chunks and asserts peak memory stays
    roughly flat is the proof the streaming profiler works. A test on a
    tiny in-memory frame does not prove it."

    Two independent proofs of boundedness, both from the same large run:
    1. peak memory (tracemalloc) does not grow with row count;
    2. the accumulators' internal fixed-size structures (the HyperLogLog
       register array and the reservoir sample) stay at their caps rather
       than growing with the number of rows seen.
    """

    def run(total_rows: int, rows_per_chunk: int) -> tuple[int, Profiler]:
        gc.collect()
        profiler = Profiler(SCHEMA)
        tracemalloc.start()
        for chunk in _make_chunks(total_rows, rows_per_chunk):
            profiler.process(chunk)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak, profiler

    small_peak, _ = run(total_rows=2_000, rows_per_chunk=200)
    large_peak, large_profiler = run(total_rows=100_000, rows_per_chunk=200)

    # A 50x increase in row count should not translate into anywhere near a
    # 50x increase in peak memory if the profiler is genuinely bounded;
    # generous headroom covers chunk-sized transients and allocator noise
    # while still catching real O(dataset) accumulation. Empirically, peak
    # memory does not grow at all between these two runs (it's actually
    # slightly *lower* at 100,000 rows, since the small run's peak is
    # dominated by one-time setup allocations).
    assert large_peak < small_peak * 3

    # The accumulator state itself must not have grown to accommodate
    # 100,000 rows: the HLL sketch is always exactly 4,096 registers, and
    # the reservoir sample is capped at 200 entries, by construction.
    for accumulator in large_profiler._accumulators.values():
        assert len(accumulator._hll_registers) == 4_096
        assert len(accumulator._reservoir) <= 200


def test_metric_correctness_against_a_hand_computed_small_dataset() -> None:
    """Cross-check every reported metric against values computed by hand
    for a small, fully-known dataset, fed across two chunks to exercise
    the real merge-across-chunks path (not a single one-shot update).

    score (Int64), 7 rows, split [10, 20, 20, None] | [30, 10, None]:
      count=7, nulls=2, null_ratio=2/7; non-null=[10,20,20,30,10] (5 values)
      min=10, max=30; distinct={10,20,30} -> cardinality=3
      duplicate% = (1 - 3/5) * 100 = 40.0
      not a string column -> average_string_length is None

    label (String), 7 rows, split ["apple","banana",None,"apple"] |
    ["cherry","apple","banana"]:
      count=7, nulls=1, null_ratio=1/7
      non-null=["apple","banana","apple","cherry","apple","banana"] (6 values)
      min="apple", max="cherry"; distinct={"apple","banana","cherry"} -> cardinality=3
      duplicate% = (1 - 3/6) * 100 = 50.0
      lengths: apple=5, banana=6, apple=5, cherry=6, apple=5, banana=6
      -> average_string_length = 33/6 = 5.5

    Both non-null counts (5 and 6) are well under the 200-entry reservoir
    cap, so no sampling loss occurs: the reservoir holds every non-null
    value, and distribution_summary's min/max are therefore exactly the
    true min/max, not sampled approximations.
    """
    schema = Schema(columns={"score": pl.Int64(), "label": pl.String()})
    profiler = Profiler(schema)

    profiler.process(
        Chunk(
            data=pl.DataFrame(
                {
                    "score": pl.Series([10, 20, 20, None], dtype=pl.Int64),
                    "label": ["apple", "banana", None, "apple"],
                }
            ),
            index=0,
            is_last=False,
            source_schema=schema,
        )
    )
    profiler.process(
        Chunk(
            data=pl.DataFrame(
                {
                    "score": pl.Series([30, 10, None], dtype=pl.Int64),
                    "label": ["cherry", "apple", "banana"],
                }
            ),
            index=1,
            is_last=True,
            source_schema=schema,
        )
    )
    profile = profiler.finalize()

    score = profile.columns["score"]
    assert score.dtype == pl.Int64()
    assert score.count == 7
    assert score.null_ratio == pytest.approx(2 / 7)
    assert score.minimum == 10
    assert score.maximum == 30
    assert score.cardinality_estimate == 3
    assert score.duplicate_percentage == pytest.approx(40.0)
    assert score.average_string_length is None
    assert score.distribution_summary["min"] == 10
    assert score.distribution_summary["max"] == 30

    label = profile.columns["label"]
    assert label.dtype == pl.String()
    assert label.count == 7
    assert label.null_ratio == pytest.approx(1 / 7)
    assert label.minimum == "apple"
    assert label.maximum == "cherry"
    assert label.cardinality_estimate == 3
    assert label.duplicate_percentage == pytest.approx(50.0)
    assert label.average_string_length == pytest.approx(5.5)
    assert label.distribution_summary["min"] == "apple"
    assert label.distribution_summary["max"] == "cherry"
