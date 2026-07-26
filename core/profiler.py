import hashlib
import math
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, cast

import polars as pl

from core.types import Chunk, Schema

# HyperLogLog cardinality sketch. p (precision) controls the number of
# registers (m = 2**p); higher p trades memory for accuracy. p=12 -> 4096
# registers (~32KB), a fixed cost independent of dataset size, with a
# standard error of roughly 1.04/sqrt(m) =~ 1.6%.
_HLL_PRECISION = 12
_HLL_NUM_REGISTERS = 1 << _HLL_PRECISION
_HLL_ALPHA = 0.7213 / (1 + 1.079 / _HLL_NUM_REGISTERS)
_HLL_HASH_BITS = 64
_HLL_REMAINING_BITS = _HLL_HASH_BITS - _HLL_PRECISION

# Reservoir sample size backing the distribution summary: fixed and
# independent of dataset size (Algorithm R reservoir sampling).
_RESERVOIR_SIZE = 200


def _hash64(value: object) -> int:
    """Stable 64-bit hash of a value's canonical string form, used to drive
    the HyperLogLog sketch. Python's built-in hash() is neither stable
    across runs nor well distributed for sequential integers, both of which
    would break the sketch."""
    digest = hashlib.blake2b(repr(value).encode("utf-8"), digest_size=8)
    return int.from_bytes(digest.digest(), byteorder="big")


def _hll_index_and_rank(h: int) -> tuple[int, int]:
    """Split a 64-bit hash into a register index (top p bits) and a rank
    (1 + leading zero count of the remaining bits)."""
    index = h >> _HLL_REMAINING_BITS
    remaining = h & ((1 << _HLL_REMAINING_BITS) - 1)
    if remaining == 0:
        rank = _HLL_REMAINING_BITS + 1
    else:
        rank = _HLL_REMAINING_BITS - remaining.bit_length() + 1
    return index, rank


def _hll_estimate(registers: list[int]) -> int:
    """Standard HyperLogLog cardinality estimator with small- and
    large-range corrections."""
    m = len(registers)
    indicator_sum = sum(2.0**-r for r in registers)
    raw_estimate = _HLL_ALPHA * m * m / indicator_sum

    zero_registers = registers.count(0)
    if raw_estimate <= 2.5 * m and zero_registers > 0:
        estimate = m * math.log(m / zero_registers)  # linear counting
    elif raw_estimate <= (1 / 30) * (1 << 32):
        estimate = raw_estimate
    else:
        estimate = -(1 << 32) * math.log(1 - raw_estimate / (1 << 32))
    return max(0, round(estimate))


def _quantile(ordered_sample: list[object], q: float) -> object:
    """Nearest-rank quantile from an already-sorted sample."""
    index = min(len(ordered_sample) - 1, max(0, round(q * (len(ordered_sample) - 1))))
    return ordered_sample[index]


def _summarize_distribution(reservoir: list[object]) -> dict[str, object]:
    """A small distribution summary (min, quartiles, max) derived from the
    bounded reservoir sample. Empty if no non-null values were observed."""
    if not reservoir:
        return {}
    ordered = sorted(cast(list[Any], reservoir))
    return {
        "min": ordered[0],
        "p25": _quantile(ordered, 0.25),
        "p50": _quantile(ordered, 0.50),
        "p75": _quantile(ordered, 0.75),
        "max": ordered[-1],
    }


@dataclass(frozen=True)
class ColumnStats:
    """Final, reported statistics for one column, derived once the stream
    is exhausted from its ColumnAccumulator's bounded running state."""

    dtype: pl.DataType | None
    count: int
    null_ratio: float
    minimum: object | None
    maximum: object | None
    cardinality_estimate: int  # approximate; see ColumnAccumulator
    duplicate_percentage: float  # approximate; derived from cardinality_estimate
    average_string_length: float | None  # None for non-string columns
    distribution_summary: dict[str, object]


@dataclass
class ColumnAccumulator:
    """Fixed-size running state for one column across all chunks. Its memory
    footprint must not grow with the number of rows seen."""

    count: int = 0
    nulls: int = 0
    minimum: object | None = None
    maximum: object | None = None
    dtype: pl.DataType | None = None
    # cardinality: a HyperLogLog sketch (a fixed-size register array), not
    # an unbounded set of distinct values.
    _hll_registers: list[int] = field(default_factory=lambda: [0] * _HLL_NUM_REGISTERS)
    _string_length_sum: int = 0
    _string_length_count: int = 0
    # distribution summary: a fixed-size reservoir sample, not all values.
    _reservoir: list[object] = field(default_factory=list)
    _reservoir_seen: int = 0
    _rng: random.Random = field(default_factory=lambda: random.Random(0))

    def update(self, column_chunk: pl.Series) -> None:
        """Fold one chunk's worth of a column into the running state. O(chunk),
        not O(dataset)."""
        if self.dtype is None:
            self.dtype = column_chunk.dtype

        self.count += len(column_chunk)
        self.nulls += column_chunk.null_count()

        non_null = column_chunk.drop_nulls()
        if len(non_null) == 0:
            return

        chunk_min = non_null.min()
        chunk_max = non_null.max()
        if chunk_min is not None and (
            self.minimum is None or cast(Any, chunk_min) < cast(Any, self.minimum)
        ):
            self.minimum = chunk_min
        if chunk_max is not None and (
            self.maximum is None or cast(Any, chunk_max) > cast(Any, self.maximum)
        ):
            self.maximum = chunk_max

        if non_null.dtype == pl.String:
            self._string_length_sum += int(non_null.str.len_chars().sum() or 0)
            self._string_length_count += len(non_null)

        for value in non_null.to_list():
            self._observe(value)

    def _observe(self, value: object) -> None:
        """Fold one non-null value into the HLL sketch and the reservoir
        sample. Both structures are fixed-size regardless of how many
        values are observed."""
        h = _hash64(value)
        index, rank = _hll_index_and_rank(h)
        if rank > self._hll_registers[index]:
            self._hll_registers[index] = rank

        self._reservoir_seen += 1
        if len(self._reservoir) < _RESERVOIR_SIZE:
            self._reservoir.append(value)
        else:
            j = self._rng.randint(0, self._reservoir_seen - 1)
            if j < _RESERVOIR_SIZE:
                self._reservoir[j] = value

    def finalize(self) -> ColumnStats:
        """Derive the reported metrics once the stream is exhausted."""
        non_null_count = self.count - self.nulls
        null_ratio = (self.nulls / self.count) if self.count else 0.0

        cardinality_estimate = (
            min(_hll_estimate(self._hll_registers), non_null_count)
            if non_null_count
            else 0
        )
        duplicate_percentage = (
            max(0.0, (1 - cardinality_estimate / non_null_count) * 100)
            if non_null_count
            else 0.0
        )
        average_string_length = (
            self._string_length_sum / self._string_length_count
            if self._string_length_count
            else None
        )

        return ColumnStats(
            dtype=self.dtype,
            count=self.count,
            null_ratio=null_ratio,
            minimum=self.minimum,
            maximum=self.maximum,
            cardinality_estimate=cardinality_estimate,
            duplicate_percentage=duplicate_percentage,
            average_string_length=average_string_length,
            distribution_summary=_summarize_distribution(self._reservoir),
        )


@dataclass(frozen=True)
class DatasetProfile:
    """Per-column statistics for a whole dataset, produced by Profiler once
    its Chunk stream is exhausted."""

    columns: dict[str, ColumnStats]


class Profiler:
    """Consumes a stream of Chunk objects and produces per-column
    statistics, using one bounded ColumnAccumulator per column. Never
    retains rows: each chunk is folded into the accumulators and then
    discarded."""

    def __init__(self, schema: Schema) -> None:
        self._accumulators: dict[str, ColumnAccumulator] = {
            name: ColumnAccumulator() for name in schema.columns
        }

    def process(self, chunk: Chunk) -> None:
        """Fold one chunk into the running per-column accumulators. Cost is
        O(chunk size), independent of how many chunks came before it."""
        for name, accumulator in self._accumulators.items():
            accumulator.update(chunk.data[name])

    def finalize(self) -> DatasetProfile:
        """Derive the final per-column statistics once the stream is
        exhausted."""
        return DatasetProfile(
            columns={
                name: accumulator.finalize()
                for name, accumulator in self._accumulators.items()
            }
        )

    def profile(self, chunks: Iterator[Chunk]) -> DatasetProfile:
        """Process an entire Chunk stream and return the finalized profile."""
        for chunk in chunks:
            self.process(chunk)
        return self.finalize()
