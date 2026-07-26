from typing import cast

import polars as pl

from core.optimization_profile import OptimizationProfile
from core.schema_analyzer import ColumnKind
from core.types import Chunk
from optimizers.base import OptimizationDecision, OptimizationResult

# Signed integer dtypes ordered smallest to largest: (dtype, low, high, itemsize).
# Only signed integers are handled here; unsigned types and float downcasting
# are out of scope for this optimizer (see class docstring).
_INTEGER_BOUNDS: list[tuple[pl.DataType, int, int, int]] = [
    (pl.Int8(), -128, 127, 1),
    (pl.Int16(), -32_768, 32_767, 2),
    (pl.Int32(), -2_147_483_648, 2_147_483_647, 4),
    (pl.Int64(), -9_223_372_036_854_775_808, 9_223_372_036_854_775_807, 8),
]
_ITEMSIZE_BY_DTYPE: dict[type, int] = {
    pl.Int8: 1,
    pl.Int16: 2,
    pl.Int32: 4,
    pl.Int64: 8,
}


def _smallest_fitting_integer_dtype(minimum: int, maximum: int) -> pl.DataType | None:
    """The smallest signed integer dtype whose range exactly contains
    [minimum, maximum], or None if none of the known dtypes fit (should not
    happen for values that already came from a Polars signed-integer
    column, but avoids an unbounded assumption)."""
    for dtype, low, high, _itemsize in _INTEGER_BOUNDS:
        if low <= minimum and maximum <= high:
            return dtype
    return None


class NumericDowncastOptimizer:
    """Downcasts integer columns to the smallest signed integer dtype that
    exactly represents their observed [min, max]. Always lossless: every
    value in that range is represented exactly by the chosen dtype, by
    construction of the bounds check in evaluate().

    Float downcasting is intentionally out of scope here. Unlike integer
    range-fitting, narrowing float precision (e.g. Float64 -> Float32) is
    not guaranteed exact even within range, so it can only run under a
    lossy mode. Nothing in the engine yet plumbs a lossless/balanced/
    aggressive mode through to is_lossless(), and OptimizationDecision
    carries no per-column dtype-kind field, so a single decision mixing
    lossless integer columns with lossy float columns could not report a
    single accurate is_lossless() value. That is a design question for
    when mode support exists, not a detail to vary silently here.
    """

    name = "numeric_downcast"

    def __init__(self) -> None:
        # Populated by evaluate(), consulted by apply(): explicitly
        # threaded, documented state mapping each proposed column to its
        # target dtype. Needed because OptimizationDecision carries only
        # column names, not dtypes. Not mutated per-chunk; apply() itself
        # stays pure with respect to the chunk it is given.
        self._target_dtypes: dict[str, pl.DataType] = {}

    def evaluate(self, profile: OptimizationProfile) -> OptimizationDecision:
        """Find integer columns whose observed [min, max] fits a strictly
        smaller signed integer dtype than the one they currently have."""
        target_dtypes: dict[str, pl.DataType] = {}
        savings_by_column: dict[str, int] = {}

        for name, column in profile.columns.items():
            if column.kind is not ColumnKind.INTEGER:
                continue
            stats = column.stats
            if stats.minimum is None or stats.maximum is None:
                continue  # no non-null values observed; nothing to size

            current_itemsize = _ITEMSIZE_BY_DTYPE.get(type(stats.dtype))
            if current_itemsize is None:
                continue  # not one of the signed dtypes handled here

            minimum = cast(int, stats.minimum)
            maximum = cast(int, stats.maximum)
            target = _smallest_fitting_integer_dtype(minimum, maximum)
            if target is None:
                continue
            target_itemsize = _ITEMSIZE_BY_DTYPE[type(target)]
            if target_itemsize >= current_itemsize:
                continue  # already at (or smaller than) the smallest fit

            target_dtypes[name] = target
            savings_by_column[name] = (current_itemsize - target_itemsize) * stats.count

        self._target_dtypes = target_dtypes

        if not target_dtypes:
            return OptimizationDecision(
                applicable=False,
                columns=[],
                rationale="no integer column has a range that fits a smaller dtype",
                estimated_saving_bytes=0,
            )

        columns = sorted(target_dtypes)
        rationale = "; ".join(
            f"'{col}': {profile.columns[col].stats.dtype} -> {target_dtypes[col]}"
            for col in columns
        )
        return OptimizationDecision(
            applicable=True,
            columns=columns,
            rationale=rationale,
            estimated_saving_bytes=sum(savings_by_column.values()),
        )

    def apply(
        self, chunk: Chunk, decision: OptimizationDecision
    ) -> tuple[Chunk, OptimizationResult]:
        """Cast each decided column in this chunk to its target dtype."""
        if not decision.columns:
            return chunk, OptimizationResult(
                columns_changed=[],
                bytes_before=0,
                bytes_after=0,
                lossless=True,
                detail="no columns to downcast",
            )

        bytes_before = int(chunk.data.select(decision.columns).estimated_size())
        new_data = chunk.data.with_columns(
            chunk.data[col].cast(self._target_dtypes[col]) for col in decision.columns
        )
        bytes_after = int(new_data.select(decision.columns).estimated_size())

        new_chunk = Chunk(
            data=new_data,
            index=chunk.index,
            is_last=chunk.is_last,
            source_schema=chunk.source_schema,
        )
        detail = "; ".join(
            f"{col}: {chunk.data[col].dtype} -> {self._target_dtypes[col]}"
            for col in decision.columns
        )
        result = OptimizationResult(
            columns_changed=list(decision.columns),
            bytes_before=bytes_before,
            bytes_after=bytes_after,
            lossless=True,
            detail=detail,
        )
        return new_chunk, result

    def is_lossless(self, decision: OptimizationDecision) -> bool:
        """Integer range-fit downcasting always preserves exact values."""
        return True
