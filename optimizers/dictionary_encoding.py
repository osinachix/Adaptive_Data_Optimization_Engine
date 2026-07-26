import polars as pl

from core.optimization_profile import OptimizationProfile
from core.schema_analyzer import ColumnKind
from core.types import Chunk
from optimizers.base import OptimizationDecision, OptimizationResult

# Polars' Categorical dtype always stores physical codes as UInt32,
# regardless of how many distinct categories exist (verified empirically:
# 2, 10, and 300 categories all produced UInt32 physical codes) - so the
# per-value code cost below is a fixed 4 bytes, not derived from
# cardinality the way numeric_downcast derives its target int width.
_CODE_WIDTH_BYTES = 4


class DictionaryEncodingOptimizer:
    """Encodes low-cardinality string columns as a dictionary of unique
    values plus integer codes (Polars' Categorical dtype), saving space
    when the same strings repeat often.

    Always lossless: encoding then decoding reproduces the exact original
    strings (including nulls and empty strings, verified by round-trip
    test), since it's purely a different physical representation of the
    same values, not a precision-reducing transformation.
    """

    name = "dictionary_encoding"

    def evaluate(self, profile: OptimizationProfile) -> OptimizationDecision:
        """Find string columns where the estimated dictionary-encoded size
        (unique values stored once, plus one 4-byte code per row) is
        smaller than the estimated current size (the full string repeated
        per row)."""
        columns: list[str] = []
        savings_by_column: dict[str, int] = {}

        for name, column in profile.columns.items():
            if column.kind is not ColumnKind.STRING:
                continue
            stats = column.stats
            if stats.average_string_length is None or stats.count == 0:
                continue

            current_bytes = stats.average_string_length * stats.count
            dictionary_bytes = stats.average_string_length * stats.cardinality_estimate
            new_bytes = dictionary_bytes + _CODE_WIDTH_BYTES * stats.count
            saving = round(current_bytes - new_bytes)
            if saving <= 0:
                continue  # dictionary + codes would be no smaller; skip

            columns.append(name)
            savings_by_column[name] = saving

        if not columns:
            return OptimizationDecision(
                applicable=False,
                columns=[],
                rationale="no string column has a low enough cardinality to benefit",
                estimated_saving_bytes=0,
            )

        columns.sort()
        rationale = "; ".join(
            f"'{col}': ~{profile.columns[col].stats.cardinality_estimate} distinct "
            f"values across {profile.columns[col].stats.count} rows"
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
        """Cast each decided column in this chunk to Polars' Categorical
        dtype."""
        if not decision.columns:
            return chunk, OptimizationResult(
                columns_changed=[],
                bytes_before=0,
                bytes_after=0,
                lossless=True,
                detail="no columns to encode",
            )

        bytes_before = int(chunk.data.select(decision.columns).estimated_size())
        new_data = chunk.data.with_columns(
            chunk.data[col].cast(pl.Categorical) for col in decision.columns
        )
        bytes_after = int(new_data.select(decision.columns).estimated_size())

        new_chunk = Chunk(
            data=new_data,
            index=chunk.index,
            is_last=chunk.is_last,
            source_schema=chunk.source_schema,
        )
        detail = "; ".join(
            f"{col}: {chunk.data[col].dtype} -> Categorical" for col in decision.columns
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
        """Dictionary encoding is a lossless representation change: the
        same strings are recoverable exactly from the code + dictionary."""
        return True
