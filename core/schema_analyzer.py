from dataclasses import dataclass
from enum import StrEnum

import polars as pl

from core.types import Schema


class ColumnKind(StrEnum):
    """Broad type family for a column, so downstream optimizers can decide
    applicability without depending on exact Polars dtypes."""

    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    OTHER = "other"


def classify_dtype(dtype: pl.DataType) -> ColumnKind:
    """Map a Polars dtype to its broad ColumnKind."""
    if dtype.is_integer():
        return ColumnKind.INTEGER
    if dtype.is_float():
        return ColumnKind.FLOAT
    if dtype == pl.Boolean:
        return ColumnKind.BOOLEAN
    if dtype == pl.String:
        return ColumnKind.STRING
    if dtype.is_temporal():
        return ColumnKind.DATETIME
    return ColumnKind.OTHER


@dataclass(frozen=True)
class SchemaAnalysis:
    """Per-column type classification for a dataset's Schema."""

    columns: dict[str, ColumnKind]


class SchemaAnalyzer:
    """Classifies each column in a Schema into a broad ColumnKind. Reads
    only the Schema (column names and dtypes); it never touches data."""

    def analyze(self, schema: Schema) -> SchemaAnalysis:
        """Return the ColumnKind classification for every column."""
        return SchemaAnalysis(
            columns={
                name: classify_dtype(dtype) for name, dtype in schema.columns.items()
            }
        )
