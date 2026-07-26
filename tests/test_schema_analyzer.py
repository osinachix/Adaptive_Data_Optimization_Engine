import polars as pl

from core.schema_analyzer import ColumnKind, SchemaAnalyzer, classify_dtype
from core.types import Schema


def test_classify_dtype_maps_each_broad_family() -> None:
    assert classify_dtype(pl.Int64()) == ColumnKind.INTEGER
    assert classify_dtype(pl.Float64()) == ColumnKind.FLOAT
    assert classify_dtype(pl.String()) == ColumnKind.STRING
    assert classify_dtype(pl.Boolean()) == ColumnKind.BOOLEAN
    assert classify_dtype(pl.Datetime()) == ColumnKind.DATETIME
    assert classify_dtype(pl.Date()) == ColumnKind.DATETIME
    assert classify_dtype(pl.List(pl.Int64())) == ColumnKind.OTHER


def test_schema_analyzer_classifies_every_column() -> None:
    schema = Schema(
        columns={
            "id": pl.Int64(),
            "price": pl.Float64(),
            "name": pl.String(),
            "active": pl.Boolean(),
            "created_at": pl.Datetime(),
        }
    )

    analysis = SchemaAnalyzer().analyze(schema)

    assert analysis.columns == {
        "id": ColumnKind.INTEGER,
        "price": ColumnKind.FLOAT,
        "name": ColumnKind.STRING,
        "active": ColumnKind.BOOLEAN,
        "created_at": ColumnKind.DATETIME,
    }
