import polars as pl
import pytest

from core.optimization_profile import build_optimization_profile
from core.profiler import ColumnStats, DatasetProfile
from core.schema_analyzer import ColumnKind, SchemaAnalysis


def _stats() -> ColumnStats:
    return ColumnStats(
        dtype=pl.Int64(),
        count=10,
        null_ratio=0.0,
        minimum=0,
        maximum=9,
        cardinality_estimate=10,
        duplicate_percentage=0.0,
        average_string_length=None,
        distribution_summary={},
    )


def test_build_optimization_profile_combines_stats_and_kind_per_column() -> None:
    dataset_profile = DatasetProfile(columns={"age": _stats()})
    schema_analysis = SchemaAnalysis(columns={"age": ColumnKind.INTEGER})

    profile = build_optimization_profile(dataset_profile, schema_analysis)

    assert profile.columns["age"].name == "age"
    assert profile.columns["age"].kind == ColumnKind.INTEGER
    assert profile.columns["age"].stats == dataset_profile.columns["age"]


def test_build_optimization_profile_rejects_mismatched_columns() -> None:
    dataset_profile = DatasetProfile(columns={"age": _stats()})
    schema_analysis = SchemaAnalysis(columns={"name": ColumnKind.STRING})

    with pytest.raises(ValueError, match="same columns"):
        build_optimization_profile(dataset_profile, schema_analysis)
