from dataclasses import dataclass

from core.profiler import ColumnStats, DatasetProfile
from core.schema_analyzer import ColumnKind, SchemaAnalysis


@dataclass(frozen=True)
class ColumnProfile:
    """Everything an optimizer needs to know about one column to decide
    applicability: its broad type classification and its finalized
    statistics. No raw data; decisions are made from this alone."""

    name: str
    kind: ColumnKind
    stats: ColumnStats


@dataclass(frozen=True)
class OptimizationProfile:
    """The finalized per-column metrics the planner and optimizers consume.
    Built once the Profiler and SchemaAnalyzer have both finished; nothing
    downstream touches raw data again until an optimizer's apply() runs on
    a chunk."""

    columns: dict[str, ColumnProfile]


def build_optimization_profile(
    dataset_profile: DatasetProfile, schema_analysis: SchemaAnalysis
) -> OptimizationProfile:
    """Combine a Profiler's per-column statistics with a SchemaAnalyzer's
    per-column type classification into the planner-facing
    OptimizationProfile. Both inputs must describe the same columns."""
    if dataset_profile.columns.keys() != schema_analysis.columns.keys():
        raise ValueError(
            "dataset_profile and schema_analysis must describe the same "
            f"columns: {sorted(dataset_profile.columns)} != "
            f"{sorted(schema_analysis.columns)}"
        )
    return OptimizationProfile(
        columns={
            name: ColumnProfile(
                name=name, kind=schema_analysis.columns[name], stats=stats
            )
            for name, stats in dataset_profile.columns.items()
        }
    )
