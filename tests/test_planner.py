from dataclasses import dataclass, field

import polars as pl

from core.optimization_profile import ColumnProfile, OptimizationProfile
from core.profiler import ColumnStats
from core.schema_analyzer import ColumnKind
from core.types import Chunk
from engine.planner import Planner
from optimizers.base import OptimizationDecision, OptimizationResult
from plugins.registry import OptimizerRegistry

_STATS = ColumnStats(
    dtype=pl.Int64(),
    count=100,
    null_ratio=0.0,
    minimum=0,
    maximum=100,
    cardinality_estimate=100,
    duplicate_percentage=0.0,
    average_string_length=None,
    distribution_summary={},
)
PROFILE = OptimizationProfile(
    columns={"age": ColumnProfile(name="age", kind=ColumnKind.INTEGER, stats=_STATS)}
)


@dataclass
class _FakeOptimizer:
    """Minimal stand-in for the Optimizer protocol: returns a fixed,
    pre-built decision regardless of the profile it's given, used only to
    drive Planner in these tests."""

    name: str
    decision: OptimizationDecision

    def evaluate(self, profile: OptimizationProfile) -> OptimizationDecision:
        return self.decision

    def apply(
        self, chunk: Chunk, decision: OptimizationDecision
    ) -> tuple[Chunk, OptimizationResult]:
        raise NotImplementedError("planner tests never call apply()")

    def is_lossless(self, decision: OptimizationDecision) -> bool:
        raise NotImplementedError("planner tests never call is_lossless()")


@dataclass
class _RecordingOptimizer:
    """Records every profile it's asked to evaluate, to prove the planner
    consults only the profile and never touches chunk data."""

    name: str
    decision: OptimizationDecision
    received_profiles: list[OptimizationProfile] = field(default_factory=list)

    def evaluate(self, profile: OptimizationProfile) -> OptimizationDecision:
        self.received_profiles.append(profile)
        return self.decision

    def apply(
        self, chunk: Chunk, decision: OptimizationDecision
    ) -> tuple[Chunk, OptimizationResult]:
        raise NotImplementedError("not exercised by this test")

    def is_lossless(self, decision: OptimizationDecision) -> bool:
        raise NotImplementedError("not exercised by this test")


def _decision(
    applicable: bool, estimated_saving_bytes: int = 0
) -> OptimizationDecision:
    return OptimizationDecision(
        applicable=applicable,
        columns=["age"],
        rationale="fake",
        estimated_saving_bytes=estimated_saving_bytes,
    )


def test_empty_registry_produces_an_empty_plan() -> None:
    plan = Planner().plan(PROFILE, OptimizerRegistry())

    assert plan.steps == []


def test_inapplicable_decisions_are_excluded_from_the_plan() -> None:
    registry = OptimizerRegistry()
    registry.register(_FakeOptimizer(name="noop", decision=_decision(applicable=False)))

    plan = Planner().plan(PROFILE, registry)

    assert plan.steps == []


def test_applicable_decision_is_included_in_the_plan() -> None:
    registry = OptimizerRegistry()
    registry.register(
        _FakeOptimizer(
            name="alpha",
            decision=_decision(applicable=True, estimated_saving_bytes=500),
        )
    )

    plan = Planner().plan(PROFILE, registry)

    assert len(plan.steps) == 1
    assert plan.steps[0].optimizer_name == "alpha"
    assert plan.steps[0].decision.estimated_saving_bytes == 500


def test_plan_is_ordered_by_estimated_saving_bytes_descending() -> None:
    registry = OptimizerRegistry()
    registry.register(_FakeOptimizer(name="small", decision=_decision(True, 100)))
    registry.register(_FakeOptimizer(name="large", decision=_decision(True, 900)))
    registry.register(_FakeOptimizer(name="medium", decision=_decision(True, 400)))

    plan = Planner().plan(PROFILE, registry)

    assert [step.optimizer_name for step in plan.steps] == ["large", "medium", "small"]


def test_mixed_applicable_and_inapplicable_optimizers() -> None:
    registry = OptimizerRegistry()
    registry.register(_FakeOptimizer(name="skip", decision=_decision(False, 999)))
    registry.register(_FakeOptimizer(name="keep", decision=_decision(True, 10)))

    plan = Planner().plan(PROFILE, registry)

    assert [step.optimizer_name for step in plan.steps] == ["keep"]


def test_plan_is_deterministic_for_the_same_profile() -> None:
    registry = OptimizerRegistry()
    registry.register(_FakeOptimizer(name="alpha", decision=_decision(True, 300)))
    registry.register(_FakeOptimizer(name="beta", decision=_decision(True, 700)))
    registry.register(_FakeOptimizer(name="gamma", decision=_decision(False, 999)))
    # Equal estimated savings: determinism must hold for ties too, not just
    # for values that already sort unambiguously.
    registry.register(_FakeOptimizer(name="delta", decision=_decision(True, 300)))

    first = Planner().plan(PROFILE, registry)
    second = Planner().plan(PROFILE, registry)

    assert first == second
    assert [step.optimizer_name for step in first.steps] == ["beta", "alpha", "delta"]


def test_planner_only_consults_the_profile_and_never_touches_chunk_data() -> None:
    optimizer = _RecordingOptimizer(name="rec", decision=_decision(True, 1))
    registry = OptimizerRegistry()
    registry.register(optimizer)

    Planner().plan(PROFILE, registry)

    assert len(optimizer.received_profiles) == 1
    assert optimizer.received_profiles[0] is PROFILE
