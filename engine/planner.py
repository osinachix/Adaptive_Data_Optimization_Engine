from dataclasses import dataclass

from core.optimization_profile import OptimizationProfile
from optimizers.base import OptimizationDecision
from plugins.registry import OptimizerRegistry


@dataclass(frozen=True)
class PlannedStep:
    """One applicable optimizer decision, placed in execution order."""

    optimizer_name: str
    decision: OptimizationDecision


@dataclass(frozen=True)
class ExecutionPlan:
    """The ordered sequence of optimizer decisions to apply: the most
    impactful (by estimated_saving_bytes) first. Derived from the profile
    alone; no chunk data has been touched to produce it."""

    steps: list[PlannedStep]


class Planner:
    """Turns an OptimizationProfile and a set of registered optimizers into
    an ordered execution plan. Consults only the profile, via each
    optimizer's evaluate(); never touches chunk data."""

    def plan(
        self, profile: OptimizationProfile, registry: OptimizerRegistry
    ) -> ExecutionPlan:
        """Ask every registered optimizer to evaluate the profile, keep
        only the applicable decisions, and order them by estimated saving,
        largest first."""
        applicable_steps = []
        for optimizer in registry.all():
            decision = optimizer.evaluate(profile)
            if decision.applicable:
                applicable_steps.append(
                    PlannedStep(optimizer_name=optimizer.name, decision=decision)
                )

        applicable_steps.sort(
            key=lambda step: step.decision.estimated_saving_bytes, reverse=True
        )
        return ExecutionPlan(steps=applicable_steps)
