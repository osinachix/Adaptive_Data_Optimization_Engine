from dataclasses import dataclass
from typing import Protocol

from core.optimization_profile import OptimizationProfile
from core.types import Chunk


@dataclass(frozen=True)
class OptimizationDecision:
    """An optimizer's judgement about one dataset, produced from the profile
    before any data is touched. Deterministic given the same profile."""

    applicable: bool  # does this optimizer apply at all?
    columns: list[str]  # which columns it would act on
    rationale: str  # human-readable why, for the report
    estimated_saving_bytes: int  # planner uses this to order the plan


@dataclass(frozen=True)
class OptimizationResult:
    """What an optimizer actually did to one chunk, for the report."""

    columns_changed: list[str]
    bytes_before: int
    bytes_after: int
    lossless: bool  # did this operation preserve exact values?
    detail: str  # specifics, e.g. "int64 -> int16 on col 'age'"


class Optimizer(Protocol):
    name: str
    """Unique identifier, e.g. 'numeric_downcast'."""

    def evaluate(self, profile: OptimizationProfile) -> OptimizationDecision:
        """Decide whether and where this optimizer applies, from the profile
        alone. No data access. Deterministic."""

    def apply(
        self, chunk: Chunk, decision: OptimizationDecision
    ) -> tuple[Chunk, OptimizationResult]:
        """Transform one chunk per the decision. Returns the new chunk and a
        record of what was done. Must be pure with respect to the chunk it is
        given: same input, same output, no shared state between chunks unless
        that state is explicitly threaded and documented."""

    def is_lossless(self, decision: OptimizationDecision) -> bool:
        """True only if apply() guarantees exact reconstruction for this
        decision. The engine refuses to run a non-lossless optimizer in
        lossless mode."""
