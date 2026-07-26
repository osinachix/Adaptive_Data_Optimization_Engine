from dataclasses import dataclass

from core.optimization_profile import OptimizationProfile
from core.types import Chunk
from optimizers.base import OptimizationDecision, OptimizationResult
from plugins.registry import OptimizerRegistry


@dataclass
class _FakeOptimizer:
    """Minimal stand-in for the Optimizer protocol, used only to drive
    OptimizerRegistry in these tests."""

    name: str

    def evaluate(self, profile: OptimizationProfile) -> OptimizationDecision:
        return OptimizationDecision(
            applicable=True, columns=[], rationale="fake", estimated_saving_bytes=0
        )

    def apply(
        self, chunk: Chunk, decision: OptimizationDecision
    ) -> tuple[Chunk, OptimizationResult]:
        raise NotImplementedError("not exercised by registry tests")

    def is_lossless(self, decision: OptimizationDecision) -> bool:
        raise NotImplementedError("not exercised by registry tests")


def test_register_and_get_returns_the_same_optimizer() -> None:
    registry = OptimizerRegistry()
    optimizer = _FakeOptimizer(name="alpha")

    registry.register(optimizer)

    assert registry.get("alpha") is optimizer


def test_get_returns_none_for_an_unregistered_name() -> None:
    registry = OptimizerRegistry()

    assert registry.get("missing") is None


def test_all_returns_every_registered_optimizer() -> None:
    registry = OptimizerRegistry()
    alpha = _FakeOptimizer(name="alpha")
    beta = _FakeOptimizer(name="beta")

    registry.register(alpha)
    registry.register(beta)

    all_optimizers = registry.all()
    assert len(all_optimizers) == 2
    assert alpha in all_optimizers
    assert beta in all_optimizers


def test_registering_the_same_name_again_replaces_the_previous_optimizer() -> None:
    registry = OptimizerRegistry()
    first = _FakeOptimizer(name="alpha")
    second = _FakeOptimizer(name="alpha")

    registry.register(first)
    registry.register(second)

    assert registry.get("alpha") is second
    assert registry.all() == [second]


def test_empty_registry_returns_no_optimizers() -> None:
    assert OptimizerRegistry().all() == []
