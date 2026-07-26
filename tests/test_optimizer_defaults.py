from optimizers.defaults import build_default_registry
from optimizers.dictionary_encoding import DictionaryEncodingOptimizer
from optimizers.numeric_downcast import NumericDowncastOptimizer


def test_build_default_registry_registers_both_shipped_optimizers() -> None:
    registry = build_default_registry()

    names = {optimizer.name for optimizer in registry.all()}
    assert names == {NumericDowncastOptimizer().name, DictionaryEncodingOptimizer().name}


def test_build_default_registry_returns_a_fresh_registry_each_call() -> None:
    """CLI and GUI each call this once per invocation; registries must not
    share mutable state across calls."""
    first = build_default_registry()
    second = build_default_registry()

    assert first is not second
    assert first.all() is not second.all()
