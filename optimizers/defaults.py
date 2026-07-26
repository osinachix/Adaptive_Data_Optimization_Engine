from optimizers.dictionary_encoding import DictionaryEncodingOptimizer
from optimizers.numeric_downcast import NumericDowncastOptimizer
from plugins.registry import OptimizerRegistry


def build_default_registry() -> OptimizerRegistry:
    """The built-in optimizers shipped by default. Shared by every door
    into the engine (CLI, GUI, or programmatic use) so they stay in sync;
    adding a new optimizer means registering it here once, never editing
    the engine itself (invariant I5)."""
    registry = OptimizerRegistry()
    registry.register(NumericDowncastOptimizer())
    registry.register(DictionaryEncodingOptimizer())
    return registry
