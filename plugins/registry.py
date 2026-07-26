from optimizers.base import Optimizer


class OptimizerRegistry:
    """Discovers and holds available optimizers. Optimizers register here;
    the engine asks here. Adding an optimizer means registering it, never
    editing the engine."""

    def __init__(self) -> None:
        self._optimizers: dict[str, Optimizer] = {}

    def register(self, optimizer: Optimizer) -> None:
        """Add an optimizer to the registry, keyed by its name. Registering
        under a name that already exists replaces the previous optimizer."""
        self._optimizers[optimizer.name] = optimizer

    def all(self) -> list[Optimizer]:
        """Return every registered optimizer."""
        return list(self._optimizers.values())

    def get(self, name: str) -> Optimizer | None:
        """Look up a registered optimizer by name, or None if it isn't
        registered."""
        return self._optimizers.get(name)
