from enum import StrEnum


class OptimizationMode(StrEnum):
    """Which class of optimizations may run and how strictly the output is
    validated against the original. LOSSLESS is the default and the only
    mode that requires an exact value-level reconstruction check; BALANCED
    and AGGRESSIVE permit optimizers that trade precision for size, so
    validation for them checks structure (schema, row count, column
    integrity) but not exact value equality."""

    LOSSLESS = "lossless"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
