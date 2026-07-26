import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-stress",
        action="store_true",
        default=False,
        help="Run stress tests for large streaming workloads (skipped by default)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Stress tests are opt-in: they exercise multi-million-row streaming
    workloads and take much longer than the rest of the suite. Skipping
    them by default (rather than excluding them via addopts) keeps them
    visible in test output as explicitly skipped, with a reason, instead
    of silently absent."""
    if config.getoption("--run-stress"):
        return
    skip_stress = pytest.mark.skip(reason="need --run-stress option to run")
    for item in items:
        if "stress" in item.keywords:
            item.add_marker(skip_stress)
