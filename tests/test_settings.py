import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from config.settings import ADOESettings
from core.mode import OptimizationMode


@pytest.fixture
def isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """ADOESettings looks for adoe.toml relative to the current directory;
    run each test from an isolated tmp_path so tests never see a real
    project-level adoe.toml (there isn't one, but this keeps the test
    honest regardless) or interfere with each other via ADOE_* env vars."""
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("ADOE_"):
            monkeypatch.delenv(key, raising=False)
    yield tmp_path


def test_defaults_apply_when_nothing_else_is_set(isolated_cwd: Path) -> None:
    settings = ADOESettings()

    assert settings.mode is OptimizationMode.LOSSLESS
    assert settings.rows_per_chunk == 10_000
    assert settings.log_level == "INFO"


def test_toml_file_overrides_defaults(isolated_cwd: Path) -> None:
    (isolated_cwd / "adoe.toml").write_text(
        'mode = "balanced"\nrows_per_chunk = 5000\n'
    )

    settings = ADOESettings()

    assert settings.mode is OptimizationMode.BALANCED
    assert settings.rows_per_chunk == 5000


def test_env_var_overrides_toml_file(
    isolated_cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (isolated_cwd / "adoe.toml").write_text('mode = "balanced"\n')
    monkeypatch.setenv("ADOE_MODE", "aggressive")

    settings = ADOESettings()

    assert settings.mode is OptimizationMode.AGGRESSIVE


def test_explicit_override_wins_over_everything(
    isolated_cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (isolated_cwd / "adoe.toml").write_text(
        'mode = "balanced"\nrows_per_chunk = 5000\n'
    )
    monkeypatch.setenv("ADOE_MODE", "aggressive")

    # Simulates a CLI flag the user actually passed.
    settings = ADOESettings(mode=OptimizationMode.LOSSLESS)

    assert settings.mode is OptimizationMode.LOSSLESS  # explicit override wins
    # Untouched field still falls through to the toml file.
    assert settings.rows_per_chunk == 5000
