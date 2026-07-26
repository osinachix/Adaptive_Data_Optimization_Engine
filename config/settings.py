from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from core.mode import OptimizationMode


class ADOESettings(BaseSettings):
    """Centralized configuration, resolved with predictable precedence,
    highest first: explicit overrides (from CLI flags the user actually
    passed) > environment variables (ADOE_*) > the adoe.toml config file
    (in the current directory) > these defaults. Verified empirically
    against pydantic-settings' documented source-ordering behavior before
    relying on it.
    """

    mode: OptimizationMode = OptimizationMode.LOSSLESS
    rows_per_chunk: int = 10_000
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_prefix="ADOE_",
        toml_file="adoe.toml",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Highest priority first: explicit init kwargs (CLI overrides),
        then environment variables, then the TOML config file, then
        dotenv/secrets, then the field defaults (implicit, always last)."""
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )
