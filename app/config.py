"""Application settings with explicit precedence CLI > env > config.yaml > defaults.

Only the keys the colorization core needs are modelled here; later slices extend
``Settings`` with job/API/telegram fields (CLAUDE.md §3).
"""

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PREFIX = "RECHROMA_"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix=ENV_PREFIX, extra="ignore")

    device: Literal["auto", "cuda", "cpu"] = "auto"
    models_dir: Path = Path("/data/models")
    model_base_url: str | None = None
    render_factor: int | None = None
    log_level: str = "info"

    # Web / API
    port: int = 8000
    data_dir: Path = Path("/data/jobs")  # SQLite DB + uploaded/result images
    web_auth_token: str | None = None  # optional shared bearer token; unset = open (warned)
    max_upload_mb: int = 25
    workers: int = 1
    retention_hours: float = 24.0  # delete originals + results after this; 0 = immediately
    rate_limit_per_hour: int = 10  # per source; 0 disables


def load_settings(config_path: Path | None = None, **overrides: Any) -> Settings:
    """Build ``Settings`` layering CLI overrides > env vars > YAML file > defaults.

    ``overrides`` values that are ``None`` are ignored so unset CLI flags do not
    clobber lower-precedence sources.
    """
    values: dict[str, Any] = {}

    if config_path is not None and config_path.exists():
        loaded = yaml.safe_load(config_path.read_text()) or {}
        values.update({k: v for k, v in loaded.items() if v is not None})

    # Env beats YAML: pull each field's value from the environment when present.
    env_settings = Settings()
    for field in Settings.model_fields:
        if f"{ENV_PREFIX}{field.upper()}" in os.environ:
            values[field] = getattr(env_settings, field)

    # CLI overrides win over everything.
    values.update({k: v for k, v in overrides.items() if v is not None})

    return Settings(**values)
