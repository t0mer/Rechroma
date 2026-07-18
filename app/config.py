"""Application settings with explicit precedence CLI > env > config.yaml > defaults.

Only the keys the colorization core needs are modelled here; later slices extend
``Settings`` with job/API/telegram fields (CLAUDE.md §3).
"""

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PREFIX = "RECHROMA_"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix=ENV_PREFIX, extra="ignore", populate_by_name=True)

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

    # Telegram (token via env only; empty token disables the bot)
    telegram_bot_token: str | None = None
    telegram_webhook_url: str | None = None  # set to use webhook mode instead of polling
    allowed_chat_ids: list[int] = []  # allowlist; empty = only admins may use the bot
    admin_chat_ids: list[int] = []

    # Video (v2)
    video_enabled: bool = True
    video_max_seconds: int = 30
    video_max_resolution: int = 1080  # longer side; larger clips are rejected
    video_max_fps: int = 24
    video_max_mb: int = 200
    telegram_video_max_mb: int = 20
    video_smoothing_window: int = 5  # temporal chroma window; 1 disables
    video_render_factor: int = 21
    video_crf: int = 18
    video_workspace_dir: Path | None = None  # defaults to ${data_dir}/video

    # Animate (living portrait) — standalone, pluggable engines
    animate_enabled: bool = True
    animate_engine: Literal["tpsmm", "diffusion", "cloud"] = "tpsmm"  # default engine
    animate_driver: str = "subtle"
    animate_max_frames: int = 120
    animate_max_resolution: int = 2048
    animate_crf: int = 18
    animate_workspace_dir: Path | None = None  # defaults to ${data_dir}/animate

    # Animate: local diffusion engine (GPU only, opt-in) — Wan2.1-I2V by default
    animate_diffusion_enabled: bool = False
    animate_diffusion_model: str = "Wan-AI/Wan2.1-I2V-1.3B-Diffusers"
    animate_diffusion_prompt: str = ""  # optional guidance; blank lets the model decide
    animate_diffusion_frames: int = 49
    animate_diffusion_steps: int = 30
    animate_diffusion_fps: int = 16

    # Animate: cloud engine (opt-in) — Replicate; token via env only
    animate_cloud_enabled: bool = False
    animate_cloud_model: str = "wan-video/wan-2.1-i2v-480p"  # a Replicate owner/name slug
    animate_cloud_prompt: str = ""
    replicate_api_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("REPLICATE_API_TOKEN", "RECHROMA_REPLICATE_API_TOKEN"),
    )

    @field_validator("allowed_chat_ids", "admin_chat_ids", mode="before")
    @classmethod
    def _split_ids(cls, v: Any) -> Any:
        """Accept a comma-separated string (env-friendly) or a real list."""
        if isinstance(v, str):
            return [int(part) for part in v.replace(",", " ").split()]
        return v

    @model_validator(mode="after")
    def _default_workspaces(self) -> "Settings":
        if self.video_workspace_dir is None:
            self.video_workspace_dir = self.data_dir / "video"
        if self.animate_workspace_dir is None:
            self.animate_workspace_dir = self.data_dir / "animate"
        return self


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
