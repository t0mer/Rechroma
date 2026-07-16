"""Device selection and per-device defaults.

The same codebase runs on CPU and GPU; the only differences are where tensors
live and a few sensible defaults (CLAUDE.md §2, §4). ``auto`` falls back to CPU
when no GPU is present, but an explicit ``cuda`` request errors rather than
silently degrading.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceDefaults:
    """Per-device tuning defaults (CLAUDE.md §4 "Device-aware defaults")."""

    render_factor: int
    max_resolution: int
    default_upscale: int


def resolve_device(pref: str) -> torch.device:
    """Resolve a device preference to a concrete ``torch.device``.

    ``auto`` → CUDA if available else CPU; ``cuda`` → error if unavailable;
    ``cpu`` → CPU.
    """
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available")
        return torch.device("cuda")
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"unknown device preference {pref!r}")


def device_defaults(dev: torch.device) -> DeviceDefaults:
    """Return the default render factor / limits for a device."""
    if dev.type == "cuda":
        return DeviceDefaults(render_factor=35, max_resolution=6000, default_upscale=2)
    return DeviceDefaults(render_factor=25, max_resolution=4000, default_upscale=2)
