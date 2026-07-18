"""Pluggable Animate engines (tpsmm / diffusion / cloud).

Public surface: the engine contract and the registry helpers. Concrete engines
are imported lazily by the registry so optional heavy deps stay out of the import
path until an engine is actually built.
"""

from .base import (
    AnimateCancelled,
    AnimateEngine,
    EngineError,
    EngineInfo,
    EngineUnavailable,
)
from .registry import (
    ENGINE_NAMES,
    build_engine,
    list_engine_infos,
    resolve_engine_name,
)

__all__ = [
    "ENGINE_NAMES",
    "AnimateCancelled",
    "AnimateEngine",
    "EngineError",
    "EngineInfo",
    "EngineUnavailable",
    "build_engine",
    "list_engine_infos",
    "resolve_engine_name",
]
