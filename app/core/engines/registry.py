"""Animate-engine registry: construct engines and report availability.

The registry is the single place that knows the set of engines and how to build
each from :class:`~app.config.Settings`. Concrete engines are imported lazily so
importing the registry never pulls in heavy optional deps (``diffusers``, etc.).
"""

from typing import TYPE_CHECKING

from .base import AnimateEngine, EngineInfo, EngineUnavailable

if TYPE_CHECKING:
    from app.config import Settings

# Display/selection order.
ENGINE_NAMES: tuple[str, ...] = ("tpsmm", "diffusion", "cloud")


def _construct(name: str, settings: "Settings") -> AnimateEngine:
    if name == "tpsmm":
        from .tpsmm import TPSMMEngine

        return TPSMMEngine(settings)
    if name == "diffusion":
        from .diffusion import DiffusionEngine

        return DiffusionEngine(settings)
    if name == "cloud":
        from .cloud import CloudEngine

        return CloudEngine(settings)
    raise EngineUnavailable(f"unknown animate engine: {name}")


def list_engine_infos(settings: "Settings") -> list[EngineInfo]:
    """Availability snapshot for every engine, in display order."""
    return [_construct(name, settings).info(settings) for name in ENGINE_NAMES]


def resolve_engine_name(requested: str | None, settings: "Settings") -> str:
    """Pick an engine name: the request if usable, else the configured default,
    else the first available engine. Raises if nothing is available."""
    infos = {i.name: i for i in list_engine_infos(settings)}
    for candidate in (requested, settings.animate_engine):
        if candidate and infos.get(candidate) and infos[candidate].available:
            return candidate
    for info in infos.values():
        if info.available:
            return info.name
    raise EngineUnavailable("no animate engine is available")


def build_engine(name: str, settings: "Settings") -> AnimateEngine:
    """Construct a ready-to-run engine, or raise :class:`EngineUnavailable`."""
    if name not in ENGINE_NAMES:
        raise EngineUnavailable(f"unknown animate engine: {name}")
    engine = _construct(name, settings)
    ok, reason = engine.check(settings)
    if not ok:
        raise EngineUnavailable(reason or f"engine '{name}' is unavailable")
    return engine
