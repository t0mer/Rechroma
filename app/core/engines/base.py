"""Animate-engine interface: the pluggable contract behind the Animate feature.

Rechroma's Animate feature has three interchangeable backends (CLAUDE.md §5/§6
extension), selectable per job:

- ``tpsmm``     -- face reenactment (CPU or GPU), the lightweight default.
- ``diffusion`` -- local generative image-to-video (GPU only, opt-in).
- ``cloud``     -- a hosted image-to-video provider (opt-in, sends the photo out).

Each engine self-reports whether it is usable on this install via :meth:`check`,
so the UI/Telegram only ever offer engines that will actually run. All three
share the same :meth:`animate` signature as the original ``FaceAnimator`` so the
job processor treats them uniformly.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from app.config import Settings


class EngineError(Exception):
    """A generic, user-safe failure raised by an engine while animating."""


class EngineUnavailable(EngineError):
    """Raised when a requested engine is not usable on this install."""


class AnimateCancelled(EngineError):
    """Raised mid-run when ``should_cancel()`` signals the job was cancelled."""


@dataclass(frozen=True)
class EngineInfo:
    """A snapshot of one engine's identity and current availability."""

    name: str
    label: str
    requires_gpu: bool
    requires_key: bool
    available: bool
    reason: str  # why it is unavailable (empty when available)
    notes: str  # short UI hint (speed / privacy / requirements)


class AnimateEngine(ABC):
    """One selectable animation backend."""

    name: str = ""
    label: str = ""
    requires_gpu: bool = False
    requires_key: bool = False
    notes: str = ""

    @abstractmethod
    def check(self, settings: "Settings") -> tuple[bool, str]:
        """Return ``(available, reason)``; ``reason`` is ``""`` when available."""

    @abstractmethod
    def animate(
        self,
        image: Image.Image,
        out_path: Path,
        workspace: Path,
        on_progress: Callable[[float], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> None:
        """Animate ``image`` into an mp4 at ``out_path`` (progress 0..1)."""

    def info(self, settings: "Settings") -> EngineInfo:
        ok, reason = self.check(settings)
        return EngineInfo(
            name=self.name,
            label=self.label,
            requires_gpu=self.requires_gpu,
            requires_key=self.requires_key,
            available=ok,
            reason="" if ok else reason,
            notes=self.notes,
        )
