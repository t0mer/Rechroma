"""TPSMM animate engine: the lightweight, always-available face reenactment.

Wraps the existing :class:`~app.core.animate.FaceAnimator` (Thin-Plate-Spline
Motion Model) behind the :class:`~app.core.engines.base.AnimateEngine` contract.
Runs on CPU or GPU; animates a single detected face.
"""

from collections.abc import Callable
from pathlib import Path

from PIL import Image

from app.config import Settings

from .base import AnimateEngine

# assets/drivers lives at the repo root: app/core/engines/tpsmm.py -> ../../../assets
_DRIVERS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "assets" / "drivers"


class TPSMMEngine(AnimateEngine):
    name = "tpsmm"
    label = "Face reenactment (TPSMM)"
    requires_gpu = False
    requires_key = False
    notes = "Animates one detected face. Runs on CPU (slow) or GPU. No downloads beyond weights."

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def check(self, settings: Settings) -> tuple[bool, str]:
        if not settings.animate_enabled:
            return False, "Animate is disabled"
        driver = _DRIVERS_DIR / f"{settings.animate_driver}.mp4"
        if not driver.exists():
            return False, f"Driver clip '{settings.animate_driver}' not found"
        return True, ""

    def animate(
        self,
        image: Image.Image,
        out_path: Path,
        workspace: Path,
        on_progress: Callable[[float], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> None:
        from app.core.animate import FaceAnimator

        animator = FaceAnimator(
            device=self.settings.device,
            models_dir=self.settings.models_dir,
            base_url=self.settings.model_base_url,
            driver_path=_DRIVERS_DIR / f"{self.settings.animate_driver}.mp4",
            max_frames=self.settings.animate_max_frames,
            crf=self.settings.animate_crf,
        )
        animator.animate(image, out_path, workspace, on_progress, should_cancel)
