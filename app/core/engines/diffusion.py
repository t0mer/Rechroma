"""Diffusion animate engine: local generative image-to-video (GPU only).

Opt-in and off by default. Uses a Hugging Face image-to-video diffusion model
(default: Wan2.1-I2V) via the ``diffusers`` library to generate whole-scene
motion from a still. This is the self-hosted path toward "the whole photo comes
alive"; it requires a CUDA GPU and the optional ``diffusion`` dependencies, and
is multi-GB and minutes-per-clip. It is unavailable (and clearly reports why) on
CPU-only installs.

Note: ``diffusers`` is imported lazily and the real inference path can only be
exercised on a GPU host; the availability gating below is what keeps it from ever
being offered where it cannot run.
"""

import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger
from PIL import Image

from app.config import Settings

from .base import AnimateCancelled, AnimateEngine, EngineError


class DiffusionEngine(AnimateEngine):
    name = "diffusion"
    label = "Generative video (local GPU)"
    requires_gpu = True
    requires_key = False
    notes = "Whole-scene motion via Wan2.1-I2V. Needs a CUDA GPU + the 'diffusion' extra."

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def check(self, settings: Settings) -> tuple[bool, str]:
        if not settings.animate_diffusion_enabled:
            return False, "Diffusion engine is disabled"
        if not settings.animate_diffusion_model:
            return False, "Set animate_diffusion_model (a diffusers image-to-video repo id)"
        if importlib.util.find_spec("diffusers") is None:
            return False, "Install the 'diffusion' extra (diffusers)"
        if importlib.util.find_spec("torch") is None:
            return False, "PyTorch is not installed"
        import torch

        if not torch.cuda.is_available():
            return False, "Requires a CUDA GPU"
        return True, ""

    def animate(
        self,
        image: Image.Image,
        out_path: Path,
        workspace: Path,
        on_progress: Callable[[float], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> None:
        ok, reason = self.check(self.settings)
        if not ok:
            raise EngineError(reason)

        def report(f: float) -> None:
            if on_progress:
                on_progress(max(0.0, min(1.0, f)))

        if should_cancel and should_cancel():
            raise AnimateCancelled()

        from . import media_out  # local ffmpeg encode helper (imported lazily)

        report(0.02)
        try:
            frames = self._generate(image, report, should_cancel)
        except AnimateCancelled:
            raise
        except Exception as e:  # diffusers/model failures -> user-safe message
            logger.exception("diffusion inference failed")
            raise EngineError(f"local diffusion failed: {e}") from e

        media_out.encode_pil_frames(frames, Path(out_path), fps=self.settings.animate_diffusion_fps)
        report(1.0)

    # --- inference (GPU-only; untested on CPU CI) -------------------------
    def _generate(
        self,
        image: Image.Image,
        report: Callable[[float], None],
        should_cancel: Callable[[], bool] | None,
    ) -> list[Image.Image]:
        import torch
        from diffusers import DiffusionPipeline  # type: ignore[import-untyped]

        s = self.settings
        pipe = DiffusionPipeline.from_pretrained(
            s.animate_diffusion_model,
            torch_dtype=torch.float16,
        )
        pipe.to("cuda")

        total = max(1, int(s.animate_diffusion_steps))

        def _cb(_pipe: Any, step: int, _t: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
            if should_cancel and should_cancel():
                raise AnimateCancelled()
            report(0.05 + 0.9 * (step + 1) / total)
            return kwargs

        result = pipe(
            image=image.convert("RGB"),
            prompt=s.animate_diffusion_prompt or None,
            num_frames=s.animate_diffusion_frames,
            num_inference_steps=total,
            callback_on_step_end=_cb,
        )
        frames = result.frames[0]
        return list(frames)
