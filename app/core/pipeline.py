"""Composable restoration/colorization pipeline.

Steps are independent, each ``process(image) -> image``, and are assembled per
job from a preset plus options. Order (CLAUDE.md §4): face restoration runs on
the grayscale input **before** colorization; upscaling runs last.

    restore_faces (optional) -> colorize (optional) -> upscale (optional)

Presets: ``colorize`` (colour only), ``restore`` (restore an already-colour
photo, no colorize), ``full`` (both). This module owns only orchestration and
step *selection*; the heavy model wrappers live in ``colorizer``/``upscale``/
``restore`` and are imported lazily so building a plan never loads a model.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from PIL import Image

Preset = Literal["colorize", "restore", "full"]
ColorizerModel = Literal["artistic", "stable"]
UpscaleFactor = Literal[2, 4]


@runtime_checkable
class Step(Protocol):
    name: str

    def process(self, image: Image.Image) -> Image.Image: ...


@dataclass(frozen=True)
class PipelineOptions:
    """User-selectable job options (CLAUDE.md §4 "Presets" + "Advanced options")."""

    preset: Preset = "full"
    colorizer_model: ColorizerModel = "artistic"
    render_factor: int | None = None  # None -> device default
    upscale: UpscaleFactor | None = None
    restore_faces: bool = True


def _plan(options: PipelineOptions) -> list[str]:
    """Return the ordered step names a preset+options resolves to."""
    steps: list[str] = []
    wants_restore = options.preset in ("restore", "full") and options.restore_faces
    wants_colorize = options.preset in ("colorize", "full")
    if wants_restore:
        steps.append("restore_faces")
    if wants_colorize:
        steps.append("colorize")
    if options.upscale is not None:
        steps.append("upscale")
    return steps


def step_names(options: PipelineOptions) -> list[str]:
    """Public helper: the ordered step names for a plan (no models loaded)."""
    return _plan(options)


def build_steps(
    options: PipelineOptions,
    device: str = "auto",
    models_dir: Path = Path("/data/models"),
    base_url: str | None = None,
) -> list[Step]:
    """Instantiate the concrete steps for ``options`` (lazy model wrappers)."""
    from .colorizer import ColorizeStep
    from .upscale import UpscaleStep

    steps: list[Step] = []
    for name in _plan(options):
        if name == "restore_faces":
            from .restore import FaceRestoreStep

            steps.append(FaceRestoreStep(device=device, models_dir=models_dir, base_url=base_url))
        elif name == "colorize":
            steps.append(
                ColorizeStep(
                    model=options.colorizer_model,
                    render_factor=options.render_factor,
                    device=device,
                    models_dir=models_dir,
                    base_url=base_url,
                )
            )
        elif name == "upscale":
            steps.append(
                UpscaleStep(
                    factor=options.upscale or 2,
                    device=device,
                    models_dir=models_dir,
                    base_url=base_url,
                )
            )
    return steps


def run_pipeline(steps: list[Step], image: Image.Image) -> Image.Image:
    """Apply each step in order, threading the image through."""
    result = image
    for step in steps:
        result = step.process(result)
    return result
