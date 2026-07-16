"""DeOldify colorization recipe (fastai-free).

The network runs on a low-resolution, ImageNet-normalized grayscale image and
predicts colour. Detail is preserved by keeping the **original-resolution
luminance** (YCbCr Y) and taking only the predicted **chroma** (Cb/Cr) from the
upsampled network output — this is the core DeOldify trick (design §3).
Post-processing (saturation/contrast) is off by default to stay faithful.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from time import perf_counter
from typing import Literal

import torch
from loguru import logger
from PIL import Image

from .archs.deoldify_unet import Backbone, build_deoldify_generator, load_state_dict_file
from .device import resolve_device
from .download import ensure_weights
from .model_registry import get_entry

ColorizerModel = Literal["artistic", "stable"]

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_BACKBONE: dict[ColorizerModel, Backbone] = {"artistic": "resnet34", "stable": "resnet101"}


def recombine_chroma(orig_rgb: Image.Image, rendered_rgb: Image.Image) -> Image.Image:
    """Combine the original image's luminance with the rendered image's chroma.

    ``rendered_rgb`` is upscaled to the original size; the result takes Y
    (luminance) from ``orig_rgb`` and Cb/Cr (chroma) from ``rendered_rgb``, then
    converts back to RGB. Using YCbCr keeps full-resolution detail intact.
    """
    orig = orig_rgb.convert("RGB")
    rendered = rendered_rgb.convert("RGB").resize(orig.size, Image.Resampling.BICUBIC)
    y_orig, _, _ = orig.convert("YCbCr").split()
    _, cb_new, cr_new = rendered.convert("YCbCr").split()
    return Image.merge("YCbCr", (y_orig, cb_new, cr_new)).convert("RGB")


class Colorizer(ABC):
    """Abstract colorizer; implementations turn a PIL image into a colorized one."""

    @abstractmethod
    def colorize(self, image: Image.Image, *, render_factor: int) -> Image.Image: ...


class DeOldifyColorizer(Colorizer):
    """DeOldify colorizer with lazy weight loading and the LAB-recombine recipe."""

    def __init__(
        self,
        model: ColorizerModel = "artistic",
        device: str = "auto",
        models_dir: Path = Path("/data/models"),
        base_url: str | None = None,
    ) -> None:
        self.model_name = model
        self.device = resolve_device(device)
        self.models_dir = models_dir
        self.base_url = base_url
        self._model: object | None = None

    def _load_model(self) -> object:
        entry = get_entry(f"deoldify-{self.model_name}")
        path = ensure_weights(entry, self.models_dir, self.base_url)
        started = perf_counter()
        net = build_deoldify_generator(_BACKBONE[self.model_name])
        net.load_state_dict(load_state_dict_file(path), strict=True)
        net.to(self.device).eval()
        logger.info(
            "loaded {} on {} in {:.1f}s",
            entry.filename,
            self.device.type,
            perf_counter() - started,
        )
        return net

    def _model_or_load(self) -> object:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _preprocess(self, image: Image.Image, render_factor: int) -> torch.Tensor:
        size = render_factor * 16
        gray = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        rgb = Image.merge("RGB", (gray, gray, gray))
        arr = torch.from_numpy(_to_float_chw(rgb)).unsqueeze(0)
        return (arr - _IMAGENET_MEAN) / _IMAGENET_STD

    def _to_pil(self, out: torch.Tensor) -> Image.Image:
        denorm = out.detach().cpu() * _IMAGENET_STD + _IMAGENET_MEAN
        chw = denorm.clamp(0, 1)[0].permute(1, 2, 0).numpy()
        return Image.fromarray((chw * 255).round().astype("uint8"), "RGB")

    def colorize(self, image: Image.Image, *, render_factor: int) -> Image.Image:
        orig = image.convert("RGB")
        model = self._model_or_load()
        x = self._preprocess(orig, render_factor).to(self.device)
        with torch.no_grad():
            out = model(x)  # type: ignore[operator]
        rendered = self._to_pil(out)
        return recombine_chroma(orig, rendered)


def _to_float_chw(rgb: Image.Image):  # type: ignore[no-untyped-def]
    import numpy as np

    arr = np.asarray(rgb, dtype="float32") / 255.0
    return arr.transpose(2, 0, 1)
