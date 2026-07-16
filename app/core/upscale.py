"""Real-ESRGAN upscaling step.

Wraps the vendored RRDBNet / SRVGGNetCompact generators behind a small
``Upscaler`` with optional tiling so large images stay within memory on CPU.
Models are lazy-loaded and weights are checksum-verified downloads
(CLAUDE.md §2, §3b).
"""

from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
import torch
from loguru import logger
from PIL import Image

from .archs.checkpoint import load_checkpoint
from .archs.rrdbnet import build_rrdbnet
from .archs.srvgg import build_srvgg
from .device import resolve_device
from .download import ensure_weights
from .model_registry import get_entry

UpscaleModel = Literal["x4plus", "x2plus", "general"]

# (registry name, native scale)
_MODELS: dict[UpscaleModel, tuple[str, int]] = {
    "x4plus": ("realesrgan-x4plus", 4),
    "x2plus": ("realesrgan-x2plus", 2),
    "general": ("realesr-general-x4v3", 4),
}


def _tiled_forward(
    model: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    scale: int,
    tile: int = 0,
    pad: int = 10,
) -> torch.Tensor:
    """Run ``model`` over ``x``, optionally in overlapping tiles to bound memory."""
    if tile <= 0:
        return model(x)

    b, c, h, w = x.shape
    out = torch.zeros(b, c, h * scale, w * scale, dtype=x.dtype, device=x.device)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            iy0, ix0 = max(y0 - pad, 0), max(x0 - pad, 0)
            iy1, ix1 = min(y1 + pad, h), min(x1 + pad, w)
            tile_out = model(x[:, :, iy0:iy1, ix0:ix1])
            # crop the padded borders back off, in output resolution
            top, left = (y0 - iy0) * scale, (x0 - ix0) * scale
            out[:, :, y0 * scale : y1 * scale, x0 * scale : x1 * scale] = tile_out[
                :, :, top : top + (y1 - y0) * scale, left : left + (x1 - x0) * scale
            ]
    return out


class Upscaler:
    """Real-ESRGAN upscaler with lazy weight loading and tiling."""

    def __init__(
        self,
        model: UpscaleModel = "general",
        device: str = "auto",
        models_dir: Path = Path("/data/models"),
        base_url: str | None = None,
        tile: int = 0,
    ) -> None:
        self.model_name = model
        self.registry_name, self.native_scale = _MODELS[model]
        self.device = resolve_device(device)
        self.models_dir = models_dir
        self.base_url = base_url
        self.tile = tile
        self._model: object | None = None

    def _load_model(self) -> object:
        path = ensure_weights(get_entry(self.registry_name), self.models_dir, self.base_url)
        started = perf_counter()
        net = build_rrdbnet(self.native_scale) if self.model_name != "general" else build_srvgg()
        net.load_state_dict(load_checkpoint(path), strict=True)
        net.to(self.device).eval()
        logger.info(
            "loaded {} on {} in {:.1f}s", path.name, self.device.type, perf_counter() - started
        )
        return net

    def _model_or_load(self) -> object:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def upscale(self, image: Image.Image, outscale: float | None = None) -> Image.Image:
        """Upscale ``image`` by the model's native scale (or resize to ``outscale``)."""
        model = self._model_or_load()
        arr = np.asarray(image.convert("RGB"), dtype="float32") / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = _tiled_forward(model, x, self.native_scale, self.tile)  # type: ignore[arg-type]
        chw = out.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        result = Image.fromarray((chw * 255).round().astype("uint8"), "RGB")
        if outscale is not None and outscale != self.native_scale:
            target = (round(image.width * outscale), round(image.height * outscale))
            result = result.resize(target, Image.Resampling.LANCZOS)
        return result
