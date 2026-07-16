import numpy as np
import torch
from PIL import Image

from app.core.upscale import Upscaler, _tiled_forward


def _img(w: int, h: int) -> Image.Image:
    rng = np.random.default_rng(1)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), "RGB")


class _Stub2x:
    """Nearest-neighbour 2x upscaler stand-in (no weights)."""

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(t, scale_factor=2, mode="nearest")


def test_tiled_forward_matches_whole_for_scale2():
    x = torch.rand(1, 3, 40, 48)
    whole = _tiled_forward(_Stub2x(), x, scale=2, tile=0)
    tiled = _tiled_forward(_Stub2x(), x, scale=2, tile=16, pad=4)
    assert whole.shape == tiled.shape == (1, 3, 80, 96)
    # nearest upscaling is exactly reconstructable from tiles
    assert torch.allclose(whole, tiled, atol=1e-5)


def test_upscaler_uses_stub_and_scales(monkeypatch):
    up = Upscaler(model="x2plus", device="cpu")
    monkeypatch.setattr(up, "_load_model", lambda: _Stub2x())
    monkeypatch.setattr(up, "native_scale", 2)
    out = up.upscale(_img(30, 20))
    assert out.size == (60, 40)
    assert out.mode == "RGB"


def test_upscaler_outscale_resizes(monkeypatch):
    up = Upscaler(model="x2plus", device="cpu")
    monkeypatch.setattr(up, "_load_model", lambda: _Stub2x())
    monkeypatch.setattr(up, "native_scale", 2)
    out = up.upscale(_img(30, 20), outscale=1.5)
    assert out.size == (45, 30)
