import numpy as np
import torch
from PIL import Image

from app.core.colorizer import DeOldifyColorizer, recombine_chroma


def _gray(w: int, h: int) -> Image.Image:
    a = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    return Image.fromarray(np.stack([a, a, a], -1), "RGB")


class _StubGen:
    """Stand-in generator returning a constant reddish image in [0, 1]."""

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        b, _, h, w = t.shape
        img = torch.zeros(b, 3, h, w)
        img[:, 0] = 0.8
        img[:, 1] = 0.2
        img[:, 2] = 0.2
        return img

    def to(self, *a, **k):
        return self

    def eval(self):
        return self


def _gray_range(w: int, h: int, lo: int, hi: int) -> Image.Image:
    a = np.tile(np.linspace(lo, hi, w, dtype=np.uint8), (h, 1))
    return Image.fromarray(np.stack([a, a, a], -1), "RGB")


def test_recombine_keeps_original_luminance():
    # Mid-range luminance + a moderate tint: avoids gamut clipping, which is what
    # would otherwise perturb luminance under extreme chroma.
    orig = _gray_range(64, 48, 60, 200)  # full-res grayscale ramp
    rendered = Image.new("RGB", (32, 24), (120, 150, 190))  # low-res soft-blue tint
    out = recombine_chroma(orig, rendered)
    assert out.size == orig.size  # output at original resolution
    l_orig = np.asarray(orig.convert("L"), float)
    l_out = np.asarray(out.convert("L"), float)
    assert np.abs(l_orig - l_out).mean() < 6  # luminance preserved
    arr = np.asarray(out).astype(int)
    assert np.abs(arr[..., 2] - arr[..., 0]).mean() > 5  # chroma introduced (B != R)


def test_colorize_uses_stub_model(monkeypatch):
    c = DeOldifyColorizer(model="artistic", device="cpu")
    monkeypatch.setattr(c, "_load_model", lambda: _StubGen())
    out = c.colorize(_gray(80, 60), render_factor=10)
    assert out.size == (80, 60)
    assert out.mode == "RGB"


def test_colorize_output_has_more_color_than_gray_input(monkeypatch):
    c = DeOldifyColorizer(model="artistic", device="cpu")
    monkeypatch.setattr(c, "_load_model", lambda: _StubGen())
    src = _gray(64, 64)
    out = c.colorize(src, render_factor=10)

    def sat(im: Image.Image) -> float:
        a = np.asarray(im.convert("RGB"), float) / 255
        mx, mn = a.max(-1), a.min(-1)
        return float(((mx - mn) / (mx + 1e-6)).mean())

    assert sat(out) > sat(src)
