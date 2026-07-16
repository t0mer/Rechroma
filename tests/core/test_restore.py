import numpy as np
from PIL import Image

from app.core.pipeline import PipelineOptions, build_steps
from app.core.restore import FaceRestorer, FaceRestoreStep, _resize_max


def _img(w: int, h: int) -> Image.Image:
    return Image.fromarray(np.full((h, w, 3), 120, np.uint8), "RGB")


def test_resize_max_downscales_and_reports_scale():
    bgr = np.zeros((1000, 2000, 3), np.uint8)
    out, scale = _resize_max(bgr, 1280)
    assert max(out.shape[:2]) == 1280
    assert abs(scale - 0.64) < 1e-6


def test_resize_max_noop_when_small():
    bgr = np.zeros((100, 200, 3), np.uint8)
    out, scale = _resize_max(bgr, 1280)
    assert out.shape == bgr.shape
    assert scale == 1.0


def test_restore_returns_original_when_no_faces(monkeypatch):
    r = FaceRestorer(device="cpu")
    monkeypatch.setattr(r, "_ensure_models", lambda: None)
    monkeypatch.setattr(r, "detect", lambda bgr: [])
    src = _img(64, 48)
    out = r.restore(src)
    assert out is src  # untouched


def test_full_preset_wires_restore_colorize_upscale():
    steps = build_steps(PipelineOptions(preset="full", upscale=2), device="cpu")
    assert [s.name for s in steps] == ["restore_faces", "colorize", "upscale"]
    assert isinstance(steps[0], FaceRestoreStep)
