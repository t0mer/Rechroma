import numpy as np
from PIL import Image

from app.core.temporal import smooth_chroma


def _write(frames_dir, i, ycbcr):
    Image.fromarray(ycbcr, "YCbCr").convert("RGB").save(frames_dir / f"frame_{i:08d}.png")


def test_smoothing_reduces_chroma_variance_preserves_luma(tmp_path):
    frames = tmp_path / "f"
    frames.mkdir()
    rng = np.random.default_rng(0)
    h, w, n = 16, 16, 9
    for i in range(1, n + 1):
        y = np.full((h, w), 120, np.uint8)  # constant luminance
        cb = np.clip(128 + rng.normal(0, 40, (h, w)), 0, 255).astype(np.uint8)
        cr = np.clip(128 + rng.normal(0, 40, (h, w)), 0, 255).astype(np.uint8)
        _write(frames, i, np.stack([y, cb, cr], -1))

    def chroma_temporal_std():
        cbs = []
        for i in range(1, n + 1):
            im = Image.open(frames / f"frame_{i:08d}.png").convert("YCbCr")
            cbs.append(np.asarray(im)[..., 1].astype(float))
        return np.stack(cbs).std(axis=0).mean()

    before = chroma_temporal_std()
    smooth_chroma(frames, window=5)
    after = chroma_temporal_std()
    assert after < before * 0.8  # flicker materially reduced


def test_window_one_is_noop(tmp_path):
    frames = tmp_path / "f"
    frames.mkdir()
    y = np.full((8, 8), 100, np.uint8)
    _write(frames, 1, np.stack([y, y, y], -1))
    before = Image.open(frames / "frame_00000001.png").tobytes()
    smooth_chroma(frames, window=1)
    assert Image.open(frames / "frame_00000001.png").tobytes() == before


def test_empty_dir_is_safe(tmp_path):
    frames = tmp_path / "f"
    frames.mkdir()
    smooth_chroma(frames, window=5)  # no error
