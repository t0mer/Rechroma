import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from app.core.animate import FaceAnimator, NoFaceError
from app.core.media import probe

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


class _FakeTPSMM:
    """Stand-in matching the real TPSMM inference interface (no weights)."""

    def detect_keypoints(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"fg_kp": torch.zeros(1, 50, 2)}

    def predict_bg_param(self, source: torch.Tensor, driving: torch.Tensor) -> torch.Tensor:
        return torch.eye(3).unsqueeze(0)

    def animate(self, source, kp_source, kp_driving, bg_param=None):
        out = source.clone()
        out[:, 2] = (out[:, 2] + 0.2).clamp(0, 1)  # tint blue so output != source
        return out


def _driver(path: Path, seconds: int = 1, fps: int = 8) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size=256x256:rate={fps}:duration={seconds}",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def _portrait(w: int = 320, h: int = 240) -> Image.Image:
    return Image.fromarray(np.full((h, w, 3), 128, np.uint8), "RGB")


def _animator(tmp_path, monkeypatch) -> FaceAnimator:
    drv = tmp_path / "d.mp4"
    _driver(drv)
    a = FaceAnimator(device="cpu", driver_path=drv, tpsmm=_FakeTPSMM())

    # crop the whole image as the "face" so no detector/weights are needed
    def _crop(bgr):
        crop = np.zeros((256, 256, 3), np.uint8)
        return (0, 0, bgr.shape[1], bgr.shape[0]), crop

    monkeypatch.setattr(a, "_detect_crop", _crop)
    return a


def test_animate_writes_video_at_source_size(tmp_path, monkeypatch):
    a = _animator(tmp_path, monkeypatch)
    out = tmp_path / "out.mp4"
    seen: list[float] = []
    a.animate(_portrait(), out, tmp_path / "ws", on_progress=seen.append)
    info = probe(out)
    assert out.exists()
    assert (info.width, info.height) == (320, 240)
    assert seen and abs(seen[-1] - 1.0) < 1e-6


def test_no_face_raises(tmp_path, monkeypatch):
    a = _animator(tmp_path, monkeypatch)

    def _no_face(bgr):
        raise NoFaceError()

    monkeypatch.setattr(a, "_detect_crop", _no_face)
    with pytest.raises(NoFaceError):
        a.animate(_portrait(), tmp_path / "o.mp4", tmp_path / "ws")


def test_cancel_aborts(tmp_path, monkeypatch):
    from app.core.animate import AnimateCancelled

    a = _animator(tmp_path, monkeypatch)
    with pytest.raises(AnimateCancelled):
        a.animate(_portrait(), tmp_path / "o.mp4", tmp_path / "ws", should_cancel=lambda: True)
