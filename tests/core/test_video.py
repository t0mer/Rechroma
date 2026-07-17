import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core.media import VideoInfo, probe
from app.core.video import VideoCapError, VideoCaps, VideoColorizer, check_caps

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


class _TintColorizer:
    """Stand-in for DeOldifyColorizer: adds a fixed blue tint."""

    def colorize(self, image: Image.Image, *, render_factor: int) -> Image.Image:
        arr = np.asarray(image.convert("RGB")).astype(int)
        arr[..., 2] = np.clip(arr[..., 2] + 60, 0, 255)
        return Image.fromarray(arr.astype("uint8"), "RGB")


def _clip(path: Path, seconds=1, fps=10, audio=True):
    args = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size=64x48:rate={fps}:duration={seconds}",
    ]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    args += ["-pix_fmt", "yuv420p", str(path)]
    subprocess.run(args, check=True, capture_output=True)


def _caps():
    return VideoCaps(
        max_seconds=30,
        max_resolution=1080,
        max_fps=24,
        smoothing_window=3,
        render_factor=21,
        crf=18,
    )


def test_check_caps_rejects_long_and_big():
    with pytest.raises(VideoCapError, match="too long"):
        check_caps(VideoInfo(99, 24, 640, 480, True), _caps())
    with pytest.raises(VideoCapError, match="resolution"):
        check_caps(VideoInfo(5, 24, 4000, 2000, True), _caps())


def test_colorize_video_end_to_end(tmp_path):
    src = tmp_path / "in.mp4"
    _clip(src, seconds=1, fps=10, audio=True)
    out = tmp_path / "out.mp4"
    ws = tmp_path / "ws"
    vc = VideoColorizer(caps=_caps(), colorizer=_TintColorizer())
    seen: list[float] = []
    vc.colorize_video(src, out, ws, on_progress=seen.append)
    assert out.exists()
    info = probe(out)
    assert info.has_audio is True
    assert 0.8 <= info.duration <= 1.3
    assert seen and seen[0] <= seen[-1] and abs(seen[-1] - 1.0) < 1e-6


def test_colorize_video_no_audio_ok(tmp_path):
    src = tmp_path / "na.mp4"
    _clip(src, audio=False)
    out = tmp_path / "out.mp4"
    vc = VideoColorizer(caps=_caps(), colorizer=_TintColorizer())
    vc.colorize_video(src, out, tmp_path / "ws")
    assert probe(out).has_audio is False
