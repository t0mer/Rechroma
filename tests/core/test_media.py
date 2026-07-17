import shutil
import subprocess
from pathlib import Path

import pytest

from app.core.media import (
    MediaError,
    encode_frames_with_audio,
    extract_audio,
    extract_frames,
    probe,
)

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _make_clip(path: Path, seconds: int = 1, fps: int = 10, with_audio: bool = True) -> None:
    args = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size=64x48:rate={fps}:duration={seconds}",
    ]
    if with_audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    args += ["-pix_fmt", "yuv420p", str(path)]
    subprocess.run(args, check=True, capture_output=True)


def _silent(tmp_path: Path) -> Path:
    clip = tmp_path / "silent.mp4"
    _make_clip(clip, with_audio=False)
    return clip


def test_probe_reports_fields(tmp_path):
    clip = tmp_path / "in.mp4"
    _make_clip(clip, seconds=1, fps=10, with_audio=True)
    info = probe(clip)
    assert 0.8 <= info.duration <= 1.3
    assert 9 <= info.fps <= 11
    assert (info.width, info.height) == (64, 48)
    assert info.has_audio is True


def test_probe_no_audio(tmp_path):
    clip = tmp_path / "na.mp4"
    _make_clip(clip, with_audio=False)
    assert probe(clip).has_audio is False


def test_extract_frames_counts(tmp_path):
    clip = tmp_path / "in.mp4"
    _make_clip(clip, seconds=1, fps=10, with_audio=False)
    frames = tmp_path / "frames"
    n = extract_frames(clip, frames, fps=10)
    written = sorted(frames.glob("frame_*.png"))
    assert n == len(written) >= 9
    assert written[0].name == "frame_00000001.png"


def test_extract_audio_roundtrip(tmp_path):
    clip = tmp_path / "in.mp4"
    _make_clip(clip, with_audio=True)
    audio = extract_audio(clip, tmp_path)
    assert audio is not None and audio.exists()
    assert extract_audio(_silent(tmp_path), tmp_path / "sub") is None


def test_encode_roundtrip_preserves_duration_and_audio(tmp_path):
    clip = tmp_path / "in.mp4"
    _make_clip(clip, seconds=1, fps=10, with_audio=True)
    frames = tmp_path / "frames"
    extract_frames(clip, frames, fps=10)
    audio = extract_audio(clip, tmp_path)
    out = tmp_path / "out.mp4"
    encode_frames_with_audio(frames, audio, out, fps=10)
    info = probe(out)
    assert info.has_audio is True
    assert 0.8 <= info.duration <= 1.3


def test_probe_missing_file_raises(tmp_path):
    with pytest.raises(MediaError):
        probe(tmp_path / "nope.mp4")
