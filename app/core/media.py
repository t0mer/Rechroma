"""ffmpeg/ffprobe wrappers for video I/O (subprocess, argument lists only).

No shell is ever used; the only external string in argv is a validated,
server-controlled path. ffmpeg is a runtime dependency (installed in the Docker
images); callers may guard on availability.
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

FRAME_GLOB = "frame_*.png"
_FRAME_PATTERN = "frame_%08d.png"


class MediaError(Exception):
    """Raised when ffmpeg/ffprobe fails or returns unusable output."""


@dataclass(frozen=True)
class VideoInfo:
    duration: float
    fps: float
    width: int
    height: int
    has_audio: bool


def _run(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(args, check=True, capture_output=True)
    except FileNotFoundError as e:
        raise MediaError(f"{args[0]} not found; is ffmpeg installed?") from e
    except subprocess.CalledProcessError as e:
        raise MediaError(f"{args[0]} failed: {e.stderr.decode('utf-8', 'ignore')[:500]}") from e


def _rate_to_float(rate: str | None) -> float:
    if not rate or "/" not in rate:
        return 0.0
    num, den = rate.split("/")
    d = float(den or 0)
    return float(num) / d if d else 0.0


def probe(path: Path) -> VideoInfo:
    if not Path(path).exists():
        raise MediaError(f"file not found: {path}")
    out = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path),
        ]
    ).stdout
    data = json.loads(out or b"{}")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise MediaError("no video stream")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    fps = _rate_to_float(video.get("avg_frame_rate")) or _rate_to_float(video.get("r_frame_rate"))
    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0.0)
    return VideoInfo(duration, fps, int(video["width"]), int(video["height"]), has_audio)


def extract_audio(src: Path, dest_dir: Path) -> Path | None:
    if not probe(src).has_audio:
        return None
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    out = Path(dest_dir) / "audio.m4a"
    _run(["ffmpeg", "-y", "-i", str(src), "-vn", "-acodec", "aac", str(out)])
    return out if out.exists() else None


def extract_frames(src: Path, frames_dir: Path, fps: float) -> int:
    Path(frames_dir).mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-vf",
            f"fps={fps}",
            str(Path(frames_dir) / _FRAME_PATTERN),
        ]
    )
    return len(list(Path(frames_dir).glob(FRAME_GLOB)))


def encode_frames_with_audio(
    frames_dir: Path, audio: Path | None, out: Path, fps: float, crf: int = 18
) -> None:
    args = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(Path(frames_dir) / _FRAME_PATTERN),
    ]
    if audio is not None:
        args += ["-i", str(audio)]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-movflags", "+faststart"]
    if audio is not None:
        args += ["-c:a", "aac", "-shortest"]
    args += [str(out)]
    _run(args)
