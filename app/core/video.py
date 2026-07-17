"""Video colorization: frames -> per-frame colorize -> temporal smooth -> encode.

Reuses the existing DeOldifyColorizer per frame (loaded once). The colorizer is
injectable for tests. Caps are re-checked defensively before processing.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from loguru import logger
from PIL import Image

from . import media, temporal
from .colorizer import ColorizerModel, DeOldifyColorizer
from .media import FRAME_GLOB


class _FrameColorizer(Protocol):
    def colorize(self, image: Image.Image, *, render_factor: int) -> Image.Image: ...


@dataclass
class VideoCaps:
    max_seconds: int
    max_resolution: int
    max_fps: int
    smoothing_window: int
    render_factor: int
    crf: int


class VideoCapError(Exception):
    """Raised when a video exceeds configured caps."""


class VideoCancelled(Exception):
    """Raised mid-run when ``should_cancel()`` signals the job was cancelled."""


def check_caps(info: media.VideoInfo, caps: VideoCaps) -> None:
    if info.duration > caps.max_seconds:
        raise VideoCapError(f"video too long: {info.duration:.1f}s > {caps.max_seconds}s")
    if max(info.width, info.height) > caps.max_resolution:
        raise VideoCapError(
            f"resolution too high: {info.width}x{info.height} > {caps.max_resolution}"
        )


class VideoColorizer:
    def __init__(
        self,
        model: ColorizerModel = "artistic",
        device: str = "auto",
        models_dir: Path = Path("/data/models"),
        base_url: str | None = None,
        caps: VideoCaps | None = None,
        colorizer: _FrameColorizer | None = None,
    ) -> None:
        self.caps = caps
        self._colorizer: _FrameColorizer = colorizer or DeOldifyColorizer(
            model, device, models_dir, base_url
        )

    def colorize_video(
        self,
        in_path: Path,
        out_path: Path,
        workspace: Path,
        on_progress: Callable[[float], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> None:
        def report(f: float) -> None:
            if on_progress:
                on_progress(max(0.0, min(1.0, f)))

        info = media.probe(in_path)
        if self.caps:
            check_caps(info, self.caps)
        fps = min(info.fps, self.caps.max_fps) if self.caps else info.fps
        frames_dir = Path(workspace) / "frames"
        Path(workspace).mkdir(parents=True, exist_ok=True)
        report(0.02)
        audio = media.extract_audio(in_path, workspace)
        report(0.05)
        n = media.extract_frames(in_path, frames_dir, fps=fps)
        logger.info("video: {} frames at {:.2f} fps", n, fps)
        rf = self.caps.render_factor if self.caps else 21
        for i, frame_path in enumerate(sorted(frames_dir.glob(FRAME_GLOB))):
            if should_cancel and should_cancel():
                raise VideoCancelled()
            with Image.open(frame_path) as im:
                colored = self._colorizer.colorize(im.convert("RGB"), render_factor=rf)
            colored.save(frame_path)
            report(0.05 + 0.80 * (i + 1) / max(n, 1))
        if self.caps:
            temporal.smooth_chroma(frames_dir, self.caps.smoothing_window)
        report(0.92)
        media.encode_frames_with_audio(
            frames_dir,
            audio,
            Path(out_path),
            fps=fps,
            crf=self.caps.crf if self.caps else 18,
        )
        report(1.0)
