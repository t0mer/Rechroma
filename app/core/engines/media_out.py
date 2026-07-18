"""Encode a list of in-memory PIL frames to an mp4 via the shared ffmpeg layer.

Used by the diffusion engine, whose model returns frames as PIL images rather
than files on disk. Frames are written to a temporary directory in the
``frame_%08d.png`` layout the media layer expects, then encoded.
"""

import tempfile
from pathlib import Path

from PIL import Image

from app.core import media


def encode_pil_frames(frames: list[Image.Image], out_path: Path, fps: float, crf: int = 18) -> None:
    if not frames:
        raise ValueError("no frames to encode")
    with tempfile.TemporaryDirectory() as td:
        frames_dir = Path(td)
        for i, frame in enumerate(frames, start=1):
            frame.convert("RGB").save(frames_dir / f"frame_{i:08d}.png")
        media.encode_frames_with_audio(frames_dir, None, Path(out_path), fps=fps, crf=crf)
