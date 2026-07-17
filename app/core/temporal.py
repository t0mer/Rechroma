"""Temporal chroma smoothing to reduce per-frame colorization flicker.

Works in YCbCr: luminance (Y) is preserved bit-for-bit; only Cb/Cr are averaged
over a centered window of neighbouring frames. Memory-bounded — a cache holds at
most ``window`` decoded frames, and it holds the *pre-smoothing* chroma so later
frames never average against already-smoothed output.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from .media import FRAME_GLOB


def smooth_chroma(frames_dir: Path, window: int) -> None:
    if window <= 1:
        return
    paths = sorted(Path(frames_dir).glob(FRAME_GLOB))
    n = len(paths)
    if n == 0:
        return
    half = window // 2
    cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def load(idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if idx not in cache:
            with Image.open(paths[idx]) as im:
                y, cb, cr = (np.asarray(c, dtype=np.float32) for c in im.convert("YCbCr").split())
            cache[idx] = (y, cb, cr)
        return cache[idx]

    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        cbs = np.stack([load(j)[1] for j in range(lo, hi)])
        crs = np.stack([load(j)[2] for j in range(lo, hi)])
        y = load(i)[0]
        merged = np.stack([y, cbs.mean(axis=0), crs.mean(axis=0)], axis=-1)
        out = merged.round().clip(0, 255).astype(np.uint8)
        Image.fromarray(out, "YCbCr").convert("RGB").save(paths[i])
        # Drop frames that will never fall inside a later window again.
        for j in [k for k in cache if k < i - half]:
            del cache[j]
