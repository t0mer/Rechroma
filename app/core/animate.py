"""Photo animation ("living portrait") via the Thin-Plate-Spline Motion Model.

A standalone feature: a still portrait -> a short mp4 where the face comes to
life. Motion is transferred (relative mode) from a bundled driving clip onto the
uploaded face. Reuses the vendored facexlib RetinaFace detector for the face
crop, the vendored TPSMM for animation, and the ffmpeg media layer for encoding.

The output is the animated head-and-shoulders region itself (upscaled toward the
source resolution) -- the still is *not* re-composited. Compositing a moving face
back into a static body produced a visible crop seam and a "sliding mask" look;
outputting the animated crop directly is both cleaner and closer to the
living-portrait demos this feature targets.

Attribution: TPSMM (Zhao & Zhang, CVPR 2022), MIT code; VoxCeleb weights are
CC BY-SA 4.0. Runs on CPU (slow) or GPU.
"""

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image

from . import media
from .archs.checkpoint import load_checkpoint
from .archs.retinaface import (
    build_retinaface,
    decode_boxes,
    nms,
    prior_box,
    strip_module_prefix,
)
from .archs.tpsmm import TPSMM, load_tpsmm
from .device import resolve_device
from .download import ensure_weights
from .engines.base import AnimateCancelled  # canonical; re-exported for compatibility
from .model_registry import get_entry

_RETINA_MEAN = np.array([104.0, 117.0, 123.0], dtype=np.float32)  # BGR
_VARIANCES = [0.1, 0.2]
_TPS = 256  # TPSMM operates on 256x256 crops
_MAX_OUT = 720  # cap on the upscaled output side (px) to keep files reasonable
_CROP_MARGIN = 0.6  # head-and-shoulders framing around the detected face box
_OUT_KEEP = 0.9  # keep this centre fraction of each animated frame (trims warp rim)


@dataclass
class AnimateOptions:
    driver: str = "subtle"
    max_frames: int = 120
    crf: int = 18


class NoFaceError(Exception):
    """Raised when no face is detected in the source image."""


__all__ = ["AnimateCancelled", "FaceAnimator", "NoFaceError"]


class _TPSMMProto(Protocol):  # minimal structural type for the injected/real model
    def detect_keypoints(self, image: torch.Tensor) -> dict[str, torch.Tensor]: ...
    def predict_bg_param(self, source: torch.Tensor, driving: torch.Tensor) -> torch.Tensor: ...
    def animate(self, source, kp_source, kp_driving, bg_param=None): ...  # type: ignore[no-untyped-def]


class FaceAnimator:
    """Animate a portrait with TPSMM relative motion from a driving clip."""

    def __init__(
        self,
        device: str = "auto",
        models_dir: Path = Path("/data/models"),
        base_url: str | None = None,
        driver_path: Path | None = None,
        max_frames: int = 120,
        crf: int = 18,
        conf_threshold: float = 0.9,
        nms_threshold: float = 0.4,
        tpsmm: _TPSMMProto | TPSMM | None = None,
    ) -> None:
        self.device = resolve_device(device)
        self.models_dir = models_dir
        self.base_url = base_url
        self.driver_path = driver_path
        self.max_frames = max_frames
        self.crf = crf
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self._tpsmm = tpsmm
        self._det: object | None = None

    # --- model loading -----------------------------------------------------
    def _model(self) -> _TPSMMProto | TPSMM:
        if self._tpsmm is None:
            started = perf_counter()
            path = ensure_weights(get_entry("tpsmm"), self.models_dir, self.base_url)
            model = load_tpsmm(path)
            model.to(self.device).eval()  # type: ignore[attr-defined]
            logger.info("loaded TPSMM on {} in {:.1f}s", self.device.type, perf_counter() - started)
            self._tpsmm = model
        return self._tpsmm

    def _detector(self) -> object:
        if self._det is None:
            det = build_retinaface()
            det.load_state_dict(
                strip_module_prefix(load_checkpoint(self._weights("gfpgan-detection"))), strict=True
            )
            det.to(self.device).eval()  # type: ignore[attr-defined]
            self._det = det
        return self._det

    def _weights(self, name: str) -> Path:
        return ensure_weights(get_entry(name), self.models_dir, self.base_url)

    # --- face crop ---------------------------------------------------------
    def _detect_crop(self, bgr: np.ndarray) -> tuple[np.ndarray, int]:
        """Return a 256x256 RGB face crop and the px side to upscale output to.

        The crop is a face-centred square padded (edge-replicated, never black)
        when it runs past the image bounds -- so TPSMM never sees black borders
        to warp into artifacts.
        """
        det = self._detector()
        h, w = bgr.shape[:2]
        img = bgr.astype(np.float32) - _RETINA_MEAN
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            loc, conf, _ = det(tensor)  # type: ignore[operator]
        priors = prior_box((h, w)).to(self.device)
        boxes = (
            (
                decode_boxes(loc.squeeze(0), priors, _VARIANCES)
                * torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device)
            )
            .cpu()
            .numpy()
        )
        scores = conf.squeeze(0)[:, 1].cpu().numpy()
        keep_mask = scores > self.conf_threshold
        boxes, scores = boxes[keep_mask], scores[keep_mask]
        if len(boxes) == 0:
            raise NoFaceError("no face detected in the image")
        keep = nms(boxes, scores, self.nms_threshold)
        boxes = boxes[keep]
        # largest face
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        box = boxes[int(np.argmax(areas))]
        return _square_crop_padded(bgr, box, margin=_CROP_MARGIN, max_side=_MAX_OUT)

    # --- animation ---------------------------------------------------------
    def animate(
        self,
        image: Image.Image,
        out_path: Path,
        workspace: Path,
        on_progress: Callable[[float], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> None:
        def report(f: float) -> None:
            if on_progress:
                on_progress(max(0.0, min(1.0, f)))

        if self.driver_path is None:
            raise ValueError("no driver clip configured")
        rgb = np.asarray(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        report(0.02)
        crop, out_side = self._detect_crop(bgr)

        Path(workspace).mkdir(parents=True, exist_ok=True)
        driver_dir = Path(workspace) / "driver"
        frames_dir = Path(workspace) / "frames"
        shutil.rmtree(frames_dir, ignore_errors=True)  # start from a clean frame set
        info = media.probe(self.driver_path)
        media.extract_frames(self.driver_path, driver_dir, fps=info.fps)
        driver_files = sorted(driver_dir.glob(media.FRAME_GLOB))[: self.max_frames]
        if not driver_files:
            raise ValueError("driver clip produced no frames")
        report(0.05)

        model = self._model()
        source_t = _to_tensor(crop, self.device)
        kp_source = model.detect_keypoints(source_t)
        first = _to_tensor(_read256(driver_files[0]), self.device)
        kp_d0 = model.detect_keypoints(first)

        frames_dir.mkdir(parents=True, exist_ok=True)
        n = len(driver_files)
        for i, dfile in enumerate(driver_files):
            if should_cancel and should_cancel():
                raise AnimateCancelled()
            drv = _to_tensor(_read256(dfile), self.device)
            kp_d = model.detect_keypoints(drv)
            kp_rel = {
                "fg_kp": kp_source["fg_kp"] + (kp_d["fg_kp"] - kp_d0["fg_kp"]),
            }
            bg_param = model.predict_bg_param(source_t, drv)
            out_t = model.animate(source_t, kp_source, kp_rel, bg_param)
            frame = _from_tensor(out_t)  # 256x256 RGB
            frame = _center_keep(frame, _OUT_KEEP)  # drop the warped peripheral rim
            if frame.shape[0] != out_side:
                frame = cv2.resize(frame, (out_side, out_side), interpolation=cv2.INTER_LANCZOS4)
            Image.fromarray(frame, "RGB").save(frames_dir / f"frame_{i + 1:08d}.png")
            report(0.05 + 0.85 * (i + 1) / n)

        media.encode_frames_with_audio(frames_dir, None, Path(out_path), fps=info.fps, crf=self.crf)
        report(1.0)


# --- helpers ---------------------------------------------------------------
def _square_crop_padded(
    bgr: np.ndarray, box: np.ndarray, margin: float, max_side: int
) -> tuple[np.ndarray, int]:
    """Face-centred square crop, edge-padded past image bounds; 256 RGB + out side.

    Padding replicates the border rather than filling black, so TPSMM never warps
    a black margin into the animated frames. ``out_side`` is the square pixel size
    to upscale the animated 256 output back to (capped at ``max_side``).
    """
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = max(x2 - x1, y2 - y1) * (1.0 + margin) / 2.0
    side = max(1, int(round(half * 2)))
    left, top = int(round(cx - half)), int(round(cy - half))
    pad_l, pad_t = max(0, -left), max(0, -top)
    pad_r, pad_b = max(0, left + side - w), max(0, top + side - h)
    if pad_l or pad_t or pad_r or pad_b:
        bgr = cv2.copyMakeBorder(bgr, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
    x0, y0 = left + pad_l, top + pad_t
    crop = bgr[y0 : y0 + side, x0 : x0 + side]
    crop256 = cv2.resize(crop, (_TPS, _TPS), interpolation=cv2.INTER_AREA)
    out_side = max(_TPS, min(side, max_side))
    return cv2.cvtColor(crop256, cv2.COLOR_BGR2RGB), out_side


def _center_keep(img: np.ndarray, frac: float) -> np.ndarray:
    """Keep the centre ``frac`` of a square frame; TPSMM warp piles up at the rim."""
    h, w = img.shape[:2]
    m = int(round(min(h, w) * (1.0 - frac) / 2.0))
    return img[m : h - m, m : w - m] if m > 0 else img


def _to_tensor(rgb256: np.ndarray, device: torch.device) -> torch.Tensor:
    arr = rgb256.astype("float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def _from_tensor(t: torch.Tensor) -> np.ndarray:
    chw = t.detach().cpu()[0].clamp(0, 1).permute(1, 2, 0).numpy()
    return (chw * 255).round().astype("uint8")


def _read256(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
    if rgb.shape[:2] != (_TPS, _TPS):
        rgb = cv2.resize(rgb, (_TPS, _TPS), interpolation=cv2.INTER_AREA)
    return rgb


class AnimateStep:
    """Thin wrapper exposing a ``name`` for symmetry with pipeline steps."""

    name = "animate"
