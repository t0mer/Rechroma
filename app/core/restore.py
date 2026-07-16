"""GFPGAN face restoration step.

Pipeline: detect faces (RetinaFace) -> align each to a 512x512 crop by a 5-point
similarity transform -> restore (GFPGANv1Clean) -> build a soft mask from the
face parse (ParseNet) -> warp the restored face back and alpha-blend it into the
image. Faces that aren't found leave the image untouched. All three model
weights are checksum-verified lazy downloads (CLAUDE.md §2, §3b).

Attribution: algorithm follows GFPGAN / facexlib ``FaceRestoreHelper``
(TencentARC, xinntao) — Apache-2.0 / MIT.
"""

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image

from .archs.checkpoint import load_checkpoint
from .archs.gfpgan_clean import build_gfpgan_clean
from .archs.parsenet import build_parsenet
from .archs.retinaface import (
    build_retinaface,
    decode_boxes,
    decode_landms,
    nms,
    prior_box,
    strip_module_prefix,
)
from .device import resolve_device
from .download import ensure_weights
from .model_registry import get_entry

# 5-point reference template (left eye, right eye, nose, left mouth, right mouth)
# for a 512x512 aligned face, from facexlib FaceRestoreHelper.
_FACE_TEMPLATE = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.1936],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float32,
)
_FACE_SIZE = 512
_RETINA_MEAN = np.array([104.0, 117.0, 123.0], dtype=np.float32)  # BGR
_VARIANCES = [0.1, 0.2]


class FaceRestorer:
    """Restore faces in an image using GFPGAN with RetinaFace + ParseNet helpers."""

    def __init__(
        self,
        device: str = "auto",
        models_dir: Path = Path("/data/models"),
        base_url: str | None = None,
        conf_threshold: float = 0.9,
        nms_threshold: float = 0.4,
        max_side: int = 1280,
    ) -> None:
        self.device = resolve_device(device)
        self.models_dir = models_dir
        self.base_url = base_url
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.max_side = max_side
        self._det: object | None = None
        self._gfp: object | None = None
        self._parse: object | None = None

    def _weights(self, name: str) -> Path:
        return ensure_weights(get_entry(name), self.models_dir, self.base_url)

    def _ensure_models(self) -> None:
        if self._det is not None:
            return
        started = perf_counter()
        det = build_retinaface()
        det.load_state_dict(
            strip_module_prefix(load_checkpoint(self._weights("gfpgan-detection"))), strict=True
        )
        gfp = build_gfpgan_clean()
        gfp.load_state_dict(load_checkpoint(self._weights("gfpgan")), strict=True)
        parse = build_parsenet()
        parse.load_state_dict(load_checkpoint(self._weights("gfpgan-parsing")), strict=True)
        for m in (det, gfp, parse):
            m.to(self.device).eval()  # type: ignore[attr-defined]
        self._det, self._gfp, self._parse = det, gfp, parse
        logger.info(
            "loaded GFPGAN face models on {} in {:.1f}s",
            self.device.type,
            perf_counter() - started,
        )

    def detect(self, bgr: np.ndarray) -> list[np.ndarray]:
        """Return a list of 5x2 landmark arrays for detected faces (image coords)."""
        assert self._det is not None
        h, w = bgr.shape[:2]
        img = bgr.astype(np.float32) - _RETINA_MEAN
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            loc, conf, landms = self._det(tensor)  # type: ignore[operator]
        priors = prior_box((h, w)).to(self.device)
        boxes_t = decode_boxes(loc.squeeze(0), priors, _VARIANCES)
        lms_t = decode_landms(landms.squeeze(0), priors, _VARIANCES)
        scale_b = torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device)
        scale_l = torch.tensor([w, h] * 5, dtype=torch.float32, device=self.device)
        boxes = (boxes_t * scale_b).cpu().numpy()
        lms = (lms_t * scale_l).cpu().numpy()
        scores = conf.squeeze(0)[:, 1].cpu().numpy()

        keep_mask = scores > self.conf_threshold
        boxes, lms, scores = boxes[keep_mask], lms[keep_mask], scores[keep_mask]
        if len(boxes) == 0:
            return []
        keep = nms(boxes, scores, self.nms_threshold)
        return [lms[i].reshape(5, 2).astype(np.float32) for i in keep]

    def _restore_crop(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Run GFPGAN on a 512x512 BGR crop, return restored 512x512 BGR uint8."""
        assert self._gfp is not None
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        t = (t - 0.5) / 0.5  # -> [-1, 1]
        with torch.no_grad():
            out = self._gfp(t)  # type: ignore[operator]
            out_t = out[0] if isinstance(out, tuple) else out
        out_t = (out_t.squeeze(0).clamp(-1, 1) + 1) / 2
        rgb_out = out_t.permute(1, 2, 0).cpu().numpy()
        return cv2.cvtColor((rgb_out * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)

    def _face_mask(self, restored_bgr: np.ndarray) -> np.ndarray:
        """Soft 512x512 alpha mask over facial regions from ParseNet."""
        assert self._parse is not None
        rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        t = (t - 0.5) / 0.5
        with torch.no_grad():
            out = self._parse(t)  # type: ignore[operator]
            seg = out[0] if isinstance(out, tuple) else out
        labels = seg.squeeze(0).argmax(0).cpu().numpy()
        # classes 0=bg, 14=neck, 16=cloth, 17=hair, 18=hat -> exclude from face mask
        mask = np.isin(labels, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]).astype(np.float32)
        eroded = cv2.erode(mask, np.ones((7, 7), np.uint8))
        blurred = cv2.GaussianBlur(eroded, (0, 0), sigmaX=9)
        return np.clip(blurred, 0, 1).astype(np.float32)

    def restore(self, image: Image.Image) -> Image.Image:
        """Restore all detected faces; return a new image (original if no faces)."""
        self._ensure_models()
        rgb = np.asarray(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        det_bgr, scale = _resize_max(bgr, self.max_side)
        faces = self.detect(det_bgr)
        if not faces:
            return image
        result = bgr.astype(np.float32)
        for landmarks in faces:
            lm = landmarks / scale  # back to full-res coords
            affine, _ = cv2.estimateAffinePartial2D(lm, _FACE_TEMPLATE, method=cv2.LMEDS)
            if affine is None:
                continue
            crop = cv2.warpAffine(bgr, affine, (_FACE_SIZE, _FACE_SIZE), flags=cv2.INTER_LINEAR)
            restored = self._restore_crop(crop)
            mask = self._face_mask(restored)
            inv = cv2.invertAffineTransform(affine)
            back = cv2.warpAffine(restored, inv, (w, h), flags=cv2.INTER_LINEAR).astype(np.float32)
            back_mask = cv2.warpAffine(mask, inv, (w, h), flags=cv2.INTER_LINEAR)[..., None]
            result = (back * back_mask + result * (1 - back_mask)).astype(np.float32)
        out_bgr = np.clip(result, 0, 255).astype(np.uint8)
        return Image.fromarray(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))


def _resize_max(bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Downscale so the longest side <= max_side; return (image, scale_applied)."""
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return bgr, 1.0
    scale = max_side / longest
    resized = cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


class FaceRestoreStep:
    """Pipeline step wrapping :class:`FaceRestorer` (CLAUDE.md §4)."""

    name = "restore_faces"

    def __init__(
        self,
        device: str = "auto",
        models_dir: Path = Path("/data/models"),
        base_url: str | None = None,
    ) -> None:
        self._restorer = FaceRestorer(device=device, models_dir=models_dir, base_url=base_url)

    def process(self, image: Image.Image) -> Image.Image:
        return self._restorer.restore(image)
