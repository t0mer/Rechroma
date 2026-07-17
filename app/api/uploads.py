"""Upload validation and safe storage (CLAUDE.md §6, §10).

Guardrails: enforce a max size, verify the *real* content type by magic bytes
(not the filename), decode with Pillow under a pixel-count bomb limit, and strip
GPS EXIF from what we persist while keeping orientation.
"""

import io
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from PIL import Image, UnidentifiedImageError

if TYPE_CHECKING:
    from app.core.video import VideoCaps

# Accept only these real types (validated by magic bytes, CLAUDE.md §6).
_MAGIC: dict[str, bytes] = {
    "jpeg": b"\xff\xd8\xff",
    "png": b"\x89PNG\r\n\x1a\n",
    "gif": b"GIF8",
    "bmp": b"BM",
    "tiff_le": b"II*\x00",
    "tiff_be": b"MM\x00*",
    "webp": b"RIFF",  # plus "WEBP" at offset 8, checked below
}

# Pixel-count ceiling to defuse decompression bombs (Pillow raises above this).
MAX_PIXELS = 40_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class UploadError(Exception):
    """Raised when an upload fails validation (maps to HTTP 400/413)."""


def sniff_image_type(data: bytes) -> str | None:
    """Return a normalized image type from magic bytes, or None if unsupported."""
    if data[:3] == _MAGIC["jpeg"]:
        return "jpeg"
    if data[:8] == _MAGIC["png"]:
        return "png"
    if data[:4] == _MAGIC["gif"]:
        return "gif"
    if data[:2] == _MAGIC["bmp"]:
        return "bmp"
    if data[:4] in (_MAGIC["tiff_le"], _MAGIC["tiff_be"]):
        return "tiff"
    if data[:4] == _MAGIC["webp"] and data[8:12] == b"WEBP":
        return "webp"
    return None


def save_validated_upload(data: bytes, dest_dir: Path, job_id: str, max_bytes: int) -> Path:
    """Validate ``data`` and persist it as ``<dest_dir>/<job_id>_input.<ext>``.

    Raises :class:`UploadError` on size, type, or decode failures.
    """
    if len(data) > max_bytes:
        raise UploadError(f"image too large ({len(data)} bytes > {max_bytes})")
    kind = sniff_image_type(data)
    if kind is None:
        raise UploadError("unsupported image type (allowed: jpeg, png, webp, tiff, bmp, gif)")

    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()  # cheap integrity check
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            clean = _strip_gps(im)
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise UploadError(f"could not decode image: {e}") from e

    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{job_id}_input.png"
    clean.save(out, format="PNG")
    return out


def _strip_gps(im: Image.Image) -> Image.Image:
    """Return an RGB copy with EXIF removed but orientation already applied."""
    from PIL import ImageOps

    oriented = ImageOps.exif_transpose(im) or im  # bake orientation, then drop EXIF
    return oriented.convert("RGB")


def sniff_media_type(data: bytes) -> Literal["image", "video"] | None:
    """Classify an upload as image, video, or unsupported, by magic bytes."""
    if sniff_image_type(data) is not None:
        return "image"
    # ISO-BMFF (mp4/mov/m4v): "ftyp" box at offset 4.
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return "video"
    # Matroska / WebM (EBML header).
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return "video"
    # AVI (RIFF ... "AVI ").
    if data[:4] == b"RIFF" and data[8:12] == b"AVI ":
        return "video"
    return None


def save_validated_video(
    data: bytes, dest_dir: Path, job_id: str, max_bytes: int, caps: "VideoCaps"
) -> Path:
    """Validate a video upload (size + ffprobe caps) and persist it.

    Raises :class:`UploadError` on size, type, or cap violations.
    """
    from app.core import media
    from app.core.video import VideoCapError, check_caps

    if len(data) > max_bytes:
        raise UploadError(f"video too large ({len(data)} bytes > {max_bytes})")
    if sniff_media_type(data) != "video":
        raise UploadError("unsupported video type (allowed: mp4, mov, webm, mkv, avi)")

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{job_id}_input.mp4"
    out.write_bytes(data)

    try:
        info = media.probe(out)
    except media.MediaError as e:
        out.unlink(missing_ok=True)
        raise UploadError(f"could not read video: {e}") from e
    try:
        check_caps(info, caps)
    except VideoCapError as e:
        out.unlink(missing_ok=True)
        raise UploadError(str(e)) from e
    return out
