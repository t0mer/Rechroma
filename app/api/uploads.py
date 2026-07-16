"""Upload validation and safe storage (CLAUDE.md §6, §10).

Guardrails: enforce a max size, verify the *real* content type by magic bytes
(not the filename), decode with Pillow under a pixel-count bomb limit, and strip
GPS EXIF from what we persist while keeping orientation.
"""

import io
from pathlib import Path

from PIL import Image, UnidentifiedImageError

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
