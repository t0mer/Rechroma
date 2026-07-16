import io

import numpy as np
import pytest
from PIL import Image

from app.api.uploads import UploadError, save_validated_upload, sniff_image_type


def _png_bytes(w=8, h=8) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((h, w, 3), 10, np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def test_sniff_detects_png_and_jpeg():
    assert sniff_image_type(_png_bytes()) == "png"
    buf = io.BytesIO()
    Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(buf, format="JPEG")
    assert sniff_image_type(buf.getvalue()) == "jpeg"


def test_sniff_rejects_non_image():
    assert sniff_image_type(b"#!/bin/sh\nrm -rf /") is None
    assert sniff_image_type(b"") is None


def test_save_validated_writes_file(tmp_path):
    path = save_validated_upload(_png_bytes(), tmp_path, "job1", max_bytes=1_000_000)
    assert path.exists()
    assert Image.open(path).size == (8, 8)


def test_save_validated_rejects_oversize(tmp_path):
    with pytest.raises(UploadError, match="too large"):
        save_validated_upload(_png_bytes(), tmp_path, "j", max_bytes=10)


def test_save_validated_rejects_non_image(tmp_path):
    with pytest.raises(UploadError, match="unsupported"):
        save_validated_upload(b"not an image at all", tmp_path, "j", max_bytes=1_000_000)


def test_save_validated_rejects_corrupt_image(tmp_path):
    # PNG magic but garbage body -> Pillow fails to decode.
    fake = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    with pytest.raises(UploadError):
        save_validated_upload(fake, tmp_path, "j", max_bytes=1_000_000)
