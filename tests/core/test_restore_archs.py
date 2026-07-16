"""Strict-load + forward smoke tests for the vendored restoration architectures.

These verify that the reconstructed GFPGAN (clean), RetinaFace and ParseNet
architectures load the *original* released weights with ``strict=True`` (no
missing / unexpected keys) and produce finite output of the expected shape.

Weights are large and never committed. The tests are skipped unless
``RECHROMA_TEST_WEIGHTS`` points at a directory containing the real ``.pth``
files, so CI without weights still passes.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from app.core.archs.gfpgan_clean import build_gfpgan_clean
from app.core.archs.parsenet import build_parsenet
from app.core.archs.retinaface import build_retinaface, strip_module_prefix

_WEIGHTS_ENV = "RECHROMA_TEST_WEIGHTS"


def _weights_dir() -> Path | None:
    raw = os.environ.get(_WEIGHTS_ENV)
    return Path(raw) if raw else None


def _weight_path(filename: str) -> Path | None:
    directory = _weights_dir()
    if directory is None:
        return None
    candidate = directory / filename
    return candidate if candidate.is_file() else None


def _require(filename: str) -> Path:
    path = _weight_path(filename)
    if path is None:
        pytest.skip(f"{_WEIGHTS_ENV} unset or {filename} missing; skipping weight test")
    return path


def _load(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "params_ema" in ckpt:
        return ckpt["params_ema"]
    return ckpt


def test_gfpgan_clean_strict_load_and_forward() -> None:
    path = _require("GFPGANv1.4.pth")
    model = build_gfpgan_clean().eval()
    result = model.load_state_dict(_load(path), strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    with torch.no_grad():
        out, _ = model(torch.randn(1, 3, 512, 512), randomize_noise=False)
    assert out.shape == (1, 3, 512, 512)
    assert torch.isfinite(out).all()


def test_retinaface_strict_load_and_forward() -> None:
    path = _require("detection_Resnet50_Final.pth")
    model = build_retinaface().eval()
    state = strip_module_prefix(_load(path))
    result = model.load_state_dict(state, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    with torch.no_grad():
        loc, conf, landms = model(torch.randn(1, 3, 480, 640))
    assert loc.shape[-1] == 4
    assert conf.shape[-1] == 2
    assert landms.shape[-1] == 10
    assert loc.shape[1] == conf.shape[1] == landms.shape[1]
    assert torch.isfinite(loc).all()
    assert torch.isfinite(conf).all()
    assert torch.isfinite(landms).all()


def test_parsenet_strict_load_and_forward() -> None:
    path = _require("parsing_parsenet.pth")
    model = build_parsenet().eval()
    result = model.load_state_dict(_load(path), strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []
    with torch.no_grad():
        mask, img = model(torch.randn(1, 3, 512, 512))
    assert mask.shape == (1, 19, 512, 512)
    assert img.shape == (1, 3, 512, 512)
    assert torch.isfinite(mask).all()
    assert torch.isfinite(img).all()
