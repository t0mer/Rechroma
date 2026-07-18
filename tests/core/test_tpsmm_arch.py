"""Structural parity tests for the vendored Thin-Plate-Spline Motion Model.

Asserts that ``build_tpsmm()`` produces, for each of the five sub-networks, a
``state_dict`` whose keys and shapes exactly match the committed manifest
(extracted from the released ``vox.pth.tar``). This guards against silent drift
in the reimplemented architecture that would break ``strict=True`` loading.
"""

import json
from pathlib import Path

import torch

from app.core.archs.tpsmm import build_tpsmm

FIX = Path(__file__).parent / "fixtures"

_SUBNETS = (
    "kp_detector",
    "dense_motion_network",
    "inpainting_network",
    "bg_predictor",
    "avd_network",
)


def _manifest() -> dict[str, dict[str, list[int]]]:
    return json.loads((FIX / "tpsmm_keys.json").read_text())


def _subnet_state_shapes(model: object, name: str) -> dict[str, list[int]]:
    net = getattr(model, name)
    return {k: list(v.shape) for k, v in net.state_dict().items()}


def test_per_net_keys_and_shapes_match_checkpoint():
    want = _manifest()
    model = build_tpsmm()
    assert set(want) == set(_SUBNETS), "manifest sub-network set changed unexpectedly"

    for name in _SUBNETS:
        got = _subnet_state_shapes(model, name)
        expected = want[name]
        missing = sorted(set(expected) - set(got))
        extra = sorted(set(got) - set(expected))
        assert not missing, f"{name}: missing keys ({len(missing)}): {missing[:12]}"
        assert not extra, f"{name}: extra keys ({len(extra)}): {extra[:12]}"
        mismatched = {k: (expected[k], got[k]) for k in expected if expected[k] != got[k]}
        assert not mismatched, f"{name}: shape mismatches: {list(mismatched.items())[:8]}"


def test_forward_pieces_wire_up_random_init():
    # No weights: random init exercises kp detection, dense motion and the
    # inpainting generator end-to-end. Magnitudes are meaningless under random
    # init; we assert the produced frame keeps the expected shape.
    model = build_tpsmm().eval()
    source = torch.rand(1, 3, 256, 256)
    driving = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        kp_source = model.detect_keypoints(source)
        kp_driving = model.detect_keypoints(driving)
        assert kp_source["fg_kp"].shape == (1, 50, 2)
        bg_param = model.predict_bg_param(source, driving)
        assert bg_param.shape == (1, 3, 3)
        frame = model.animate(source, kp_source, kp_driving, bg_param)
    assert frame.shape == (1, 3, 256, 256)
