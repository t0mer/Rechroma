import json
from pathlib import Path

import pytest
import torch

from app.core.archs.deoldify_unet import build_deoldify_generator

FIX = Path(__file__).parent / "fixtures"


def _manifest(name: str) -> dict[str, list[int]]:
    return json.loads((FIX / name).read_text())


def test_artistic_keys_and_shapes_match():
    want = _manifest("deoldify_artistic_keys.json")
    model = build_deoldify_generator("resnet34")
    got = {k: list(v.shape) for k, v in model.state_dict().items()}
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    assert not missing, f"missing keys ({len(missing)}): {missing[:12]}"
    assert not extra, f"extra keys ({len(extra)}): {extra[:12]}"
    mismatched = {k: (want[k], got[k]) for k in want if want[k] != got[k]}
    assert not mismatched, f"shape mismatches: {list(mismatched.items())[:8]}"


@pytest.mark.parametrize("size", [160, 240])
def test_forward_preserves_shape_random_init(size):
    # No weights: random init exercises the module wiring on a non-power-of-2
    # render size (DeOldify uses render_factor*16), catching interpolation bugs.
    model = build_deoldify_generator("resnet34").eval()
    x = torch.randn(1, 3, size, size)
    with torch.no_grad():
        y = model(x)
    # Shape preservation is the wiring check; magnitudes are meaningless under
    # random init (a deep spectral-normed net can overflow before weights load).
    assert y.shape == (1, 3, size, size)


@pytest.mark.skipif(
    not (FIX / "deoldify_stable_keys.json").exists(),
    reason="stable manifest not yet committed",
)
def test_stable_keys_and_shapes_match():
    want = _manifest("deoldify_stable_keys.json")
    model = build_deoldify_generator("resnet101")
    got = {k: list(v.shape) for k, v in model.state_dict().items()}
    assert set(want) == set(got)
    assert all(want[k] == got[k] for k in want)
