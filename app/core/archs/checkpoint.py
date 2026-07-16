"""Safe checkpoint loading shared across all vendored architectures.

Always loads with ``weights_only=True`` so no arbitrary pickle code can run
(CLAUDE.md §10). Different upstream projects wrap the tensors under different
keys (``params_ema`` for Real-ESRGAN, ``params`` for the compact model,
``model`` for DeOldify); this unwraps whichever is present. A leading
``module.`` prefix (DataParallel checkpoints) is stripped.
"""

from pathlib import Path

import torch

_WRAPPER_KEYS = ("params_ema", "params", "model", "state_dict")


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    """Load a ``.pth`` file safely and return its plain ``{name: tensor}`` dict."""
    with torch.serialization.safe_globals([slice]):
        obj = torch.load(path, map_location="cpu", weights_only=True)
    sd = obj
    if isinstance(obj, dict):
        for key in _WRAPPER_KEYS:
            if key in obj and isinstance(obj[key], dict):
                sd = obj[key]
                break
    if any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    return sd
