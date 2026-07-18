"""Thin-Plate-Spline Motion Model — assembled network and safe weight loading.

Bundles the five TPSMM sub-networks (``kp_detector``, ``dense_motion_network``,
``inpainting_network``, ``bg_predictor``, ``avd_network``) under one module whose
child names match the released ``vox.pth.tar`` checkpoint, so each sub-network's
``state_dict`` loads with ``strict=True``.

Hyperparameters are the pinned VoxCeleb config (``config/vox-256.yaml`` upstream):
``num_tps=10``, ``num_channels=3``, dense-motion ``block_expansion=64``,
``max_features=1024``, ``num_blocks=5``, ``scale_factor=0.25``; generator
``block_expansion=64``, ``max_features=512``, ``num_down_blocks=3``; AVD bottlenecks
of 128. Inference-only — no training paths.

Attribution: github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model (MIT License).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .tpsmm_modules import (
    AVDNetwork,
    BGMotionPredictor,
    DenseMotionNetwork,
    InpaintingNetwork,
    KPDetector,
)

# Top-level checkpoint keys, in the order they appear in ``vox.pth.tar``.
_SUBNET_KEYS = (
    "kp_detector",
    "dense_motion_network",
    "inpainting_network",
    "bg_predictor",
    "avd_network",
)


class TPSMM(nn.Module):
    """Container holding the five TPSMM sub-networks under checkpoint-matching names."""

    def __init__(
        self,
        num_tps: int = 10,
        num_channels: int = 3,
        dm_block_expansion: int = 64,
        dm_max_features: int = 1024,
        dm_num_blocks: int = 5,
        dm_scale_factor: float = 0.25,
        gen_block_expansion: int = 64,
        gen_max_features: int = 512,
        gen_num_down_blocks: int = 3,
        avd_id_bottle_size: int = 128,
        avd_pose_bottle_size: int = 128,
    ) -> None:
        super().__init__()
        self.num_tps = num_tps
        self.kp_detector = KPDetector(num_tps=num_tps)
        self.dense_motion_network = DenseMotionNetwork(
            block_expansion=dm_block_expansion,
            num_blocks=dm_num_blocks,
            max_features=dm_max_features,
            num_tps=num_tps,
            num_channels=num_channels,
            scale_factor=dm_scale_factor,
            bg=True,
            multi_mask=True,
        )
        self.inpainting_network = InpaintingNetwork(
            num_channels=num_channels,
            block_expansion=gen_block_expansion,
            max_features=gen_max_features,
            num_down_blocks=gen_num_down_blocks,
            multi_mask=True,
        )
        self.bg_predictor = BGMotionPredictor(num_channels=num_channels)
        self.avd_network = AVDNetwork(
            num_tps=num_tps,
            id_bottle_size=avd_id_bottle_size,
            pose_bottle_size=avd_pose_bottle_size,
        )

    # -- inference helpers --------------------------------------------------

    @torch.no_grad()
    def detect_keypoints(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Detect TPS foreground keypoints for a ``(B, 3, H, W)`` image in ``[0, 1]``."""
        return self.kp_detector(image)

    @torch.no_grad()
    def predict_bg_param(self, source: torch.Tensor, driving: torch.Tensor) -> torch.Tensor:
        """Predict the background affine 3x3 matrix from source + driving frames."""
        return self.bg_predictor(source, driving)

    @torch.no_grad()
    def animate(
        self,
        source: torch.Tensor,
        kp_source: dict[str, torch.Tensor],
        kp_driving: dict[str, torch.Tensor],
        bg_param: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Render one animated frame: dense motion + inpainting generator.

        Returns the ``(B, 3, H, W)`` predicted frame in ``[0, 1]``.
        """
        dense_motion = self.dense_motion_network(
            source_image=source,
            kp_driving=kp_driving,
            kp_source=kp_source,
            bg_param=bg_param,
        )
        out = self.inpainting_network(source, dense_motion)
        return out["prediction"]

    @torch.no_grad()
    def forward(  # type: ignore[no-untyped-def]
        self,
        source: torch.Tensor,
        driving: torch.Tensor,
        use_bg: bool = True,
    ):
        """Convenience: animate ``source`` with the pose of ``driving`` (both ``[0,1]``)."""
        kp_source = self.detect_keypoints(source)
        kp_driving = self.detect_keypoints(driving)
        bg_param = self.predict_bg_param(source, driving) if use_bg else None
        return self.animate(source, kp_source, kp_driving, bg_param)


def build_tpsmm() -> TPSMM:
    """Build the VoxCeleb-256 TPSMM with randomly initialised weights."""
    return TPSMM()


def _safe_load(path: Path) -> dict[str, dict[str, torch.Tensor]]:
    """Load the multi-net checkpoint with ``weights_only=True`` (CLAUDE.md §10)."""
    with torch.serialization.safe_globals([slice]):
        obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict):
        raise ValueError(f"unexpected TPSMM checkpoint type: {type(obj)!r}")
    return obj


def load_tpsmm(path: str | Path) -> TPSMM:
    """Build a TPSMM and strictly load every sub-network from ``vox.pth.tar``."""
    ckpt = _safe_load(Path(path))
    model = build_tpsmm()
    submodules: dict[str, nn.Module] = {
        "kp_detector": model.kp_detector,
        "dense_motion_network": model.dense_motion_network,
        "inpainting_network": model.inpainting_network,
        "bg_predictor": model.bg_predictor,
        "avd_network": model.avd_network,
    }
    for name in _SUBNET_KEYS:
        if name not in ckpt:
            raise KeyError(f"checkpoint missing sub-network '{name}'")
        submodules[name].load_state_dict(ckpt[name], strict=True)
    model.eval()
    return model
