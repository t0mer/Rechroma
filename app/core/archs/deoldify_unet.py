"""Assembly of the fastai-free DeOldify generator (``DynamicUnet`` reconstruction).

The released weights are plain ``state_dict``s whose keys follow fastai's
``layers.<N>`` numbering. This module rebuilds that exact module tree in modern
PyTorch so the weights load with ``strict=True`` — no fastai, inference only
(CLAUDE.md §2). Per-backbone decoder dimensions are read directly from the
released weights and recorded in ``_CONFIG``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import torch
from torch import nn

from .layers import (
    MergeLayer,
    PixelShuffleICNR,
    ResBlock,
    SigmoidRange,
    UnetBlock,
    UnetBlockWide,
    custom_conv_layer,
)
from .resnet import resnet34, resnet101

Backbone = Literal["resnet34", "resnet101"]

# Indices within the encoder ``nn.Sequential`` (resnet children[:-2]) whose
# outputs feed U-Net cross connections, deepest last. Same for both resnets:
# stem-relu, layer1, layer2, layer3.
_SKIP_INDICES = (2, 4, 5, 6)


@dataclass(frozen=True)
class _DecoderBlock:
    up_in: int
    skip: int
    nf: int
    self_attention: bool


@dataclass(frozen=True)
class _Config:
    deep_channels: int  # encoder output channels (layer4)
    blocks: tuple[_DecoderBlock, ...]
    final_nf: int  # channels entering the final pixel-shuffle
    wide: bool  # True -> UnetBlockWide (single conv); False -> UnetBlock (two convs)
    y_range: tuple[float, float]


# Configs verified against the released weights (key-for-key state_dict match).
# resnet34 = artistic "deep"; resnet101 = stable "wide".
_CONFIG: dict[Backbone, _Config] = {
    "resnet34": _Config(
        deep_channels=512,
        blocks=(
            _DecoderBlock(up_in=512, skip=256, nf=768, self_attention=False),
            _DecoderBlock(up_in=768, skip=128, nf=768, self_attention=True),
            _DecoderBlock(up_in=768, skip=64, nf=672, self_attention=False),
            _DecoderBlock(up_in=672, skip=64, nf=300, self_attention=False),
        ),
        final_nf=300,
        wide=False,
        y_range=(-3.0, 3.0),
    ),
    "resnet101": _Config(
        deep_channels=2048,
        blocks=(
            _DecoderBlock(up_in=2048, skip=1024, nf=512, self_attention=False),
            _DecoderBlock(up_in=512, skip=512, nf=512, self_attention=True),
            _DecoderBlock(up_in=512, skip=256, nf=512, self_attention=False),
            _DecoderBlock(up_in=512, skip=64, nf=256, self_attention=False),
        ),
        final_nf=256,
        wide=True,
        y_range=(-3.0, 3.0),
    ),
}


def _encoder(backbone: Backbone) -> nn.Sequential:
    factory = {"resnet34": resnet34, "resnet101": resnet101}[backbone]
    net = factory()
    # Drop avgpool + fc, keeping [conv1, bn1, relu, maxpool, layer1..4].
    return nn.Sequential(*list(net.children())[:-2])


class DeOldifyGenerator(nn.Module):
    """DynamicUnet-shaped colorization generator (fastai-free)."""

    def __init__(self, backbone: Backbone) -> None:
        super().__init__()
        cfg = _CONFIG[backbone]
        enc = _encoder(backbone)
        ni = cfg.deep_channels

        block_cls = UnetBlockWide if cfg.wide else UnetBlock
        decoder = [block_cls(b.up_in, b.skip, b.nf, b.self_attention) for b in cfg.blocks]

        # Top-level module list; indices mirror the released ``layers.<N>`` keys.
        self.layers = nn.ModuleList(
            [
                enc,  # 0 encoder
                nn.BatchNorm2d(ni),  # 1
                nn.ReLU(inplace=True),  # 2 (no params)
                nn.Sequential(  # 3 middle conv
                    custom_conv_layer(ni, ni * 2, ks=3, norm="spectral"),
                    custom_conv_layer(ni * 2, ni, ks=3, norm="spectral"),
                ),
                *decoder,  # 4..7
                PixelShuffleICNR(cfg.final_nf, cfg.final_nf, scale=2, norm="weight"),  # 8
                MergeLayer(),  # 9 (no params) concat original input
                ResBlock(cfg.final_nf + 3),  # 10
                custom_conv_layer(  # 11 final conv -> 3 (Sequential; conv at .0)
                    cfg.final_nf + 3, 3, ks=1, use_activ=False, use_bn=False, norm="spectral"
                ),
                SigmoidRange(*cfg.y_range),  # 12 (no params)
            ]
        )
        self._skip_indices = _SKIP_INDICES

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        enc = cast(nn.Sequential, self.layers[0])
        skips: list[torch.Tensor] = []
        h = x
        for i, module in enumerate(enc):
            h = module(h)
            if i in self._skip_indices:
                skips.append(h)

        h = self.layers[2](self.layers[1](h))  # bn + relu
        h = self.layers[3](h)  # middle conv

        for block, skip in zip(self.layers[4:8], reversed(skips), strict=True):
            h = block(h, skip)

        h = self.layers[8](h)  # final pixel-shuffle to input resolution
        if h.shape[-2:] != inp.shape[-2:]:
            h = nn.functional.interpolate(h, size=inp.shape[-2:], mode="nearest")
        h = self.layers[9](h, inp)  # merge original input -> +3 channels
        h = self.layers[10](h)  # res block
        h = self.layers[11](h)  # final conv
        return self.layers[12](h)  # sigmoid range


def build_deoldify_generator(backbone: Backbone) -> DeOldifyGenerator:
    """Construct the generator for a backbone (``resnet34`` artistic / ``resnet101`` stable)."""
    if backbone not in _CONFIG:
        raise NotImplementedError(f"backbone {backbone!r} not yet configured")
    return DeOldifyGenerator(backbone)


def load_state_dict_file(path: Path) -> dict[str, torch.Tensor]:
    """Safely load a weights file (``weights_only=True``) and unwrap ``{'model': ...}``.

    DeOldify checkpoints need only Python's builtin ``slice`` allow-listed; nothing
    else is unpickled, so arbitrary-code execution is not possible (CLAUDE.md §10).
    """
    with torch.serialization.safe_globals([slice]):
        obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    return obj
