"""Inference-only port of ParseNet (face parsing / segmentation).

Reconstructs the ParseNet encoder-body-decoder network so the *original*
released weights (``parsing_parsenet.pth``) load with
``load_state_dict(strict=True)``. Given a 512x512 aligned face it produces a
19-class parsing map (plus a reconstructed image branch).

Attribution: architecture from xinntao/facexlib
(``facexlib/parsing/parsenet.py``), MIT-style license. Reimplemented for
inference only (training-only initialisation dropped). Normalisation and ReLU
variants are kept configurable to match the checkpoint module tree exactly
(``conv2d`` / ``norm.norm`` naming).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class NormLayer(nn.Module):
    """Normalisation wrapper; only batch-norm is needed for the released weights."""

    def __init__(self, channels: int, norm_type: str = "bn") -> None:
        super().__init__()
        self.norm_type = norm_type.lower()
        if self.norm_type == "bn":
            self.norm: nn.Module = nn.BatchNorm2d(channels, affine=True)
        elif self.norm_type == "in":
            self.norm = nn.InstanceNorm2d(channels, affine=False)
        elif self.norm_type == "gn":
            self.norm = nn.GroupNorm(32, channels, affine=True)
        elif self.norm_type == "none":
            self.norm = nn.Identity()
        else:
            raise ValueError(f"unsupported norm type: {norm_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class ReluLayer(nn.Module):
    """Activation wrapper; parameter-free variants keep the checkpoint tree clean."""

    def __init__(self, channels: int, relu_type: str = "relu") -> None:
        super().__init__()
        relu_type = relu_type.lower()
        if relu_type == "relu":
            self.func: nn.Module = nn.ReLU(inplace=True)
        elif relu_type == "leakyrelu":
            self.func = nn.LeakyReLU(0.2, inplace=True)
        elif relu_type == "prelu":
            self.func = nn.PReLU(channels)
        elif relu_type == "selu":
            self.func = nn.SELU(inplace=True)
        elif relu_type == "none":
            self.func = nn.Identity()
        else:
            raise ValueError(f"unsupported relu type: {relu_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.func(x)


class ConvLayer(nn.Module):
    """Reflection-padded conv with optional rescale, norm and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        scale: str = "none",
        norm_type: str = "none",
        relu_type: str = "none",
        use_pad: bool = True,
    ) -> None:
        super().__init__()
        self.use_pad = use_pad
        bias = norm_type != "bn"
        stride = 2 if scale == "down" else 1

        self.scale = scale
        self.reflection_pad = nn.ReflectionPad2d(int(np.ceil((kernel_size - 1.0) / 2)))
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)
        self.relu = ReluLayer(out_channels, relu_type)
        self.norm = NormLayer(out_channels, norm_type=norm_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        if self.scale == "up":
            out = F.interpolate(out, scale_factor=2, mode="nearest")
        if self.use_pad:
            out = self.reflection_pad(out)
        out = self.conv2d(out)
        out = self.norm(out)
        return self.relu(out)


class ResidualBlock(nn.Module):
    """Two-conv residual block with an identity / conv shortcut."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        relu_type: str = "prelu",
        norm_type: str = "bn",
        scale: str = "none",
    ) -> None:
        super().__init__()
        if scale == "none" and c_in == c_out:
            self.shortcut_func: nn.Module = nn.Identity()
        else:
            self.shortcut_func = ConvLayer(c_in, c_out, 3, scale)

        scale_config = {
            "down": ["none", "down"],
            "up": ["up", "none"],
            "none": ["none", "none"],
        }[scale]
        self.conv1 = ConvLayer(
            c_in, c_out, 3, scale_config[0], norm_type=norm_type, relu_type=relu_type
        )
        self.conv2 = ConvLayer(
            c_out, c_out, 3, scale_config[1], norm_type=norm_type, relu_type="none"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut_func(x)
        res = self.conv2(self.conv1(x))
        return identity + res


class ParseNet(nn.Module):
    """Face parsing network: encoder → residual body → decoder → mask/image heads."""

    def __init__(
        self,
        in_size: int = 512,
        out_size: int = 512,
        min_feat_size: int = 32,
        base_ch: int = 64,
        parsing_ch: int = 19,
        res_depth: int = 10,
        relu_type: str = "LeakyReLU",
        norm_type: str = "bn",
        ch_range: tuple[int, int] = (32, 256),
    ) -> None:
        super().__init__()
        act_args = {"norm_type": norm_type, "relu_type": relu_type}
        min_ch, max_ch = ch_range

        def ch_clip(x: int) -> int:
            return max(min_ch, min(x, max_ch))

        min_feat_size = min(in_size, min_feat_size)
        down_steps = int(np.log2(in_size // min_feat_size))
        up_steps = int(np.log2(out_size // min_feat_size))

        encoder: list[nn.Module] = [ConvLayer(3, base_ch, 3, "none")]
        head_ch = base_ch
        for _ in range(down_steps):
            cin, cout = ch_clip(head_ch), ch_clip(head_ch * 2)
            encoder.append(ResidualBlock(cin, cout, scale="down", **act_args))
            head_ch = head_ch * 2

        body: list[nn.Module] = []
        for _ in range(res_depth):
            body.append(ResidualBlock(ch_clip(head_ch), ch_clip(head_ch), **act_args))

        decoder: list[nn.Module] = []
        for _ in range(up_steps):
            cin, cout = ch_clip(head_ch), ch_clip(head_ch // 2)
            decoder.append(ResidualBlock(cin, cout, scale="up", **act_args))
            head_ch = head_ch // 2

        self.encoder = nn.Sequential(*encoder)
        self.body = nn.Sequential(*body)
        self.decoder = nn.Sequential(*decoder)
        self.out_img_conv = ConvLayer(ch_clip(head_ch), 3)
        self.out_mask_conv = ConvLayer(ch_clip(head_ch), parsing_ch)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(x)
        x = feat + self.body(feat)
        x = self.decoder(x)
        out_mask = self.out_mask_conv(x)
        out_img = self.out_img_conv(x)
        return out_mask, out_img


def build_parsenet() -> nn.Module:
    """Build a ParseNet matching the released ``parsing_parsenet`` weights."""
    return ParseNet(
        in_size=512,
        out_size=512,
        min_feat_size=32,
        base_ch=64,
        parsing_ch=19,
        res_depth=10,
        relu_type="LeakyReLU",
        norm_type="bn",
        ch_range=(32, 256),
    )
