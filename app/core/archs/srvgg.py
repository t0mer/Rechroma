"""SRVGGNetCompact — the lightweight ``realesr-general-x4v3`` upscaler (vendored).

Inference-only reimplementation matching the released weights key-for-key.
Attribution: Real-ESRGAN ``realesrgan/archs/srvgg_arch.py`` (BSD-3-Clause).
"""

import torch
import torch.nn.functional as F
from torch import nn


class SRVGGNetCompact(nn.Module):
    """A plain VGG-style body with a pixel-shuffle upsampler and a global residual."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 32,
        upscale: int = 4,
    ) -> None:
        super().__init__()
        self.upscale = upscale
        body: list[nn.Module] = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), nn.PReLU(num_feat)]
        for _ in range(num_conv):
            body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            body.append(nn.PReLU(num_feat))
        body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.body = nn.Sequential(*body)
        self.upsampler = nn.PixelShuffle(upscale)
        self.num_out_ch = num_out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        # Global residual: add the nearest-upsampled input.
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base


def build_srvgg() -> SRVGGNetCompact:
    """Build the SRVGGNetCompact matching ``realesr-general-x4v3.pth``."""
    return SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4)
