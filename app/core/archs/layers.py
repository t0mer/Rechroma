"""Building blocks for the fastai-free DeOldify generator.

These modules reproduce fastai's ``DynamicUnet`` primitives (custom conv layer,
pixel-shuffle-ICNR upsampler, self-attention, residual block) closely enough
that the original released ``state_dict`` loads key-for-key. Only inference is
supported; there is no fastai dependency (CLAUDE.md §2).

Key-name compatibility relies on the *classic* ``torch.nn.utils`` weight
parametrisations: ``spectral_norm`` → ``weight_orig/weight_u/weight_v`` and
``weight_norm`` → ``weight_g/weight_v``. The newer ``parametrizations`` variants
use different names and must not be substituted.
"""

import warnings
from typing import Literal

import torch
from torch import nn
from torch.nn.utils import spectral_norm, weight_norm

NormType = Literal["spectral", "weight", None]


def _conv(ni: int, nf: int, ks: int, stride: int, bias: bool, norm: NormType) -> nn.Module:
    pad = ks // 2
    conv = nn.Conv2d(ni, nf, ks, stride=stride, padding=pad, bias=bias)
    if norm == "spectral":
        return spectral_norm(conv)
    if norm == "weight":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return weight_norm(conv)
    return conv


def custom_conv_layer(
    ni: int,
    nf: int,
    ks: int = 3,
    stride: int = 1,
    use_activ: bool = True,
    use_bn: bool = True,
    norm: NormType = "spectral",
    self_attention: bool = False,
) -> nn.Sequential:
    """Conv (optionally normed) → ReLU → BatchNorm, matching fastai layer order.

    The conv carries a bias only when no BatchNorm follows it. When ``use_activ``
    is False the BatchNorm (if any) sits directly after the conv, reproducing the
    released key indices (``.0`` conv, ``.1``/``.2`` norm).
    """
    layers: list[nn.Module] = [_conv(ni, nf, ks, stride, bias=not use_bn, norm=norm)]
    if use_activ:
        layers.append(nn.ReLU(inplace=True))
    if use_bn:
        layers.append(nn.BatchNorm2d(nf))
    if self_attention:
        layers.append(SelfAttention(nf))
    return nn.Sequential(*layers)


class SelfAttention(nn.Module):
    """Self-attention block (fastai ``SelfAttention``), spectral-normed 1×1 conv1d."""

    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.query = spectral_norm(nn.Conv1d(n_channels, n_channels // 8, 1, bias=False))
        self.key = spectral_norm(nn.Conv1d(n_channels, n_channels // 8, 1, bias=False))
        self.value = spectral_norm(nn.Conv1d(n_channels, n_channels, 1, bias=False))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = torch.softmax(torch.bmm(f.transpose(1, 2), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()


class PixelShuffleICNR(nn.Module):
    """Pixel-shuffle upsampler with ICNR-style conv (fastai ``PixelShuffle_ICNR``)."""

    def __init__(self, ni: int, nf: int, scale: int = 2, norm: NormType = "spectral") -> None:
        super().__init__()
        # use_activ/use_bn mirror the released structure: spectral variant keeps a
        # BatchNorm (`.1`); the weight-norm variant is a bare conv.
        use_bn = norm == "spectral"
        self.conv = custom_conv_layer(
            ni, nf * (scale**2), ks=1, use_activ=False, use_bn=use_bn, norm=norm
        )
        self.shuf = nn.PixelShuffle(scale)
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x))


class MergeLayer(nn.Module):
    """Concatenate a stored residual onto the input (fastai ``MergeLayer(dense=True)``)."""

    def forward(self, x: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, res], dim=1)


class ResBlock(nn.Module):
    """Residual block: two spectral convs (no BN, with bias) plus a skip add.

    Registered as ``layers.0``/``layers.1`` sub-convs to match the released keys.
    """

    def __init__(self, nf: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                custom_conv_layer(nf, nf, ks=3, use_activ=True, use_bn=False, norm="spectral"),
                custom_conv_layer(nf, nf, ks=3, use_activ=True, use_bn=False, norm="spectral"),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out + x


class SigmoidRange(nn.Module):
    """Scale a sigmoid into ``[low, high]`` (fastai ``SigmoidRange``); no parameters."""

    def __init__(self, low: float, high: float) -> None:
        super().__init__()
        self.low, self.high = low, high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * (self.high - self.low) + self.low


class UnetBlock(nn.Module):
    """One decoder stage: pixel-shuffle the deep feature, merge the encoder skip.

    Children ``shuf``/``bn``/``conv1``/``conv2`` match the released key layout;
    ``conv2`` gains a ``SelfAttention`` child (``.3``) when ``self_attention``.
    """

    def __init__(self, up_in_c: int, x_in_c: int, nf: int, self_attention: bool = False) -> None:
        super().__init__()
        up_out = up_in_c // 2
        self.shuf = PixelShuffleICNR(up_in_c, up_out, scale=2, norm="spectral")
        self.bn = nn.BatchNorm2d(x_in_c)
        ni = up_out + x_in_c
        self.conv1 = custom_conv_layer(ni, nf, ks=3, norm="spectral")
        self.conv2 = custom_conv_layer(nf, nf, ks=3, norm="spectral", self_attention=self_attention)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, up_in: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up_out = self.shuf(up_in)
        if up_out.shape[-2:] != skip.shape[-2:]:
            up_out = nn.functional.interpolate(up_out, size=skip.shape[-2:], mode="nearest")
        cat = self.relu(torch.cat([up_out, self.bn(skip)], dim=1))
        return self.conv2(self.conv1(cat))


class UnetBlockWide(nn.Module):
    """Decoder stage for the "wide" DeOldify unet (stable): a single fused conv.

    Unlike :class:`UnetBlock` (two convs), the wide block pixel-shuffles the deep
    feature to ``n_out`` channels and applies one ``conv`` over the concatenation
    with the encoder skip. ``conv`` gains a ``SelfAttention`` child (``.3``) when
    ``self_attention``. Matches the released ``ColorizeStable_gen.pth`` keys.
    """

    def __init__(self, up_in_c: int, x_in_c: int, n_out: int, self_attention: bool = False) -> None:
        super().__init__()
        self.shuf = PixelShuffleICNR(up_in_c, n_out, scale=2, norm="spectral")
        self.bn = nn.BatchNorm2d(x_in_c)
        ni = n_out + x_in_c
        self.conv = custom_conv_layer(
            ni, n_out, ks=3, norm="spectral", self_attention=self_attention
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, up_in: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up_out = self.shuf(up_in)
        if up_out.shape[-2:] != skip.shape[-2:]:
            up_out = nn.functional.interpolate(up_out, size=skip.shape[-2:], mode="nearest")
        cat = self.relu(torch.cat([up_out, self.bn(skip)], dim=1))
        return self.conv(cat)
