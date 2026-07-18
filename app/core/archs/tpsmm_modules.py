"""Thin-Plate-Spline Motion Model — inference building blocks (vendored).

Inference-only reimplementation of the TPSMM (CVPR 2022) sub-networks and their
shared utilities, matching the released ``vox.pth.tar`` weights key-for-key.

Attribution: Zhao & Zhang, "Thin-Plate Spline Motion Model for Image Animation"
(CVPR 2022). Ported from ``modules/util.py``, ``modules/keypoint_detector.py``,
``modules/dense_motion.py``, ``modules/inpainting_network.py``,
``modules/bg_motion_predictor.py`` and ``modules/avd_network.py`` of
github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model (MIT License).

We vendor rather than pip-install the upstream repo (CLAUDE.md §3): it is not a
package and would drag in fastai/torchvision-op dependencies. The ResNet-18 used
by the keypoint/background encoders comes from the repo's own vendored
``resnet.py`` so it loads without an importable torchvision.

MIT License. Copyright (c) 2021 yoyo-nb. See upstream LICENSE for full text.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .resnet import resnet18

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def make_coordinate_grid(
    spatial_size: torch.Size | tuple[int, int], dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Create a meshgrid ``[-1, 1] x [-1, 1]`` of the given spatial size."""
    h, w = spatial_size
    x = torch.arange(w, dtype=dtype)
    y = torch.arange(h, dtype=dtype)
    x = 2 * (x / (w - 1)) - 1
    y = 2 * (y / (h - 1)) - 1
    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)
    return torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2)


def kp2gaussian(
    kp: torch.Tensor, spatial_size: torch.Size | tuple[int, int], kp_variance: float
) -> torch.Tensor:
    """Transform keypoints into a gaussian heatmap representation."""
    grid = make_coordinate_grid(spatial_size, kp.dtype).to(kp.device)
    lead = len(kp.shape) - 1
    grid = grid.view((1,) * lead + grid.shape)
    grid = grid.repeat(*(kp.shape[:lead] + (1, 1, 1)))
    kp = kp.view(kp.shape[:lead] + (1, 1, 2))
    mean_sub = grid - kp
    return torch.exp(-0.5 * (mean_sub**2).sum(-1) / kp_variance)


def to_homogeneous(coordinates: torch.Tensor) -> torch.Tensor:
    ones_shape = list(coordinates.shape)
    ones_shape[-1] = 1
    ones = torch.ones(ones_shape).type(coordinates.type())
    return torch.cat([coordinates, ones], dim=-1)


def from_homogeneous(coordinates: torch.Tensor) -> torch.Tensor:
    return coordinates[..., :2] / coordinates[..., 2:3]


class TPS:
    """Thin-plate-spline transform (inference ``mode='kp'`` only, Eq. (2))."""

    def __init__(self, bs: int, kp_1: torch.Tensor, kp_2: torch.Tensor) -> None:
        self.bs = bs
        device = kp_1.device
        kp_type = kp_1.type()
        self.gs = kp_1.shape[1]
        n = kp_1.shape[2]

        k = torch.norm(kp_1[:, :, :, None] - kp_1[:, :, None, :], dim=4, p=2)
        k = k**2
        k = k * torch.log(k + 1e-9)

        one1 = torch.ones(bs, kp_1.shape[1], kp_1.shape[2], 1).to(device).type(kp_type)
        kp_1p = torch.cat([kp_1, one1], 3)

        zero = torch.zeros(bs, kp_1.shape[1], 3, 3).to(device).type(kp_type)
        p = torch.cat([kp_1p, zero], 2)
        left = torch.cat([k, kp_1p.permute(0, 1, 3, 2)], 2)
        left = torch.cat([left, p], 3)

        zero = torch.zeros(bs, kp_1.shape[1], 3, 2).to(device).type(kp_type)
        y = torch.cat([kp_2, zero], 2)
        eye = torch.eye(left.shape[2]).expand(left.shape).to(device).type(kp_type) * 0.01
        left = left + eye

        param = torch.matmul(torch.inverse(left), y)
        self.theta = param[:, :, n:, :].permute(0, 1, 3, 2)
        self.control_points = kp_1
        self.control_params = param[:, :, :n, :]

    def transform_frame(self, frame: torch.Tensor) -> torch.Tensor:
        grid = make_coordinate_grid(frame.shape[2:], frame.dtype).unsqueeze(0).to(frame.device)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        shape = [self.bs, self.gs, frame.shape[2], frame.shape[3], 2]
        return self.warp_coordinates(grid).view(*shape)

    def warp_coordinates(self, coordinates: torch.Tensor) -> torch.Tensor:
        theta = self.theta.type(coordinates.type()).to(coordinates.device)
        control_points = self.control_points.type(coordinates.type()).to(coordinates.device)
        control_params = self.control_params.type(coordinates.type()).to(coordinates.device)

        transformed = (
            torch.matmul(theta[:, :, :, :2], coordinates.permute(0, 2, 1)) + theta[:, :, :, 2:]
        )
        distances = coordinates.view(coordinates.shape[0], 1, 1, -1, 2) - control_points.view(
            self.bs, control_points.shape[1], -1, 1, 2
        )
        distances = distances**2
        result = distances.sum(-1)
        result = result * torch.log(result + 1e-9)
        result = torch.matmul(result.permute(0, 1, 3, 2), control_params)
        return transformed.permute(0, 1, 3, 2) + result


# ---------------------------------------------------------------------------
# Convolutional blocks
# ---------------------------------------------------------------------------


class ResBlock2d(nn.Module):
    """Residual block that preserves spatial resolution."""

    def __init__(self, in_features: int, kernel_size: int, padding: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=kernel_size, padding=padding)
        self.norm1 = nn.InstanceNorm2d(in_features, affine=True)
        self.norm2 = nn.InstanceNorm2d(in_features, affine=True)

    def forward(self, x):  # type: ignore[no-untyped-def]
        out = self.conv1(F.relu(self.norm1(x)))
        out = self.conv2(F.relu(self.norm2(out)))
        return out + x


class UpBlock2d(nn.Module):
    """Upsampling block used in decoders."""

    def __init__(
        self, in_features: int, out_features: int, kernel_size: int = 3, padding: int = 1
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=kernel_size, padding=padding)
        self.norm = nn.InstanceNorm2d(out_features, affine=True)

    def forward(self, x):  # type: ignore[no-untyped-def]
        out = F.interpolate(x, scale_factor=2)
        return F.relu(self.norm(self.conv(out)))


class DownBlock2d(nn.Module):
    """Downsampling block used in encoders."""

    def __init__(
        self, in_features: int, out_features: int, kernel_size: int = 3, padding: int = 1
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=kernel_size, padding=padding)
        self.norm = nn.InstanceNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):  # type: ignore[no-untyped-def]
        return self.pool(F.relu(self.norm(self.conv(x))))


class SameBlock2d(nn.Module):
    """Block that preserves spatial resolution."""

    def __init__(
        self, in_features: int, out_features: int, kernel_size: int = 3, padding: int = 1
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=kernel_size, padding=padding)
        self.norm = nn.InstanceNorm2d(out_features, affine=True)

    def forward(self, x):  # type: ignore[no-untyped-def]
        return F.relu(self.norm(self.conv(x)))


class Encoder(nn.Module):
    """Hourglass encoder."""

    def __init__(
        self, block_expansion: int, in_features: int, num_blocks: int = 3, max_features: int = 256
    ) -> None:
        super().__init__()
        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock2d(
                    in_features if i == 0 else min(max_features, block_expansion * (2**i)),
                    min(max_features, block_expansion * (2 ** (i + 1))),
                    kernel_size=3,
                    padding=1,
                )
            )
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):  # type: ignore[no-untyped-def]
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    """Hourglass decoder returning multi-scale feature maps."""

    def __init__(
        self, block_expansion: int, in_features: int, num_blocks: int = 3, max_features: int = 256
    ) -> None:
        super().__init__()
        up_blocks = []
        self.out_channels: list[int] = []
        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(
                max_features, block_expansion * (2 ** (i + 1))
            )
            self.out_channels.append(in_filters)
            out_filters = min(max_features, block_expansion * (2**i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_channels.append(block_expansion + in_features)

    def forward(self, x, mode: int = 0):  # type: ignore[no-untyped-def]
        out = x.pop()
        outs = []
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
            outs.append(out)
        return out if mode == 0 else outs


class Hourglass(nn.Module):
    """Hourglass architecture (encoder + decoder)."""

    def __init__(
        self, block_expansion: int, in_features: int, num_blocks: int = 3, max_features: int = 256
    ) -> None:
        super().__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_channels = self.decoder.out_channels

    def forward(self, x, mode: int = 0):  # type: ignore[no-untyped-def]
        return self.decoder(self.encoder(x), mode)


class AntiAliasInterpolation2d(nn.Module):
    """Band-limited (gaussian) downsampling."""

    def __init__(self, channels: int, scale: float) -> None:
        super().__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_sizes = [kernel_size, kernel_size]
        sigmas = [sigma, sigma]
        kernel = torch.tensor(1.0)
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_sizes],
            indexing="ij",
        )
        for size, std, mgrid in zip(kernel_sizes, sigmas, meshgrids, strict=True):
            mean = (size - 1) / 2
            kernel = kernel * torch.exp(-((mgrid - mean) ** 2) / (2 * std**2))

        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels
        self.scale = scale

    def forward(self, x):  # type: ignore[no-untyped-def]
        if self.scale == 1.0:
            return x
        out = F.pad(x, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        return F.interpolate(out, scale_factor=(self.scale, self.scale))


# ---------------------------------------------------------------------------
# Sub-networks
# ---------------------------------------------------------------------------


class KPDetector(nn.Module):
    """Predict ``num_tps * 5`` foreground keypoints via a ResNet-18 backbone."""

    def __init__(self, num_tps: int) -> None:
        super().__init__()
        self.num_tps = num_tps
        self.fg_encoder = resnet18()
        num_features = self.fg_encoder.fc.in_features
        self.fg_encoder.fc = nn.Linear(num_features, num_tps * 5 * 2)

    def forward(self, image):  # type: ignore[no-untyped-def]
        fg_kp = self.fg_encoder(image)
        bs = fg_kp.shape[0]
        fg_kp = torch.sigmoid(fg_kp)
        fg_kp = fg_kp * 2 - 1
        return {"fg_kp": fg_kp.view(bs, self.num_tps * 5, -1)}


class BGMotionPredictor(nn.Module):
    """Predict a single background affine transform as a 3x3 matrix."""

    def __init__(self, num_channels: int = 3) -> None:
        super().__init__()
        self.bg_encoder = resnet18()
        self.bg_encoder.conv1 = nn.Conv2d(
            num_channels * 2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        num_features = self.bg_encoder.fc.in_features
        self.bg_encoder.fc = nn.Linear(num_features, 6)

    def forward(self, source_image, driving_image):  # type: ignore[no-untyped-def]
        bs = source_image.shape[0]
        out = torch.eye(3).unsqueeze(0).repeat(bs, 1, 1).type(source_image.type())
        prediction = self.bg_encoder(torch.cat([source_image, driving_image], dim=1))
        out[:, :2, :] = prediction.view(bs, 2, 3)
        return out


class DenseMotionNetwork(nn.Module):
    """Estimate optical flow + multi-resolution occlusion masks from TPS + affine."""

    def __init__(
        self,
        block_expansion: int,
        num_blocks: int,
        max_features: int,
        num_tps: int,
        num_channels: int,
        scale_factor: float = 0.25,
        bg: bool = False,
        multi_mask: bool = True,
        kp_variance: float = 0.01,
    ) -> None:
        super().__init__()
        if scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, scale_factor)
        self.scale_factor = scale_factor
        self.multi_mask = multi_mask

        self.hourglass = Hourglass(
            block_expansion=block_expansion,
            in_features=(num_channels * (num_tps + 1) + num_tps * 5 + 1),
            max_features=max_features,
            num_blocks=num_blocks,
        )
        hourglass_output_size = self.hourglass.out_channels
        self.maps = nn.Conv2d(
            hourglass_output_size[-1], num_tps + 1, kernel_size=(7, 7), padding=(3, 3)
        )

        if multi_mask:
            up = []
            self.up_nums = int(math.log(1 / scale_factor, 2))
            self.occlusion_num = 4
            channel = [hourglass_output_size[-1] // (2**i) for i in range(self.up_nums)]
            for i in range(self.up_nums):
                up.append(UpBlock2d(channel[i], channel[i] // 2, kernel_size=3, padding=1))
            self.up = nn.ModuleList(up)

            occ_channel = [
                hourglass_output_size[-i - 1]
                for i in range(self.occlusion_num - self.up_nums)[::-1]
            ]
            for i in range(self.up_nums):
                occ_channel.append(hourglass_output_size[-1] // (2 ** (i + 1)))
            occlusion = [
                nn.Conv2d(occ_channel[i], 1, kernel_size=(7, 7), padding=(3, 3))
                for i in range(self.occlusion_num)
            ]
            self.occlusion = nn.ModuleList(occlusion)
        else:
            self.occlusion = nn.ModuleList(
                [nn.Conv2d(hourglass_output_size[-1], 1, kernel_size=(7, 7), padding=(3, 3))]
            )

        self.num_tps = num_tps
        self.bg = bg
        self.kp_variance = kp_variance

    def create_heatmap_representations(self, source_image, kp_driving, kp_source):  # type: ignore[no-untyped-def]
        spatial_size = source_image.shape[2:]
        gaussian_driving = kp2gaussian(kp_driving["fg_kp"], spatial_size, self.kp_variance)
        gaussian_source = kp2gaussian(kp_source["fg_kp"], spatial_size, self.kp_variance)
        heatmap = gaussian_driving - gaussian_source
        zeros = (
            torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1])
            .type(heatmap.type())
            .to(heatmap.device)
        )
        return torch.cat([zeros, heatmap], dim=1)

    def create_transformations(self, source_image, kp_driving, kp_source, bg_param):  # type: ignore[no-untyped-def]
        bs, _, h, w = source_image.shape
        kp_1 = kp_driving["fg_kp"].view(bs, -1, 5, 2)
        kp_2 = kp_source["fg_kp"].view(bs, -1, 5, 2)
        trans = TPS(bs=bs, kp_1=kp_1, kp_2=kp_2)
        driving_to_source = trans.transform_frame(source_image)

        identity_grid = make_coordinate_grid((h, w), kp_1.dtype).to(kp_1.device)
        identity_grid = identity_grid.view(1, 1, h, w, 2).repeat(bs, 1, 1, 1, 1)
        if bg_param is not None:
            identity_grid = to_homogeneous(identity_grid)
            identity_grid = torch.matmul(
                bg_param.view(bs, 1, 1, 1, 3, 3), identity_grid.unsqueeze(-1)
            ).squeeze(-1)
            identity_grid = from_homogeneous(identity_grid)
        return torch.cat([identity_grid, driving_to_source], dim=1)

    def create_deformed_source_image(self, source_image, transformations):  # type: ignore[no-untyped-def]
        bs, _, h, w = source_image.shape
        source_repeat = (
            source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_tps + 1, 1, 1, 1, 1)
        )
        source_repeat = source_repeat.view(bs * (self.num_tps + 1), -1, h, w)
        transformations = transformations.view((bs * (self.num_tps + 1), h, w, -1))
        deformed = F.grid_sample(source_repeat, transformations, align_corners=True)
        return deformed.view((bs, self.num_tps + 1, -1, h, w))

    def forward(self, source_image, kp_driving, kp_source, bg_param=None):  # type: ignore[no-untyped-def]
        if self.scale_factor != 1:
            source_image = self.down(source_image)
        bs, _, h, w = source_image.shape

        out_dict: dict[str, object] = {}
        heatmap_representation = self.create_heatmap_representations(
            source_image, kp_driving, kp_source
        )
        transformations = self.create_transformations(source_image, kp_driving, kp_source, bg_param)
        deformed_source = self.create_deformed_source_image(source_image, transformations)
        out_dict["deformed_source"] = deformed_source
        deformed_source = deformed_source.view(bs, -1, h, w)
        model_input = torch.cat([heatmap_representation, deformed_source], dim=1).view(bs, -1, h, w)

        prediction = self.hourglass(model_input, mode=1)

        contribution_maps = self.maps(prediction[-1])
        contribution_maps = F.softmax(contribution_maps, dim=1)
        out_dict["contribution_maps"] = contribution_maps

        contribution_maps = contribution_maps.unsqueeze(2)
        transformations = transformations.permute(0, 1, 4, 2, 3)
        deformation = (transformations * contribution_maps).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)
        out_dict["deformation"] = deformation

        occlusion_map = []
        if self.multi_mask:
            for i in range(self.occlusion_num - self.up_nums):
                occlusion_map.append(
                    torch.sigmoid(
                        self.occlusion[i](prediction[self.up_nums - self.occlusion_num + i])
                    )
                )
            last = prediction[-1]
            for i in range(self.up_nums):
                last = self.up[i](last)
                occlusion_map.append(
                    torch.sigmoid(self.occlusion[i + self.occlusion_num - self.up_nums](last))
                )
        else:
            occlusion_map.append(torch.sigmoid(self.occlusion[0](prediction[-1])))
        out_dict["occlusion_map"] = occlusion_map
        return out_dict


class InpaintingNetwork(nn.Module):
    """Inpaint missing regions and reconstruct the animated (driving) frame."""

    def __init__(
        self,
        num_channels: int,
        block_expansion: int,
        max_features: int,
        num_down_blocks: int,
        multi_mask: bool = True,
    ) -> None:
        super().__init__()
        self.num_down_blocks = num_down_blocks
        self.multi_mask = multi_mask
        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=7, padding=3)

        down_blocks = []
        up_blocks = []
        resblock = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2**i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=3, padding=1))
            decoder_in_feature = out_features * 2
            if i == num_down_blocks - 1:
                decoder_in_feature = out_features
            up_blocks.append(UpBlock2d(decoder_in_feature, in_features, kernel_size=3, padding=1))
            resblock.append(ResBlock2d(decoder_in_feature, kernel_size=3, padding=1))
            resblock.append(ResBlock2d(decoder_in_feature, kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)
        self.up_blocks = nn.ModuleList(up_blocks[::-1])
        self.resblock = nn.ModuleList(resblock[::-1])

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.num_channels = num_channels

    def deform_input(self, inp, deformation):  # type: ignore[no-untyped-def]
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(
                deformation, size=(h, w), mode="bilinear", align_corners=True
            )
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation, align_corners=True)

    def occlude_input(self, inp, occlusion_map):  # type: ignore[no-untyped-def]
        if not self.multi_mask and (
            inp.shape[2] != occlusion_map.shape[2] or inp.shape[3] != occlusion_map.shape[3]
        ):
            occlusion_map = F.interpolate(
                occlusion_map, size=inp.shape[2:], mode="bilinear", align_corners=True
            )
        return inp * occlusion_map

    def forward(self, source_image, dense_motion):  # type: ignore[no-untyped-def]
        out = self.first(source_image)
        encoder_map = [out]
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
            encoder_map.append(out)

        output_dict: dict[str, object] = {}
        output_dict["contribution_maps"] = dense_motion["contribution_maps"]
        output_dict["deformed_source"] = dense_motion["deformed_source"]
        occlusion_map = dense_motion["occlusion_map"]
        output_dict["occlusion_map"] = occlusion_map

        deformation = dense_motion["deformation"]
        out = self.deform_input(out, deformation)
        out = self.occlude_input(out, occlusion_map[0])

        encode_i = out
        for i in range(self.num_down_blocks):
            out = self.resblock[2 * i](out)
            out = self.resblock[2 * i + 1](out)
            out = self.up_blocks[i](out)

            encode_i = encoder_map[-(i + 2)]
            encode_i = self.deform_input(encode_i, deformation)
            occlusion_ind = i + 1 if self.multi_mask else 0
            encode_i = self.occlude_input(encode_i, occlusion_map[occlusion_ind])

            if i == self.num_down_blocks - 1:
                break
            out = torch.cat([out, encode_i], 1)

        deformed_source = self.deform_input(source_image, deformation)
        output_dict["deformed"] = deformed_source

        occlusion_last = occlusion_map[-1]
        if not self.multi_mask:
            occlusion_last = F.interpolate(
                occlusion_last, size=out.shape[2:], mode="bilinear", align_corners=True
            )
        out = out * (1 - occlusion_last) + encode_i
        out = torch.sigmoid(self.final(out))
        out = out * (1 - occlusion_last) + deformed_source * occlusion_last
        output_dict["prediction"] = out
        return output_dict


class AVDNetwork(nn.Module):
    """Animation-via-disentanglement network (identity/pose decoupling)."""

    def __init__(
        self, num_tps: int, id_bottle_size: int = 128, pose_bottle_size: int = 128
    ) -> None:
        super().__init__()
        input_size = 5 * 2 * num_tps
        self.num_tps = num_tps

        self.id_encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, id_bottle_size),
        )
        self.pose_encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, pose_bottle_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(pose_bottle_size + id_bottle_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, input_size),
        )

    def forward(self, kp_source, kp_random):  # type: ignore[no-untyped-def]
        bs = kp_source["fg_kp"].shape[0]
        pose_emb = self.pose_encoder(kp_random["fg_kp"].view(bs, -1))
        id_emb = self.id_encoder(kp_source["fg_kp"].view(bs, -1))
        rec = self.decoder(torch.cat([pose_emb, id_emb], dim=1))
        return {"fg_kp": rec.view(bs, self.num_tps * 5, -1)}
