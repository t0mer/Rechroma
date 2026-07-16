"""Inference-only port of the RetinaFace face detector (ResNet-50 backbone).

Reconstructs the RetinaFace network (ResNet-50 + FPN + SSH + Class/Bbox/Landmark
heads) so the *original* released weights (``detection_Resnet50_Final.pth``) load
with ``load_state_dict(strict=True)`` after stripping the ``module.`` prefix.
Also provides prior-box generation and box/landmark decoding + NMS helpers so
detections can be turned into face boxes without depending on facexlib.

Attribution: architecture from xinntao/facexlib
(``facexlib/detection/retinaface.py`` and ``retinaface_net.py``), MIT/BSD-style
license; RetinaFace paper (Deng et al., 2019). The ResNet-50 backbone reuses the
vendored torchvision-compatible ``Bottleneck`` block (BSD-3, see ``resnet.py``).
Reimplemented for inference only.
"""

from __future__ import annotations

from itertools import product
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from .resnet import Bottleneck, _conv1x1

# RetinaFace ResNet-50 configuration (facexlib ``cfg_re50``).
CFG_RE50: dict = {
    "name": "Resnet50",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
    "in_channel": 256,
    "out_channel": 256,
}


def _conv_bn(inp: int, oup: int, stride: int = 1, leaky: float = 0) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def _conv_bn_no_relu(inp: int, oup: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )


def _conv_bn1x1(inp: int, oup: int, stride: int = 1, leaky: float = 0) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


class SSH(nn.Module):
    """Single-stage headless context module."""

    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__()
        assert out_channel % 4 == 0
        leaky = 0.1 if out_channel <= 64 else 0
        self.conv3X3 = _conv_bn_no_relu(in_channel, out_channel // 2, stride=1)
        self.conv5X5_1 = _conv_bn(in_channel, out_channel // 4, stride=1, leaky=leaky)
        self.conv5X5_2 = _conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)
        self.conv7X7_2 = _conv_bn(out_channel // 4, out_channel // 4, stride=1, leaky=leaky)
        self.conv7x7_3 = _conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv3x3 = self.conv3X3(x)
        conv5x5_1 = self.conv5X5_1(x)
        conv5x5 = self.conv5X5_2(conv5x5_1)
        conv7x7_2 = self.conv7X7_2(conv5x5_1)
        conv7x7 = self.conv7x7_3(conv7x7_2)
        out = torch.cat([conv3x3, conv5x5, conv7x7], dim=1)
        return F.relu(out)


class FPN(nn.Module):
    """Feature pyramid network over three backbone stages."""

    def __init__(self, in_channels_list: list[int], out_channels: int) -> None:
        super().__init__()
        leaky = 0.1 if out_channels <= 64 else 0
        self.output1 = _conv_bn1x1(in_channels_list[0], out_channels, stride=1, leaky=leaky)
        self.output2 = _conv_bn1x1(in_channels_list[1], out_channels, stride=1, leaky=leaky)
        self.output3 = _conv_bn1x1(in_channels_list[2], out_channels, stride=1, leaky=leaky)
        self.merge1 = _conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = _conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, x: dict[int, torch.Tensor]) -> list[torch.Tensor]:
        feats = list(x.values())
        output1 = self.output1(feats[0])
        output2 = self.output2(feats[1])
        output3 = self.output3(feats[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)
        return [output1, output2, output3]


class ClassHead(nn.Module):
    def __init__(self, inchannels: int = 512, num_anchors: int = 2) -> None:
        super().__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 2, kernel_size=(1, 1), stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1x1(x).permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, inchannels: int = 512, num_anchors: int = 2) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1x1(x).permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, inchannels: int = 512, num_anchors: int = 2) -> None:
        super().__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=(1, 1), stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1x1(x).permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 10)


class _ResNet50Body(nn.Module):
    """ResNet-50 trunk exposing intermediate layer2/layer3/layer4 features.

    Mirrors ``torchvision.models._utils.IntermediateLayerGetter`` over a
    resnet50 with ``return_layers={'layer2': 1, 'layer3': 2, 'layer4': 3}`` so
    the checkpoint's ``body.*`` keys load key-for-key.
    """

    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * Bottleneck.expansion, stride),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return {1: c3, 2: c4, 3: c5}


class RetinaFace(nn.Module):
    """RetinaFace detector with a ResNet-50 backbone (test/inference phase)."""

    def __init__(self, cfg: dict = CFG_RE50) -> None:
        super().__init__()
        self.cfg = cfg
        self.body = _ResNet50Body()

        in_channels_stage2 = cfg["in_channel"]
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg["out_channel"]
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = nn.ModuleList([ClassHead(out_channels) for _ in range(3)])
        self.BboxHead = nn.ModuleList([BboxHead(out_channels) for _ in range(3)])
        self.LandmarkHead = nn.ModuleList([LandmarkHead(out_channels) for _ in range(3)])

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.body(inputs)
        fpn = self.fpn(out)
        features = [self.ssh1(fpn[0]), self.ssh2(fpn[1]), self.ssh3(fpn[2])]

        bbox_regressions = torch.cat([self.BboxHead[i](f) for i, f in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](f) for i, f in enumerate(features)], dim=1)
        ldm_regressions = torch.cat(
            [self.LandmarkHead[i](f) for i, f in enumerate(features)], dim=1
        )
        return bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions


def build_retinaface() -> nn.Module:
    """Build a RetinaFace (ResNet-50) detector matching ``detection_Resnet50_Final``."""
    return RetinaFace(CFG_RE50)


def strip_module_prefix(state_dict: dict) -> dict:
    """Remove a leading ``module.`` (DataParallel) prefix from every key."""
    return {
        (k[len("module.") :] if k.startswith("module.") else k): v for k, v in state_dict.items()
    }


def prior_box(image_size: tuple[int, int], cfg: dict = CFG_RE50) -> torch.Tensor:
    """Generate normalised prior anchors (cx, cy, w, h) for a given image size."""
    steps = cfg["steps"]
    min_sizes_cfg = cfg["min_sizes"]
    feature_maps = [[ceil(image_size[0] / step), ceil(image_size[1] / step)] for step in steps]
    anchors: list[float] = []
    for k, f in enumerate(feature_maps):
        min_sizes = min_sizes_cfg[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                cx = (j + 0.5) * steps[k] / image_size[1]
                cy = (i + 0.5) * steps[k] / image_size[0]
                anchors += [cx, cy, s_kx, s_ky]
    output = torch.tensor(anchors).view(-1, 4)
    if cfg["clip"]:
        output.clamp_(max=1, min=0)
    return output


def decode_boxes(loc: torch.Tensor, priors: torch.Tensor, variances: list[float]) -> torch.Tensor:
    """Decode predicted locations back into (x1, y1, x2, y2) boxes."""
    boxes = torch.cat(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1]),
        ),
        1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landms(pre: torch.Tensor, priors: torch.Tensor, variances: list[float]) -> torch.Tensor:
    """Decode predicted landmark offsets into 5 (x, y) coordinates."""
    return torch.cat(
        (
            priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
        ),
        dim=1,
    )


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    """Greedy IoU non-maximum suppression.

    Uses ``torchvision.ops.nms`` when its compiled op is importable, otherwise a
    pure-numpy fallback (torchvision's C extension is not always available).
    """
    try:
        from torchvision.ops import nms as tv_nms

        keep = tv_nms(
            torch.as_tensor(boxes, dtype=torch.float32),
            torch.as_tensor(scores, dtype=torch.float32),
            threshold,
        )
        return keep.cpu().numpy().tolist()
    except Exception:
        return _nms_numpy(boxes, scores, threshold)


def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]
    return keep
