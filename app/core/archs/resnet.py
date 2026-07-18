"""Minimal, inference-only ResNet encoders (vendored).

torchvision's compiled ops are not always importable (version-skewed C
extensions), and CLAUDE.md §3 mandates vendoring minimal architecture code
rather than depending on unmaintained/mismatched packages. This reimplements
ResNet-34 (BasicBlock) and ResNet-101 (Bottleneck) with module/attribute names
identical to torchvision so the original DeOldify encoder weights load key-for-key.

Attribution: architecture after "Deep Residual Learning for Image Recognition"
(He et al., 2015); layout mirrors torchvision.models.resnet (BSD-3-Clause).
"""

import torch
from torch import nn


def _conv3x3(ni: int, nf: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(ni, nf, 3, stride=stride, padding=1, bias=False)


def _conv1x1(ni: int, nf: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(ni, nf, 1, stride=stride, bias=False)


class _Block(nn.Module):
    """Base for residual blocks; ``expansion`` scales the output channel count."""

    expansion: int = 1


class BasicBlock(_Block):
    expansion = 1

    def __init__(
        self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None
    ) -> None:
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):  # type: ignore[no-untyped-def]
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class Bottleneck(_Block):
    expansion = 4

    def __init__(
        self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None
    ) -> None:
        super().__init__()
        self.conv1 = _conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = _conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = _conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):  # type: ignore[no-untyped-def]
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet(nn.Module):
    """ResNet trunk with torchvision-compatible child ordering and names."""

    def __init__(self, block: type[_Block], layers: list[int]) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 1000)  # unused, kept for child parity

    def _make_layer(
        self, block: type[_Block], planes: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):  # type: ignore[no-untyped-def]
        # Standard torchvision ResNet forward (classification head). Consumers that
        # only want features slice the child modules instead of calling this.
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def resnet18() -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2])


def resnet34() -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3])


def resnet101() -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3])
