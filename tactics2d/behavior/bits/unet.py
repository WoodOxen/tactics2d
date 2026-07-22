# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""UNet decoder and rasterized-map backbone."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
from torchvision.models.feature_extraction import create_feature_extractor


class ConvBlock(nn.Module):
    """Official UNet helper block: conv, batchnorm, ReLU twice."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        mid_channels = int(mid_channels or out_channels)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.double_conv(inputs)


class Upsample(nn.Module):
    """Official-style bilinear upsample plus double conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_channels, out_channels, in_channels // 2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class BottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filters: tuple,
        stride: int = 1,
        final_relu: bool = True,
        shortcut: bool = False,
    ):
        super().__init__()
        self.final_relu = final_relu
        f1, f2, f3 = filters
        self.conv1 = nn.Conv2d(in_channels, f1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(f1)
        self.conv2 = nn.Conv2d(f1, f2, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(f2)
        self.conv3 = nn.Conv2d(f2, f3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(f3)
        if shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, f3, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(f3),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(inputs)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = x + self.shortcut(inputs)
        if self.final_relu:
            x = F.relu(x)
        return x


class BitsRasterBackbone(nn.Module):
    """ResNet raster encoder matching TBSIM's RasterizedMapEncoder layout."""

    def __init__(
        self,
        image_channels: int,
        model_arch: str = "resnet18",
        feature_dim: Optional[int] = None,
        output_activation=nn.ReLU,
    ):
        super().__init__()
        self.model_arch = model_arch
        self.num_input_channels = int(image_channels)
        self._feature_dim = feature_dim
        if model_arch == "resnet18":
            self.map_model = resnet18(weights=None)
        elif model_arch == "resnet50":
            self.map_model = resnet50(weights=None)
        else:
            raise ValueError("model_arch must be either 'resnet18' or 'resnet50'.")
        self.map_model.conv1 = nn.Conv2d(
            self.num_input_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        self.map_model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        final_channels = self.feature_channels["layer4"]
        if feature_dim is None:
            self.map_model.fc = nn.Identity()
        else:
            self.map_model.fc = nn.Linear(final_channels, int(feature_dim))
        self.output_activation = nn.Identity() if output_activation is None else output_activation()

    @property
    def feature_channels(self) -> Dict[str, int]:
        if self.model_arch in {"resnet18", "resnet34"}:
            return {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}
        return {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}

    @property
    def feature_scales(self) -> Dict[str, float]:
        return {"layer1": 1 / 4, "layer2": 1 / 8, "layer3": 1 / 16, "layer4": 1 / 32}

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.map_model(image)
        return self.output_activation(features)

    def extract_features(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        # This helper is only for tests/debugging; production UNet/ROI encoders
        # still use torchvision create_feature_extractor to keep state_dict names
        # close to the official code.
        x = self.map_model.conv1(image)
        x = self.map_model.bn1(x)
        x = self.map_model.relu(x)
        x = self.map_model.maxpool(x)
        layer1 = self.map_model.layer1(x)
        layer2 = self.map_model.layer2(layer1)
        layer3 = self.map_model.layer3(layer2)
        layer4 = self.map_model.layer4(layer3)
        final = self.map_model.avgpool(layer4)
        final = torch.flatten(final, 1)
        final = self.map_model.fc(final)
        return {
            "layer1": layer1,
            "layer2": layer2,
            "layer3": layer3,
            "layer4": layer4,
            "final": self.output_activation(final),
        }


class SharedRasterEncoder(nn.Module):
    """Shared BITS raster encoder used by planner and predictor heads."""

    def __init__(self, image_channels: int, model_arch: str = "resnet18", feature_dim: int = 128):
        super().__init__()
        encoder = BitsRasterBackbone(image_channels, model_arch=model_arch, feature_dim=feature_dim)
        self.encoder_heads = create_feature_extractor(
            encoder,
            {
                "map_model.layer1": "layer1",
                "map_model.layer2": "layer2",
                "map_model.layer3": "layer3",
                "map_model.layer4": "layer4",
                "map_model.fc": "final",
            },
        )
        self.feature_channels = encoder.feature_channels
        self.feature_scales = encoder.feature_scales

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.encoder_heads(image)


class UNetDecoder(nn.Module):
    """UNet decoder used by the high-level BITS spatial planner."""

    def __init__(self, encoder_channels: Dict[str, int], output_channels: int = 4):
        super().__init__()
        c1 = encoder_channels["layer1"]
        c2 = encoder_channels["layer2"]
        c3 = encoder_channels["layer3"]
        c4 = encoder_channels["layer4"]
        self.conv1 = nn.Sequential(
            nn.Conv2d(c4, 1024, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(True)
        )
        self.up1 = Upsample(1024 + c3, 512)
        self.up2 = Upsample(512 + c2, 256)
        self.up3 = Upsample(256 + c1, 128)
        self.layer1 = nn.Sequential(
            BottleneckBlock(128, (64, 64, 64), shortcut=True),
            BottleneckBlock(64, (64, 64, 64)),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.layer2 = nn.Sequential(
            BottleneckBlock(64, (32, 32, 32), shortcut=True),
            BottleneckBlock(32, (32, 32, 32)),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.layer3 = nn.Sequential(
            BottleneckBlock(32, (16, 16, 16), shortcut=True),
            BottleneckBlock(16, (16, 16, 16)),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.conv2 = nn.Sequential(nn.Conv2d(16, output_channels, kernel_size=1))

    def forward(self, encoder_features: Dict[str, torch.Tensor], target_hw: tuple) -> torch.Tensor:
        x = self.conv1(encoder_features["layer4"])
        x = self.up1(x, encoder_features["layer3"])
        x = self.up2(x, encoder_features["layer2"])
        x = self.up3(x, encoder_features["layer1"])
        for layer in (self.layer1, self.layer2, self.layer3, self.conv2):
            x = layer(x)
        return F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)


class GoalDecoder(nn.Module):
    """BITS high-level spatial goal decoder fed by shared raster features."""

    def __init__(self, encoder_channels: Dict[str, int], output_channels: int = 4):
        super().__init__()
        self.decoder = UNetDecoder(encoder_channels, output_channels)

    def forward(self, encoder_features: Dict[str, torch.Tensor], target_hw: tuple) -> torch.Tensor:
        return self.decoder(encoder_features, target_hw=target_hw)
