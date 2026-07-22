# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""ROI alignment and rasterized-map encoder."""

from typing import Dict, Optional

import torch
import torch.nn as nn
from torchvision.ops import RoIAlign

from .unet import BitsRasterBackbone


def _add_batch_dim(tensor, min_ndim: int):
    """Add a leading batch dimension if tensor has fewer than min_ndim dims."""
    while tensor.ndim < min_ndim:
        tensor = tensor.unsqueeze(0)
    return tensor


def _homogeneous_transform(points: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Apply a 3x3 homogeneous transform to batched 2D points."""
    ones = torch.ones(*points.shape[:-1], 1, dtype=points.dtype, device=points.device)
    homogeneous = torch.cat([points, ones], dim=-1)
    transform = matrix.to(device=points.device, dtype=points.dtype)
    return torch.matmul(homogeneous, transform.transpose(1, 2))[..., :2]


def _build_upright_rois(raster_points: torch.Tensor, context_size: int) -> torch.Tensor:
    """Build official-style [batch_index, x1, y1, x2, y2] ROI boxes."""
    batch_size, agent_count = raster_points.shape[:2]
    half = float(context_size) / 2.0
    x_center = raster_points[..., 0]
    y_center = raster_points[..., 1]
    boxes = torch.stack(
        [x_center - half, y_center - half, x_center + half, y_center + half], dim=-1
    ).reshape(batch_size * agent_count, 4)
    batch_indices = (
        torch.arange(batch_size, dtype=raster_points.dtype, device=raster_points.device)
        .unsqueeze(1)
        .expand(-1, agent_count)
        .reshape(-1, 1)
    )
    return torch.cat([batch_indices, boxes], dim=1)


class ROIHead(nn.Module):
    """Agent ROI feature head that consumes already-shared raster features."""

    def __init__(
        self,
        global_feature_dim: int = 128,
        agent_feature_dim: int = 128,
        context_size: int = 30,
        roi_feature_size: int = 7,
        roi_layer_key: str = "layer2",
        feature_channels: Optional[Dict[str, int]] = None,
        feature_scales: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.global_feature_dim = int(global_feature_dim)
        self.agent_feature_dim = int(agent_feature_dim)
        self.context_size = int(context_size)
        self.roi_feature_size = int(roi_feature_size)
        self.roi_layer_key = roi_layer_key
        if feature_channels is None:
            feature_channels = BitsRasterBackbone(3).feature_channels
        if feature_scales is None:
            feature_scales = BitsRasterBackbone(3).feature_scales
        self.feature_channels = dict(feature_channels)
        self.feature_scales = dict(feature_scales)
        if roi_layer_key not in self.feature_channels:
            raise ValueError("roi_layer_key must be one of layer1, layer2, layer3, or layer4.")
        roi_channels = self.feature_channels[roi_layer_key]
        self.roi_align = RoIAlign(
            output_size=(self.roi_feature_size, self.roi_feature_size),
            spatial_scale=self.feature_scales[roi_layer_key],
            sampling_ratio=-1,
            aligned=True,
        )
        self.activation = nn.ReLU()
        self.agent_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(roi_channels, self.agent_feature_dim),
            self.activation,
        )

    def forward(
        self,
        tensors: Dict[str, torch.Tensor],
        agent_positions: torch.Tensor,
        encoder_features: Dict[str, torch.Tensor],
    ) -> tuple:
        image = _add_batch_dim(tensors["image"], 4)
        global_features = self.activation(encoder_features["final"])
        raster_from_agent = _add_batch_dim(tensors["raster_from_agent"], 3).to(
            device=image.device, dtype=image.dtype
        )
        raster_points = _homogeneous_transform(
            agent_positions.to(device=image.device, dtype=image.dtype), raster_from_agent
        )
        # Official BITS defaults to use_rotated_roi=False: build an axis-aligned
        # ROI around each agent raster position, then apply torchvision RoIAlign.
        rois = _build_upright_rois(raster_points, context_size=self.context_size)
        roi_features = self.roi_align(encoder_features[self.roi_layer_key], rois)
        batch_size, agent_count = raster_points.shape[:2]
        agent_features = self.agent_net(roi_features).reshape(
            batch_size, agent_count, self.agent_feature_dim
        )
        return agent_features, global_features
