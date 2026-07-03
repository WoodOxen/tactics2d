# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Low-level PyTorch building blocks for BITS-style models.

Classes and functions here are BITS-specific (tuned for TBSIM checkpoint
compatibility) and have no dependency on :mod:`model` or :mod:`schema`.
"""

import copy
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
from torchvision.models.feature_extraction import create_feature_extractor

from .config import BitsConfig

# ---------------------------------------------------------------------------
# Torch batch container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BitsTorchBatch:
    """A torch-backed view of one or more BITS samples."""

    tensors: Dict[str, object]
    metadata: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {"tensors": self.tensors, "metadata": self.metadata}

    def to(self, device) -> "BitsTorchBatch":
        """Move all tensor values to another torch device."""

        return BitsTorchBatch(
            tensors={
                name: value.to(device) if hasattr(value, "to") else value
                for name, value in self.tensors.items()
            },
            metadata=self.metadata,
        )


# ---------------------------------------------------------------------------
# Unicycle dynamics helpers (TBSIM-compatible)
# ---------------------------------------------------------------------------


def integrate_unicycle_controls(
    controls: torch.Tensor, current_states: torch.Tensor, config: Optional[BitsConfig] = None
) -> tuple:
    """Integrate acceleration/yaw-rate controls with TBSIM-style unicycle dynamics."""

    resolved_config = config or BitsConfig(future_steps=controls.shape[-2])
    states = _ensure_unicycle_state_shape(current_states, controls)
    positions = []
    yaws = []
    for step in range(controls.shape[-2]):
        states = _unicycle_step(states, controls[..., step, :], resolved_config)
        positions.append(states[..., 0:2])
        yaws.append(states[..., 3:4])
    return torch.stack(positions, dim=-2), torch.stack(yaws, dim=-2)


def _unicycle_step(
    states: torch.Tensor, controls: torch.Tensor, config: BitsConfig
) -> torch.Tensor:
    # TBSIM Unicycle state is [x, y, speed, yaw], with controls
    # [acceleration, yaw_rate]. Clamp controls from current speed, then
    # integrate position with a half-step acceleration approximation.
    acceleration, yaw_rate = _clip_unicycle_controls(states, controls, config)
    dt = float(config.dt)
    speed = states[..., 2:3]
    yaw = states[..., 3:4]
    next_speed_for_position = speed + acceleration * dt * 0.5
    dx = torch.cos(yaw) * next_speed_for_position * dt
    dy = torch.sin(yaw) * next_speed_for_position * dt
    next_speed = speed + acceleration * dt
    next_yaw = yaw + yaw_rate * dt
    return torch.cat([states[..., 0:1] + dx, states[..., 1:2] + dy, next_speed, next_yaw], dim=-1)


def _clip_unicycle_controls(
    states: torch.Tensor, controls: torch.Tensor, config: BitsConfig
) -> tuple:
    speed = states[..., 2:3]
    speed_for_yaw = torch.clamp(torch.abs(speed), min=0.1)
    yaw_bound = torch.minimum(
        torch.as_tensor(config.dynamics_max_steer, dtype=speed.dtype, device=speed.device)
        * speed_for_yaw,
        torch.as_tensor(config.dynamics_max_yawvel, dtype=speed.dtype, device=speed.device)
        / speed_for_yaw,
    )
    yaw_bound = torch.clamp(yaw_bound, min=0.1)

    acceleration = controls[..., 0:1]
    yaw_rate = controls[..., 1:2]
    acceleration_lower = torch.clamp(
        torch.as_tensor(config.dynamics_speed_min, dtype=speed.dtype, device=speed.device) - speed,
        max=float(config.dynamics_acceleration_max),
    )
    acceleration_lower = torch.clamp(
        acceleration_lower, min=float(config.dynamics_acceleration_min)
    )
    acceleration_upper = torch.clamp(
        torch.as_tensor(config.dynamics_speed_max, dtype=speed.dtype, device=speed.device) - speed,
        min=float(config.dynamics_acceleration_min),
    )
    acceleration_upper = torch.clamp(
        acceleration_upper, max=float(config.dynamics_acceleration_max)
    )
    return (
        torch.clamp(acceleration, acceleration_lower, acceleration_upper),
        torch.clamp(yaw_rate, -yaw_bound, yaw_bound),
    )


def _ensure_unicycle_state_shape(
    current_states: torch.Tensor, controls: torch.Tensor
) -> torch.Tensor:
    states = current_states.to(device=controls.device, dtype=controls.dtype)
    while states.ndim < controls.ndim - 1:
        states = states.unsqueeze(1)
    return states.expand(*controls.shape[:-2], states.shape[-1])


# ---------------------------------------------------------------------------
# MLP building blocks
# ---------------------------------------------------------------------------


class BitsMLP(nn.Module):
    """MLP with the same public state-dict layout as TBSIM base_models.MLP."""

    def __init__(
        self, input_dim: int, output_dim: int, layer_dims: tuple = (), output_activation=None
    ):
        super().__init__()
        layers = []
        current_dim = int(input_dim)
        for layer_dim in layer_dims:
            layers.extend([nn.Linear(current_dim, int(layer_dim)), nn.ReLU(inplace=True)])
            current_dim = int(layer_dim)
        layers.append(nn.Linear(current_dim, int(output_dim)))
        if output_activation is not None:
            layers.append(output_activation())
        self._model = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._model(inputs)


class BitsSplitMLP(BitsMLP):
    """Multi-output MLP used by the official BITS trajectory decoder."""

    def __init__(self, input_dim: int, output_shapes: OrderedDict, layer_dims: tuple = ()):
        self._output_shapes = output_shapes
        output_dim = int(sum(np.prod(shape) for shape in output_shapes.values()))
        super().__init__(input_dim=input_dim, output_dim=output_dim, layer_dims=layer_dims)

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw_output = super().forward(inputs)
        outputs = {}
        start = 0
        for name, shape in self._output_shapes.items():
            width = int(np.prod(shape))
            outputs[name] = raw_output[..., start : start + width].reshape(
                *raw_output.shape[:-1], *shape
            )
            start += width
        return outputs


# ---------------------------------------------------------------------------
# Trajectory decoder (MLP + unicycle dynamics)
# ---------------------------------------------------------------------------


class BitsMLPTrajectoryDecoder(nn.Module):
    """Official-style MLP decoder that predicts unicycle controls then integrates them."""

    def __init__(
        self,
        feature_dim: int,
        future_steps: int,
        layer_dims: tuple = (128, 128, 128),
        config: Optional[BitsConfig] = None,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.future_steps = int(future_steps)
        self.config = config or BitsConfig(future_steps=future_steps)
        # Official Unicycle decoder uses state_as_input=True: concatenate the
        # current [x, y, v, yaw] state with features before predicting controls.
        self.mlp = BitsSplitMLP(
            input_dim=self.feature_dim + 4,
            output_shapes=OrderedDict(trajectories=(self.future_steps, 2)),
            layer_dims=layer_dims,
        )

    def forward(
        self, inputs: torch.Tensor, current_states: torch.Tensor, predict: bool = True
    ) -> Dict[str, torch.Tensor]:
        decoded = self.mlp(torch.cat([inputs, current_states.to(inputs)], dim=-1))
        controls = decoded["trajectories"]
        positions, yaws = integrate_unicycle_controls(controls, current_states, config=self.config)
        return {
            "controls": controls,
            "trajectories": torch.cat([positions, yaws], dim=-1),
            "positions": positions,
            "yaws": yaws,
        }


class BitsRNNTrajectoryEncoder(nn.Module):
    """Official-style RNNTrajectoryEncoder used by history_conditioning checkpoints."""

    def __init__(
        self,
        trajectory_dim: int,
        rnn_hidden_size: int,
        feature_dim: int,
        mlp_layer_dims: tuple = (128, 128),
    ):
        super().__init__()
        self.lstm = nn.LSTM(int(trajectory_dim), hidden_size=int(rnn_hidden_size), batch_first=True)
        self.mlp = BitsMLP(
            input_dim=int(rnn_hidden_size),
            output_dim=int(feature_dim),
            layer_dims=mlp_layer_dims,
            output_activation=nn.ReLU,
        )

    def forward(self, input_trajectory: torch.Tensor) -> torch.Tensor:
        trajectory_feature = self.lstm(input_trajectory)[0][:, -1, :]
        return self.mlp(trajectory_feature)


# ---------------------------------------------------------------------------
# Transformer components (TBSIM SimpleTransformer)
# ---------------------------------------------------------------------------


class _BitsLayerNorm(nn.Module):
    """LayerNorm with TBSIM-compatible parameter names."""

    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(-1, keepdim=True)
        std = inputs.std(-1, keepdim=True)
        return self.a_2 * (inputs - mean) / (std + self.eps) + self.b_2


class _BitsSublayerConnection(nn.Module):
    """Residual connection followed by TBSIM-style layer normalization."""

    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.norm = _BitsLayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, sublayer) -> torch.Tensor:
        return inputs + self.dropout(sublayer(self.norm(inputs)))


class _BitsPositionwiseFeedForward(nn.Module):
    """Feed-forward block with official TBSIM parameter names w_1/w_2."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(F.relu(self.w_1(inputs))))


class _BitsMultiHeadedAttention(nn.Module):
    """TBSIM SimpleTransformer attention block."""

    def __init__(self, head_count: int, d_model: int, dropout: float = 0.1, pooling_dim=None):
        super().__init__()
        if d_model % head_count != 0:
            raise ValueError("d_model must be divisible by head_count.")
        self.d_k = d_model // head_count
        self.h = head_count
        self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(4)])
        self.attn = None
        self.pooling_dim = pooling_dim
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pooling_dim = -2 if self.pooling_dim is None else self.pooling_dim
        if mask is not None:
            if mask.ndim == query.ndim - 1:
                mask = mask.view([*mask.shape, 1, 1]).transpose(-1, pooling_dim - 1)
            elif mask.ndim == query.ndim:
                mask = mask.unsqueeze(-2).transpose(-2, pooling_dim - 1)
            else:
                raise ValueError("mask dimension mismatch")
        query, key, value = (
            layer(inputs).view(*inputs.shape[:-1], self.h, self.d_k)
            for layer, inputs in zip(self.linears, (query, key, value))
        )
        attended, self.attn = _bits_scaled_dot_product_attention(
            query.transpose(-2, pooling_dim - 1),
            key.transpose(-2, pooling_dim - 1),
            value.transpose(-2, pooling_dim - 1),
            mask,
            dropout=self.dropout,
        )
        attended = attended.transpose(-2, pooling_dim - 1).contiguous()
        attended = attended.view(*attended.shape[:-2], self.h * self.d_k)
        return self.linears[-1](attended)


def _bits_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
) -> tuple:
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attention = F.softmax(scores, dim=-1)
    if dropout is not None:
        attention = dropout(attention)
    return torch.matmul(attention, value), attention


class _BitsEncoderLayer(nn.Module):
    """Single TBSIM StaticEncoder layer."""

    def __init__(
        self,
        size: int,
        self_attn: _BitsMultiHeadedAttention,
        feed_forward: _BitsPositionwiseFeedForward,
        dropout: float,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([_BitsSublayerConnection(size, dropout) for _ in range(2)])
        self.size = size

    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        inputs = self.sublayer[0](inputs, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](inputs, self.feed_forward)


class _BitsPositionalEncodingNd(nn.Module):
    """Official XY positional encoding for SimpleTransformer."""

    def __init__(self, dim: int, dropout: float, step_size=(0.1, 0.1)):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError("dim must be divisible by 4.")
        self.dropout = nn.Dropout(p=dropout)
        self.dim = int(dim)
        self.step_size = tuple(float(value) for value in step_size)
        axis_dim = dim // 2
        self.div_term = torch.exp(torch.arange(0, axis_dim, 2) * -(math.log(10000.0) / axis_dim))

    def forward(self, inputs: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        encoded = torch.zeros(
            *inputs.shape[:-1], self.dim, dtype=inputs.dtype, device=inputs.device
        )
        div_term = self.div_term.to(device=inputs.device, dtype=inputs.dtype)
        axis_dim = self.dim // 2
        for axis, step in enumerate(self.step_size):
            phase = position[..., axis : axis + 1] / step * div_term
            start = axis * axis_dim
            axis_encoded = encoded[..., start : start + axis_dim]
            axis_encoded[..., 0::2] = torch.sin(phase)
            axis_encoded[..., 1::2] = torch.cos(phase)
        return self.dropout(encoded)


class _BitsStaticEncoder(nn.Module):
    """Agent-axis transformer encoder with official state_dict names."""

    def __init__(
        self, agent_enc: _BitsEncoderLayer, xy_pe: _BitsPositionalEncodingNd, layer_count: int = 1
    ):
        super().__init__()
        self.N_layer = int(layer_count)
        self.agent_encs = nn.ModuleList([copy.deepcopy(agent_enc) for _ in range(self.N_layer)])
        self.XY_pe = xy_pe

    def forward(
        self,
        inputs: torch.Tensor,
        source_mask: torch.Tensor,
        source_position: torch.Tensor,
        map_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pieces = [inputs, self.XY_pe(inputs, source_position)]
        if map_embedding is not None:
            pieces.append(map_embedding)
        encoded = torch.cat(pieces, dim=-1) * source_mask.unsqueeze(-1)
        for layer in self.agent_encs:
            encoded = layer(encoded, source_mask)
        return encoded


class BitsSimpleTransformer(nn.Module):
    """Official SimpleTransformer used by the released BITS predictor checkpoint."""

    def __init__(
        self,
        src_dim: int,
        N_a: int = 3,
        d_model: int = 384,
        XY_pe_dim: int = 64,
        d_ff: int = 2048,
        head: int = 8,
        dropout: float = 0.1,
        step_size=(0.1, 0.1),
    ):
        super().__init__()
        agent_attn = _BitsMultiHeadedAttention(head, d_model, pooling_dim=-3)
        feed_forward = _BitsPositionwiseFeedForward(d_model, d_ff, dropout)
        xy_pe = _BitsPositionalEncodingNd(XY_pe_dim, dropout, step_size=step_size)
        self.agent_enc = _BitsStaticEncoder(
            _BitsEncoderLayer(
                d_model, copy.deepcopy(agent_attn), copy.deepcopy(feed_forward), dropout
            ),
            xy_pe,
            N_a,
        )
        self.pre_emb = nn.Linear(src_dim, d_model - XY_pe_dim)
        self.post_emb = nn.Linear(d_model, src_dim)

    def forward(
        self, features: torch.Tensor, availability: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        embedded = self.pre_emb(features)
        encoded = self.agent_enc(embedded, availability, positions)
        return self.post_emb(encoded)


# ---------------------------------------------------------------------------
# UNet decoder components
# ---------------------------------------------------------------------------


class _BitsConvBlock(nn.Module):
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


class _BitsUp(nn.Module):
    """Official-style bilinear upsample plus double conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = _BitsConvBlock(in_channels, out_channels, in_channels // 2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class _BitsBottleneckBlock(nn.Module):
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


class _BitsUNetDecoder(nn.Module):
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
        self.up1 = _BitsUp(1024 + c3, 512)
        self.up2 = _BitsUp(512 + c2, 256)
        self.up3 = _BitsUp(256 + c1, 128)
        self.layer1 = nn.Sequential(
            _BitsBottleneckBlock(128, (64, 64, 64), shortcut=True),
            _BitsBottleneckBlock(64, (64, 64, 64)),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.layer2 = nn.Sequential(
            _BitsBottleneckBlock(64, (32, 32, 32), shortcut=True),
            _BitsBottleneckBlock(32, (32, 32, 32)),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )
        self.layer3 = nn.Sequential(
            _BitsBottleneckBlock(32, (16, 16, 16), shortcut=True),
            _BitsBottleneckBlock(16, (16, 16, 16)),
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


# ---------------------------------------------------------------------------
# Raster backbone and shared encoder
# ---------------------------------------------------------------------------


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


class SpatialGoalUNetDecoder(nn.Module):
    """BITS high-level spatial goal decoder fed by shared raster features."""

    def __init__(self, encoder_channels: Dict[str, int], output_channels: int = 4):
        super().__init__()
        self.decoder = _BitsUNetDecoder(encoder_channels, output_channels)

    def forward(self, encoder_features: Dict[str, torch.Tensor], target_hw: tuple) -> torch.Tensor:
        return self.decoder(encoder_features, target_hw=target_hw)


class BitsRasterizedMapUNet(nn.Module):
    """BITS SpatialPlanner network: raster backbone plus UNet goal-map decoder."""

    def __init__(self, image_channels: int, model_arch: str = "resnet18", output_channels: int = 4):
        super().__init__()
        encoder = BitsRasterBackbone(image_channels, model_arch=model_arch)
        self.encoder_heads = create_feature_extractor(
            encoder,
            {
                "map_model.layer1": "layer1",
                "map_model.layer2": "layer2",
                "map_model.layer3": "layer3",
                "map_model.layer4": "layer4",
            },
        )
        self.decoder = _BitsUNetDecoder(encoder.feature_channels, output_channels)

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        encoder_features: Optional[Dict[str, torch.Tensor]] = None,
        target_hw: Optional[tuple] = None,
    ) -> torch.Tensor:
        if encoder_features is None:
            if image is None:
                raise ValueError("image is required when encoder_features is not provided.")
            encoder_features = self.encoder_heads(image)
        if target_hw is None:
            if image is None:
                layer1 = encoder_features["layer1"]
                target_hw = (layer1.shape[-2] * 4, layer1.shape[-1] * 4)
            else:
                target_hw = image.shape[-2:]
        return self.decoder(encoder_features, target_hw=target_hw)


__all__ = [
    "BitsMLP",
    "BitsMLPTrajectoryDecoder",
    "BitsRasterBackbone",
    "BitsRasterizedMapUNet",
    "BitsRNNTrajectoryEncoder",
    "BitsSimpleTransformer",
    "BitsSplitMLP",
    "BitsTorchBatch",
    "SharedRasterEncoder",
    "SpatialGoalUNetDecoder",
    "integrate_unicycle_controls",
]
