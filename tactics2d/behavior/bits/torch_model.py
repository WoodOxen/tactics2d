# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""PyTorch model utilities for BITS-style imitation."""

import copy
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import RoIAlign

from .config import BitsConfig
from .model import BitsAgentPrediction, BitsPlan, BitsPlanScorer, BitsPolicy, BitsPrediction
from .schema import BitsBatch
from .supervision import BitsGoalSupervision, build_goal_supervision


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


@dataclass(frozen=True)
class BitsTorchEpochResult:
    """Summary from one BITS torch training or validation epoch."""

    sample_count: int
    step_count: int
    mean_total_loss: float
    mean_losses: Dict[str, float]

    def as_dict(self) -> Dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "step_count": self.step_count,
            "mean_total_loss": self.mean_total_loss,
            "mean_losses": self.mean_losses,
        }


_FLOAT_BATCH_KEYS = (
    "history_positions",
    "history_yaws",
    "target_positions",
    "target_yaws",
    "curr_speed",
    "centroid",
    "yaw",
    "extent",
    "agent_from_world",
    "world_from_agent",
    "all_other_agents_history_positions",
    "all_other_agents_history_yaws",
    "all_other_agents_future_positions",
    "all_other_agents_future_yaws",
    "all_other_agents_curr_speed",
    "all_other_agents_extents",
    "all_other_agents_history_extents",
)

_BOOL_BATCH_KEYS = (
    "history_availabilities",
    "target_availabilities",
    "all_other_agents_history_availability",
    "all_other_agents_future_availability",
)

_INT_BATCH_KEYS = ("type", "all_other_agents_types")

_OPTIONAL_FLOAT_BATCH_KEYS = (
    "image",
    "raster_from_agent",
    "agent_from_raster",
    "static_image",
    "dynamic_image",
)

_OPTIONAL_BOOL_BATCH_KEYS = ("drivable_map",)


def bits_batch_to_torch(
    batch: BitsBatch,
    device=None,
    dtype=None,
    include_optional: bool = True,
) -> BitsTorchBatch:
    """Convert one BITS batch to tensors for BITS PyTorch models."""

    resolved_dtype = dtype or torch.float32
    tensors = {}

    # Boundary between Tactics2D scene data and BITS torch modules:
    # batches describe scenarios, while modules operate on tensors.
    for key in _FLOAT_BATCH_KEYS:
        tensors[key] = _as_tensor(getattr(batch, key), resolved_dtype, device)
    for key in _BOOL_BATCH_KEYS:
        tensors[key] = _as_tensor(getattr(batch, key), torch.bool, device)
    for key in _INT_BATCH_KEYS:
        tensors[key] = _as_tensor(getattr(batch, key), torch.long, device)

    if include_optional:
        for key in _OPTIONAL_FLOAT_BATCH_KEYS:
            value = getattr(batch, key)
            if value is not None:
                tensors[key] = _as_tensor(value, resolved_dtype, device)
        for key in _OPTIONAL_BOOL_BATCH_KEYS:
            value = getattr(batch, key)
            if value is not None:
                tensors[key] = _as_tensor(value, torch.bool, device)

    return BitsTorchBatch(
        tensors=tensors,
        metadata={
            "ego_id": batch.ego_id,
            "frame": batch.frame,
            "agent_ids": list(batch.agent_ids),
            "lane_id": batch.lane_id,
        },
    )


def collate_bits_batches_to_torch(
    batches: Iterable[BitsBatch],
    device=None,
    dtype=None,
    include_optional: bool = True,
) -> BitsTorchBatch:
    """Stack same-shaped BITS batches into a torch mini-batch."""

    samples = [
        bits_batch_to_torch(
            batch,
            device=device,
            dtype=dtype,
            include_optional=include_optional,
        )
        for batch in batches
    ]
    if not samples:
        return BitsTorchBatch(
            tensors={},
            metadata={"ego_id": [], "frame": [], "agent_ids": [], "lane_id": []},
        )

    keys = set(samples[0].tensors.keys())
    for sample in samples[1:]:
        if set(sample.tensors.keys()) != keys:
            raise ValueError("Cannot collate BITS samples with different tensor fields.")

    return BitsTorchBatch(
        tensors={
            key: torch.stack([sample.tensors[key] for sample in samples], dim=0)
            for key in sorted(keys)
        },
        metadata={
            "ego_id": [sample.metadata["ego_id"] for sample in samples],
            "frame": [sample.metadata["frame"] for sample in samples],
            "agent_ids": [sample.metadata["agent_ids"] for sample in samples],
            "lane_id": [sample.metadata["lane_id"] for sample in samples],
        },
    )


def collate_bits_goal_supervisions_to_torch(
    goals: Iterable[BitsGoalSupervision],
    device=None,
    dtype=None,
) -> Dict[str, object]:
    """Stack spatial planner supervision for a torch mini-batch."""

    goal_tensors = [
        bits_goal_supervision_to_torch(goal, device=device, dtype=dtype) for goal in goals
    ]
    if not goal_tensors:
        return {}

    keys = set(goal_tensors[0].keys())
    for goal in goal_tensors[1:]:
        if set(goal.keys()) != keys:
            raise ValueError("Cannot collate goal supervision with different fields.")

    return {key: torch.stack([goal[key] for goal in goal_tensors], dim=0) for key in keys}


def bits_goal_supervision_to_torch(
    goal: BitsGoalSupervision,
    device=None,
    dtype=None,
) -> Dict[str, object]:
    """Convert BITS spatial planner supervision to torch tensors."""

    resolved_dtype = dtype or torch.float32
    tensors = {
        "goal_position": _as_tensor(goal.goal_position, resolved_dtype, device),
        "goal_yaw": _as_tensor(goal.goal_yaw, resolved_dtype, device),
        "goal_index": _as_tensor(goal.goal_index, torch.long, device),
    }
    if goal.goal_position_pixel is not None:
        tensors["goal_position_pixel"] = _as_tensor(
            goal.goal_position_pixel, torch.long, device
        )
    if goal.goal_position_pixel_flat is not None:
        tensors["goal_position_pixel_flat"] = _as_tensor(
            goal.goal_position_pixel_flat, torch.long, device
        )
    if goal.goal_position_residual is not None:
        tensors["goal_position_residual"] = _as_tensor(
            goal.goal_position_residual, resolved_dtype, device
        )
    if goal.goal_spatial_map is not None:
        tensors["goal_spatial_map"] = _as_tensor(
            goal.goal_spatial_map, resolved_dtype, device
        )
    return tensors


def integrate_unicycle_controls(
    controls: torch.Tensor,
    current_states: torch.Tensor,
    config: Optional[BitsConfig] = None,
) -> tuple:
    """Integrate acceleration/yaw-rate controls with TBSIM-style unicycle dynamics."""

    resolved_config = config or BitsConfig(future_steps=controls.shape[-2])
    states = _ensure_unicycle_state_shape(current_states, controls)
    positions = []
    yaws = []
    for step in range(controls.shape[-2]):
        states = _unicycle_step(
            states,
            controls[..., step, :],
            resolved_config,
        )
        positions.append(states[..., 0:2])
        yaws.append(states[..., 3:4])
    return torch.stack(positions, dim=-2), torch.stack(yaws, dim=-2)


def _unicycle_step(
    states: torch.Tensor,
    controls: torch.Tensor,
    config: BitsConfig,
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
    return torch.cat(
        [states[..., 0:1] + dx, states[..., 1:2] + dy, next_speed, next_yaw],
        dim=-1,
    )


def _clip_unicycle_controls(
    states: torch.Tensor,
    controls: torch.Tensor,
    config: BitsConfig,
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
        acceleration_lower,
        min=float(config.dynamics_acceleration_min),
    )
    acceleration_upper = torch.clamp(
        torch.as_tensor(config.dynamics_speed_max, dtype=speed.dtype, device=speed.device) - speed,
        min=float(config.dynamics_acceleration_min),
    )
    acceleration_upper = torch.clamp(
        acceleration_upper,
        max=float(config.dynamics_acceleration_max),
    )
    return (
        torch.clamp(acceleration, acceleration_lower, acceleration_upper),
        torch.clamp(yaw_rate, -yaw_bound, yaw_bound),
    )


def _ensure_unicycle_state_shape(
    current_states: torch.Tensor,
    controls: torch.Tensor,
) -> torch.Tensor:
    states = current_states.to(device=controls.device, dtype=controls.dtype)
    while states.ndim < controls.ndim - 1:
        states = states.unsqueeze(1)
    return states.expand(*controls.shape[:-2], states.shape[-1])


class BitsMLP(nn.Module):
    """MLP with the same public state-dict layout as TBSIM base_models.MLP."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layer_dims: tuple = (),
        output_activation=None,
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

    def __init__(
        self,
        input_dim: int,
        output_shapes: OrderedDict,
        layer_dims: tuple = (),
    ):
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
                *raw_output.shape[:-1],
                *shape,
            )
            start += width
        return outputs


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
        self,
        inputs: torch.Tensor,
        current_states: torch.Tensor,
        predict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        decoded = self.mlp(torch.cat([inputs, current_states.to(inputs)], dim=-1))
        controls = decoded["trajectories"]
        positions, yaws = integrate_unicycle_controls(
            controls,
            current_states,
            config=self.config,
        )
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
        self.lstm = nn.LSTM(
            int(trajectory_dim),
            hidden_size=int(rnn_hidden_size),
            batch_first=True,
        )
        self.mlp = BitsMLP(
            input_dim=int(rnn_hidden_size),
            output_dim=int(feature_dim),
            layer_dims=mlp_layer_dims,
            output_activation=nn.ReLU,
        )

    def forward(self, input_trajectory: torch.Tensor) -> torch.Tensor:
        trajectory_feature = self.lstm(input_trajectory)[0][:, -1, :]
        return self.mlp(trajectory_feature)


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
        self.linears = nn.ModuleList(
            [copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(4)]
        )
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
        query, key, value = [
            layer(inputs).view(*inputs.shape[:-1], self.h, self.d_k)
            for layer, inputs in zip(self.linears, (query, key, value))
        ]
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
        self.sublayer = nn.ModuleList(
            [_BitsSublayerConnection(size, dropout) for _ in range(2)]
        )
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
        self.div_term = torch.exp(
            torch.arange(0, axis_dim, 2) * -(math.log(10000.0) / axis_dim)
        )

    def forward(self, inputs: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        encoded = torch.zeros(*inputs.shape[:-1], self.dim, dtype=inputs.dtype, device=inputs.device)
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
        self,
        agent_enc: _BitsEncoderLayer,
        xy_pe: _BitsPositionalEncodingNd,
        layer_count: int = 1,
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
            _BitsEncoderLayer(d_model, copy.deepcopy(agent_attn), copy.deepcopy(feed_forward), dropout),
            xy_pe,
            N_a,
        )
        self.pre_emb = nn.Linear(src_dim, d_model - XY_pe_dim)
        self.post_emb = nn.Linear(d_model, src_dim)

    def forward(
        self,
        features: torch.Tensor,
        availability: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.pre_emb(features)
        encoded = self.agent_enc(embedded, availability, positions)
        return self.post_emb(encoded)


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
        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
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

    def __init__(
        self,
        image_channels: int,
        model_arch: str = "resnet18",
        feature_dim: int = 128,
    ):
        super().__init__()
        encoder = BitsRasterBackbone(
            image_channels,
            model_arch=model_arch,
            feature_dim=feature_dim,
        )
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


class _BitsUNetDecoder(nn.Module):
    """UNet decoder used by the high-level BITS spatial planner."""

    def __init__(self, encoder_channels: Dict[str, int], output_channels: int = 4):
        super().__init__()
        c1 = encoder_channels["layer1"]
        c2 = encoder_channels["layer2"]
        c3 = encoder_channels["layer3"]
        c4 = encoder_channels["layer4"]
        self.conv1 = nn.Sequential(
            nn.Conv2d(c4, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
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


class SpatialGoalUNetDecoder(nn.Module):
    """BITS high-level spatial goal decoder fed by shared raster features."""

    def __init__(self, encoder_channels: Dict[str, int], output_channels: int = 4):
        super().__init__()
        self.decoder = _BitsUNetDecoder(encoder_channels, output_channels)

    def forward(self, encoder_features: Dict[str, torch.Tensor], target_hw: tuple) -> torch.Tensor:
        return self.decoder(encoder_features, target_hw=target_hw)


class BitsRasterizedMapUNet(nn.Module):
    """BITS SpatialPlanner network: raster backbone plus UNet goal-map decoder."""

    def __init__(
        self,
        image_channels: int,
        model_arch: str = "resnet18",
        output_channels: int = 4,
    ):
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


class BitsRasterizeROIHead(nn.Module):
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
        image = _ensure_image_batch(tensors["image"])
        global_features = self.activation(encoder_features["final"])
        raster_from_agent = _ensure_matrix_batch(tensors["raster_from_agent"]).to(
            device=image.device,
            dtype=image.dtype,
        )
        raster_points = _transform_agent_points(
            agent_positions.to(device=image.device, dtype=image.dtype),
            raster_from_agent,
        )
        # Official BITS defaults to use_rotated_roi=False: build an axis-aligned
        # ROI around each agent raster position, then apply torchvision RoIAlign.
        rois = _build_indexed_upright_rois(
            raster_points,
            context_size=self.context_size,
        )
        roi_features = self.roi_align(encoder_features[self.roi_layer_key], rois)
        batch_size, agent_count = raster_points.shape[:2]
        agent_features = self.agent_net(roi_features).reshape(
            batch_size,
            agent_count,
            self.agent_feature_dim,
        )
        return agent_features, global_features


class BitsRasterizeROIEncoder(nn.Module):
    """BITS traffic map encoder with global feature and per-agent ROI feature."""

    def __init__(
        self,
        image_channels: int,
        global_feature_dim: int = 128,
        agent_feature_dim: int = 128,
        context_size: int = 30,
        roi_feature_size: int = 7,
        roi_layer_key: str = "layer2",
        model_arch: str = "resnet18",
    ):
        super().__init__()
        self.global_feature_dim = int(global_feature_dim)
        self.agent_feature_dim = int(agent_feature_dim)
        self.context_size = int(context_size)
        self.roi_feature_size = int(roi_feature_size)
        self.roi_layer_key = roi_layer_key
        encoder = BitsRasterBackbone(
            image_channels,
            model_arch=model_arch,
            feature_dim=global_feature_dim,
        )
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
        if roi_layer_key not in self.feature_channels:
            raise ValueError("roi_layer_key must be one of layer1, layer2, layer3, or layer4.")
        roi_channels = self.feature_channels[roi_layer_key]
        self.roi_align = RoIAlign(
            output_size=(self.roi_feature_size, self.roi_feature_size),
            spatial_scale=encoder.feature_scales[roi_layer_key],
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
        encoder_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> tuple:
        image = _ensure_image_batch(tensors["image"])
        if encoder_features is None:
            encoder_features = self.encoder_heads(image)
        global_features = self.activation(encoder_features["final"])
        raster_from_agent = _ensure_matrix_batch(tensors["raster_from_agent"]).to(
            device=image.device,
            dtype=image.dtype,
        )
        raster_points = _transform_agent_points(
            agent_positions.to(device=image.device, dtype=image.dtype),
            raster_from_agent,
        )
        # Official BITS defaults to use_rotated_roi=False: build an axis-aligned
        # ROI box around each agent raster position, then apply RoIAlign.
        rois = _build_indexed_upright_rois(
            raster_points,
            context_size=self.context_size,
        )
        roi_features = self.roi_align(encoder_features[self.roi_layer_key], rois)
        batch_size, agent_count = raster_points.shape[:2]
        agent_features = self.agent_net(roi_features).reshape(
            batch_size,
            agent_count,
            self.agent_feature_dim,
        )
        return agent_features, global_features, encoder_features


class GoalConditionalPolicyHead(nn.Module):
    """BITS ego policy head conditioned on high-level goal samples."""

    def __init__(
        self,
        agent_feature_dim_total: int,
        goal_feature_dim: int,
        future_steps: int,
        decoder_layer_dims: tuple = (128, 128, 128),
        config: Optional[BitsConfig] = None,
    ):
        super().__init__()
        self.goal_encoder = BitsMLP(3, goal_feature_dim, output_activation=nn.ReLU)
        self.ego_decoder = BitsMLPTrajectoryDecoder(
            feature_dim=agent_feature_dim_total + goal_feature_dim,
            future_steps=future_steps,
            layer_dims=decoder_layer_dims,
            config=config,
        )

    def forward(
        self,
        ego_features: torch.Tensor,
        ego_states: torch.Tensor,
        goal_positions: torch.Tensor,
        goal_yaws: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size = ego_features.shape[0]
        mode_count = goal_positions.shape[1]
        goal_states = torch.cat([goal_positions, goal_yaws], dim=-1)
        goal_features = self.goal_encoder(goal_states.reshape(batch_size * mode_count, 3))
        goal_features = goal_features.reshape(batch_size, mode_count, -1)
        ego_features = ego_features[:, None].expand(-1, mode_count, -1)
        ego_states = ego_states[:, None].expand(-1, mode_count, -1)
        ego_inputs = torch.cat([ego_features, goal_features], dim=-1)
        decoded = self.ego_decoder(
            ego_inputs.reshape(batch_size * mode_count, -1),
            ego_states.reshape(batch_size * mode_count, -1),
        )
        return {
            key: value.reshape(batch_size, mode_count, *value.shape[1:])
            for key, value in decoded.items()
        }


class FutureStatePredictorHead(nn.Module):
    """BITS neighboring-agent future-state predictor head."""

    def __init__(
        self,
        agent_feature_dim_total: int,
        future_steps: int,
        decoder_layer_dims: tuple = (128, 128, 128),
        config: Optional[BitsConfig] = None,
    ):
        super().__init__()
        self.agents_decoder = BitsMLPTrajectoryDecoder(
            feature_dim=agent_feature_dim_total,
            future_steps=future_steps,
            layer_dims=decoder_layer_dims,
            config=config,
        )

    def forward(
        self,
        other_features: torch.Tensor,
        other_states: torch.Tensor,
        future_steps: int,
    ) -> Dict[str, torch.Tensor]:
        batch_size, other_count = other_features.shape[:2]
        if other_count == 0:
            return {
                "controls": other_features.new_zeros(batch_size, 0, future_steps, 2),
                "positions": other_features.new_zeros(batch_size, 0, future_steps, 2),
                "yaws": other_features.new_zeros(batch_size, 0, future_steps, 1),
            }
        decoded = self.agents_decoder(
            other_features.reshape(batch_size * other_count, -1),
            other_states.reshape(batch_size * other_count, -1),
        )
        return {
            key: value.reshape(batch_size, other_count, *value.shape[1:])
            for key, value in decoded.items()
        }


class BitsSpatialPlannerModule(nn.Module):
    """High-level BITS spatial planner: shared features -> 4-channel goal map."""

    def __init__(
        self,
        image_channels: Optional[int] = None,
        model_arch: str = "resnet18",
        encoder_channels: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.standalone_encoder = None
        if encoder_channels is None:
            if image_channels is None:
                raise ValueError("image_channels is required without encoder_channels.")
            encoder = BitsRasterBackbone(image_channels, model_arch=model_arch)
            self.standalone_encoder = create_feature_extractor(
                encoder,
                {
                    "map_model.layer1": "layer1",
                    "map_model.layer2": "layer2",
                    "map_model.layer3": "layer3",
                    "map_model.layer4": "layer4",
                },
            )
            encoder_channels = encoder.feature_channels
        self.spatial_goal_decoder = SpatialGoalUNetDecoder(encoder_channels, output_channels=4)

    def forward(
        self,
        tensors: Dict[str, torch.Tensor],
        num_samples: Optional[int] = None,
        mask_drivable: bool = False,
        encoder_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        image = _ensure_image_batch(tensors["image"])
        if encoder_features is None:
            if self.standalone_encoder is None:
                raise ValueError("encoder_features are required for the shared-encoder planner.")
            encoder_features = self.standalone_encoder(image)
        spatial_prediction = self.spatial_goal_decoder(
            encoder_features,
            target_hw=image.shape[-2:],
        )
        drivable_map = _resolve_drivable_map(tensors) if mask_drivable else None
        decoded = decode_bits_spatial_prediction(
            spatial_prediction=spatial_prediction,
            agent_from_raster=_ensure_matrix_batch(tensors["agent_from_raster"]),
            drivable_map=drivable_map,
            num_samples=num_samples,
            mask_drivable=mask_drivable,
        )
        return {
            "spatial_prediction": spatial_prediction,
            "location_logits": spatial_prediction[:, 0],
            "location_prob_map": decoded["location_prob_map"],
            "positions": decoded["positions"],
            "yaws": decoded["yaws"],
            "scores": decoded["scores"],
            "log_likelihood": torch.log(decoded["scores"].clamp_min(1e-12)),
            "availabilities": torch.ones_like(decoded["scores"], dtype=torch.bool),
        }


class BitsAgentAwareTrajectoryModule(nn.Module):
    """Low-level BITS traffic model: ROI map features + goal-conditioned dynamics decoder."""

    def __init__(
        self,
        future_steps: int,
        image_channels: int,
        global_feature_dim: int = 128,
        agent_feature_dim: int = 128,
        goal_feature_dim: int = 32,
        decoder_layer_dims: tuple = (128, 128, 128),
        context_size: int = 30,
        roi_feature_size: int = 7,
        roi_layer_key: str = "layer2",
        model_arch: str = "resnet18",
        history_conditioning: bool = False,
        use_transformer: bool = False,
        config: Optional[BitsConfig] = None,
        shared_encoder: Optional[SharedRasterEncoder] = None,
    ):
        super().__init__()
        self.future_steps = int(future_steps)
        self.config = config or BitsConfig()
        self.goal_feature_dim = int(goal_feature_dim)
        self.history_conditioning = bool(history_conditioning)
        self.register_buffer(
            "roi_size",
            torch.tensor(
                [float(context_size), float(context_size), float(context_size), float(context_size)]
            ),
        )
        self.register_buffer("weights_scaling", torch.ones(3))
        if shared_encoder is None:
            self.shared_encoder = SharedRasterEncoder(
                image_channels=image_channels,
                model_arch=model_arch,
                feature_dim=global_feature_dim,
            )
        else:
            object.__setattr__(self, "shared_encoder", shared_encoder)
        self.roi_head = BitsRasterizeROIHead(
            global_feature_dim=global_feature_dim,
            agent_feature_dim=agent_feature_dim,
            context_size=context_size,
            roi_feature_size=roi_feature_size,
            roi_layer_key=roi_layer_key,
            feature_channels=self.shared_encoder.feature_channels,
            feature_scales=self.shared_encoder.feature_scales,
        )
        # Official AgentAwareRasterizedModel encodes the high-level goal only for
        # the ego decoder. Neighbor predictions are copied across sampled modes.
        history_feature_dim = 16 if self.history_conditioning else 0
        if self.history_conditioning:
            self.history_encoder = BitsRNNTrajectoryEncoder(
                trajectory_dim=3,
                rnn_hidden_size=100,
                mlp_layer_dims=(128, 128),
                feature_dim=history_feature_dim,
            )
        else:
            self.history_encoder = None
        agent_feature_dim_total = agent_feature_dim + global_feature_dim + history_feature_dim
        self.transformer = (
            BitsSimpleTransformer(src_dim=agent_feature_dim_total)
            if use_transformer
            else None
        )
        self.policy_head = GoalConditionalPolicyHead(
            agent_feature_dim_total=agent_feature_dim_total,
            goal_feature_dim=self.goal_feature_dim,
            future_steps=self.future_steps,
            decoder_layer_dims=decoder_layer_dims,
            config=self.config,
        )
        self.future_state_head = FutureStatePredictorHead(
            agent_feature_dim_total=agent_feature_dim_total,
            future_steps=self.future_steps,
            decoder_layer_dims=decoder_layer_dims,
            config=self.config,
        )

    def forward(
        self,
        tensors: Dict[str, torch.Tensor],
        goal_positions: torch.Tensor,
        goal_yaws: torch.Tensor,
        feature_context: Optional[tuple] = None,
    ) -> Dict[str, torch.Tensor]:
        if feature_context is None:
            all_features, current_states, current_availability, _encoder_features = (
                self.extract_features(tensors, return_encoder_features=True)
            )
        else:
            all_features, current_states, current_availability, _encoder_features = feature_context
        batch_size, agent_count = all_features.shape[:2]
        goal_positions = _ensure_goal_batch(goal_positions)
        goal_yaws = _ensure_goal_batch(goal_yaws)
        mode_count = goal_positions.shape[1]

        ego_features = all_features[:, 0]
        ego_states = current_states[:, 0]
        ego_decoded = self.policy_head(
            ego_features,
            ego_states,
            goal_positions,
            goal_yaws,
        )
        ego_controls = ego_decoded["controls"]
        ego_positions = ego_decoded["positions"]
        ego_yaws = ego_decoded["yaws"]

        other_count = max(agent_count - 1, 0)
        other_features = all_features[:, 1:]
        other_states = current_states[:, 1:]
        agent_decoded = self.future_state_head(
            other_features,
            other_states,
            future_steps=self.future_steps,
        )
        agent_controls = agent_decoded["controls"]
        agent_positions = agent_decoded["positions"]
        agent_yaws = agent_decoded["yaws"]
        agent_mask = current_availability[:, 1:].to(agent_controls)
        agent_controls = agent_controls * agent_mask[:, :, None, None]
        agent_positions = agent_positions * agent_mask[:, :, None, None]
        agent_yaws = agent_yaws * agent_mask[:, :, None, None]

        expanded_agent_positions = agent_positions[:, None].expand(-1, mode_count, -1, -1, -1)
        expanded_agent_yaws = agent_yaws[:, None].expand(-1, mode_count, -1, -1, -1)
        expanded_agent_controls = agent_controls[:, None].expand(-1, mode_count, -1, -1, -1)
        scene_positions = torch.cat([ego_positions[:, :, None], expanded_agent_positions], dim=2)
        scene_yaws = torch.cat([ego_yaws[:, :, None], expanded_agent_yaws], dim=2)
        scene_controls = torch.cat([ego_controls[:, :, None], expanded_agent_controls], dim=2)
        ego_scene_availability = torch.ones(
            batch_size,
            mode_count,
            1,
            self.future_steps,
            dtype=torch.bool,
            device=ego_positions.device,
        )
        agent_scene_availability = current_availability[:, None, 1:, None].expand(
            -1,
            mode_count,
            -1,
            self.future_steps,
        )
        scene_availabilities = torch.cat(
            [ego_scene_availability, agent_scene_availability],
            dim=2,
        )

        return {
            "positions": ego_positions,
            "yaws": ego_yaws,
            "availabilities": torch.ones(
                batch_size,
                mode_count,
                self.future_steps,
                dtype=torch.bool,
                device=ego_positions.device,
            ),
            "scores": torch.ones(batch_size, mode_count, device=ego_positions.device),
            "controls": ego_controls,
            "agent_positions": expanded_agent_positions,
            "agent_yaws": expanded_agent_yaws,
            "agent_controls": expanded_agent_controls,
            "scene_positions": scene_positions,
            "scene_yaws": scene_yaws,
            "scene_availabilities": scene_availabilities,
            "scene_controls": scene_controls,
            "trajectories": torch.cat([scene_positions, scene_yaws], dim=-1),
            "agent_availabilities": agent_scene_availability,
        }

    def extract_features(
        self,
        tensors: Dict[str, torch.Tensor],
        return_encoder_features: bool = False,
        encoder_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> tuple:
        agent_positions, current_states, current_availability = _current_scene_states(tensors)
        image = _ensure_image_batch(tensors["image"])
        if encoder_features is None:
            encoder_features = self.shared_encoder(image)
        agent_features, global_features = self.roi_head(tensors, agent_positions, encoder_features)
        global_features = global_features[:, None].expand(-1, agent_features.shape[1], -1)
        # Preserve the official AgentAwareRasterizedModel feature concat:
        # per-agent local ROI features plus current-scene global raster features.
        all_features = torch.cat([agent_features, global_features], dim=-1)
        if self.history_encoder is not None:
            history_positions, history_yaws = _scene_history_trajectories(tensors, all_features)
            history_trajectory = torch.cat([history_positions, history_yaws], dim=-1)
            history_features = self.history_encoder(
                history_trajectory.reshape(-1, history_trajectory.shape[-2], 3)
            ).reshape(*history_trajectory.shape[:2], -1)
            all_features = torch.cat([all_features, history_features], dim=-1)
        if self.transformer is not None:
            # The released predictor uses SimpleTransformer across agents.
            all_features = self.transformer(
                all_features,
                current_availability.to(all_features),
                agent_positions.to(all_features),
            )
        all_features = all_features * current_availability.to(all_features).unsqueeze(-1)
        if return_encoder_features:
            return all_features, current_states.to(all_features), current_availability, encoder_features
        return all_features, current_states.to(all_features), current_availability


class BitsBiLevelTorchModel(nn.Module):
    """BITS core model with the official planner/predictor boundary."""

    def __init__(
        self,
        image_channels: int,
        future_steps: int,
        hidden_dim: int = 128,
        model_arch: str = "resnet18",
        context_size: int = 30,
        roi_feature_size: int = 7,
        roi_layer_key: str = "layer2",
        history_conditioning: bool = False,
        use_transformer: bool = False,
        config: Optional[BitsConfig] = None,
    ):
        super().__init__()
        self.image_channels = int(image_channels)
        self.future_steps = int(future_steps)
        self.hidden_dim = int(hidden_dim)
        self.model_arch = model_arch
        self.context_size = int(context_size)
        self.roi_feature_size = int(roi_feature_size)
        self.roi_layer_key = roi_layer_key
        self.history_conditioning = bool(history_conditioning)
        self.use_transformer = bool(use_transformer)
        self.config = config or BitsConfig(future_steps=future_steps)
        self.shared_encoder = SharedRasterEncoder(
            image_channels=image_channels,
            model_arch=model_arch,
            feature_dim=hidden_dim,
        )
        self.planner = BitsSpatialPlannerModule(
            encoder_channels=self.shared_encoder.feature_channels,
        )
        self.predictor = BitsAgentAwareTrajectoryModule(
            future_steps=future_steps,
            image_channels=image_channels,
            global_feature_dim=hidden_dim,
            agent_feature_dim=hidden_dim,
            goal_feature_dim=max(8, hidden_dim // 4),
            decoder_layer_dims=(hidden_dim, hidden_dim, hidden_dim),
            context_size=context_size,
            roi_feature_size=roi_feature_size,
            roi_layer_key=roi_layer_key,
            model_arch=model_arch,
            history_conditioning=history_conditioning,
            use_transformer=use_transformer,
            config=self.config,
            shared_encoder=self.shared_encoder,
        )

    def forward(
        self,
        tensors: Dict[str, torch.Tensor],
        goal_tensors: Dict[str, torch.Tensor] = None,
        use_ground_truth_goal: bool = False,
        num_samples: Optional[int] = None,
        mask_drivable: bool = False,
    ) -> Dict[str, object]:
        image = _ensure_image_batch(tensors["image"])
        encoder_features = self.shared_encoder(image)
        feature_context = self.predictor.extract_features(
            tensors,
            return_encoder_features=True,
            encoder_features=encoder_features,
        )
        plan = self.planner(
            tensors,
            num_samples=num_samples,
            mask_drivable=mask_drivable,
            encoder_features=encoder_features,
        )
        if use_ground_truth_goal:
            if goal_tensors is None:
                raise ValueError("goal_tensors are required when use_ground_truth_goal=True.")
            goal_positions = _ensure_vector_batch(goal_tensors["goal_position"])
            goal_yaws = _ensure_vector_batch(goal_tensors["goal_yaw"])
        else:
            goal_positions = plan["positions"]
            goal_yaws = plan["yaws"]

        predictions = self.predictor(
            tensors,
            goal_positions,
            goal_yaws,
            feature_context=feature_context,
        )
        return {"plan": plan, "predictions": predictions}


def decode_bits_spatial_prediction(
    spatial_prediction: torch.Tensor,
    agent_from_raster: torch.Tensor,
    drivable_map=None,
    num_samples: Optional[int] = None,
    mask_drivable: bool = False,
) -> Dict[str, torch.Tensor]:
    """Decode BITS spatial planner logits into ego-frame goal candidates."""

    batch_size, _channels, height, width = spatial_prediction.shape
    if _channels != 4:
        raise ValueError("BITS spatial prediction must have 4 channels.")

    logits = spatial_prediction[:, 0]
    prob_map = torch.softmax(logits.flatten(1), dim=-1).reshape(batch_size, height, width)
    if mask_drivable and drivable_map is not None:
        drivable = _ensure_map_batch(drivable_map).to(device=prob_map.device, dtype=torch.bool)
        empty_mask = drivable.flatten(1).sum(dim=-1) == 0
        if torch.any(empty_mask):
            drivable = drivable.clone()
            drivable[empty_mask] = True
        drivable = drivable.to(dtype=prob_map.dtype)
        prob_map = prob_map * drivable

    prob_sum = prob_map.flatten(1).sum(dim=-1)
    zero_index = prob_sum == 0
    if torch.any(zero_index):
        prob_map = prob_map.clone()
        prob_map[zero_index] = torch.ones_like(prob_map[zero_index])
        prob_sum = prob_map.flatten(1).sum(dim=-1)
    prob_map = prob_map / prob_sum[:, None, None].clamp_min(1e-8)

    flat_prob_map = prob_map.flatten(1)
    if num_samples is not None:
        flat_indices = torch.multinomial(
            flat_prob_map,
            num_samples=int(num_samples),
            replacement=True,
        )
        scores = torch.gather(flat_prob_map, dim=1, index=flat_indices)
    else:
        scores, flat_indices = torch.max(flat_prob_map, dim=1)
        scores = scores.unsqueeze(1)
        flat_indices = flat_indices.unsqueeze(1)

    cols = flat_indices % width
    rows = torch.div(flat_indices, width, rounding_mode="floor")

    # TBSIM SpatialPlanner keeps residual x/y inside the selected pixel. Preserve
    # that behavior by treating network outputs as logits before sigmoid.
    residual_map = torch.sigmoid(spatial_prediction[:, 1:3])
    yaw_map = spatial_prediction[:, 3:4]
    gather_index = flat_indices[:, None].expand(-1, 2, -1)
    residuals = torch.gather(residual_map.flatten(2), dim=2, index=gather_index).transpose(1, 2)
    yaws = torch.gather(yaw_map.flatten(2), dim=2, index=flat_indices[:, None]).transpose(1, 2)

    pixel_positions = torch.stack([cols, rows], dim=-1).to(residuals.dtype) + residuals
    positions = _transform_raster_points(pixel_positions, agent_from_raster)
    return {
        "positions": positions,
        "yaws": yaws,
        "scores": scores,
        "location_prob_map": prob_map,
        "pixel_positions": pixel_positions,
    }


def compute_bits_torch_losses(
    output: Dict[str, object],
    tensors: Dict[str, torch.Tensor],
    goal_tensors: Dict[str, torch.Tensor] = None,
    config: Optional[BitsConfig] = None,
    loss_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    """Compute differentiable BITS losses for the torch reproduction path."""

    losses: Dict[str, torch.Tensor] = {}
    predictions = output["predictions"]
    losses.update(_trajectory_losses(predictions, tensors))
    if goal_tensors is not None and "plan" in output and _has_spatial_goal_targets(goal_tensors):
        losses.update(_spatial_planner_losses(output["plan"], goal_tensors))
    resolved_weights = _resolve_torch_loss_weights(config, loss_weights)
    losses["total"] = _weighted_torch_loss_sum(losses, resolved_weights)
    return losses


def _batch_chunks(batches: Iterable[BitsBatch], batch_size: int):
    chunk = []
    for batch in batches:
        chunk.append(batch)
        if len(chunk) == batch_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def run_bits_torch_epoch(
    model: nn.Module,
    batches: Iterable[BitsBatch],
    optimizer=None,
    batch_size: int = 1,
    device=None,
    dtype=None,
    use_ground_truth_goal: bool = True,
    num_samples: Optional[int] = None,
    mask_drivable: bool = False,
    include_optional: bool = True,
    config: Optional[BitsConfig] = None,
    loss_weights: Optional[Dict[str, float]] = None,
) -> BitsTorchEpochResult:
    """Run one compact BITS torch training or validation epoch."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if device is not None:
        model.to(device)
    if dtype is not None:
        model.to(dtype=dtype)
    model.train(optimizer is not None)

    loss_values: Dict[str, list] = {}
    sample_count = 0
    step_count = 0
    for batch_chunk in _batch_chunks(batches, batch_size):
        torch_batch = collate_bits_batches_to_torch(
            batch_chunk,
            device=device,
            dtype=dtype,
            include_optional=include_optional,
        )
        goals = collate_bits_goal_supervisions_to_torch(
            [build_goal_supervision(batch) for batch in batch_chunk],
            device=device,
            dtype=dtype,
        )

        # Keep the training loop narrow: collate, forward, loss, optimizer step.
        # Checkpointing, logging, and distributed orchestration live outside.
        with torch.set_grad_enabled(optimizer is not None):
            output = model(
                torch_batch.tensors,
                goal_tensors=goals,
                use_ground_truth_goal=use_ground_truth_goal,
                num_samples=num_samples,
                mask_drivable=mask_drivable,
            )
            losses = compute_bits_torch_losses(
                output,
                torch_batch.tensors,
                goals,
                config=config,
                loss_weights=loss_weights,
            )
            if optimizer is not None:
                optimizer.zero_grad()
                losses["total"].backward()
                optimizer.step()

        for name, value in losses.items():
            loss_values.setdefault(name, []).append(float(value.detach().cpu()))
        sample_count += len(batch_chunk)
        step_count += 1

    mean_losses = {
        name: float(np.mean(values)) for name, values in loss_values.items()
    }
    return BitsTorchEpochResult(
        sample_count=sample_count,
        step_count=step_count,
        mean_total_loss=mean_losses.get("total", 0.0),
        mean_losses=mean_losses,
    )


def run_bits_planner_torch_epoch(
    model: BitsBiLevelTorchModel,
    batches: Iterable[BitsBatch],
    optimizer=None,
    batch_size: int = 1,
    device=None,
    dtype=None,
    include_optional: bool = True,
    config: Optional[BitsConfig] = None,
    loss_weights: Optional[Dict[str, float]] = None,
    use_shared_encoder: bool = True,
    freeze_shared_encoder: bool = False,
) -> BitsTorchEpochResult:
    """Train or validate only the high-level BITS spatial planner."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if device is not None:
        model.to(device)
    if dtype is not None:
        model.to(dtype=dtype)
    model.train(optimizer is not None)
    if freeze_shared_encoder:
        model.shared_encoder.eval()

    resolved_weights = _resolve_torch_loss_weights(config, loss_weights)
    loss_values: Dict[str, list] = {}
    sample_count = 0
    step_count = 0
    for batch_chunk in _batch_chunks(batches, batch_size):
        torch_batch = collate_bits_batches_to_torch(
            batch_chunk,
            device=device,
            dtype=dtype,
            include_optional=include_optional,
        )
        goals = collate_bits_goal_supervisions_to_torch(
            [build_goal_supervision(batch) for batch in batch_chunk],
            device=device,
            dtype=dtype,
        )
        with torch.set_grad_enabled(optimizer is not None):
            encoder_features = None
            if use_shared_encoder:
                if freeze_shared_encoder:
                    with torch.no_grad():
                        encoder_features = model.shared_encoder(
                            _ensure_image_batch(torch_batch.tensors["image"])
                        )
                else:
                    encoder_features = model.shared_encoder(
                        _ensure_image_batch(torch_batch.tensors["image"])
                    )
            plan = model.planner(torch_batch.tensors, encoder_features=encoder_features)
            losses = _spatial_planner_losses(plan, goals)
            losses["total"] = _weighted_torch_loss_sum(losses, resolved_weights)
            if optimizer is not None:
                optimizer.zero_grad()
                losses["total"].backward()
                optimizer.step()

        for name, value in losses.items():
            loss_values.setdefault(name, []).append(float(value.detach().cpu()))
        sample_count += len(batch_chunk)
        step_count += 1

    mean_losses = {
        name: float(np.mean(values)) for name, values in loss_values.items()
    }
    return BitsTorchEpochResult(
        sample_count=sample_count,
        step_count=step_count,
        mean_total_loss=mean_losses.get("total", 0.0),
        mean_losses=mean_losses,
    )


def bits_prediction_from_torch(
    positions,
    yaws,
    availabilities=None,
    scores=None,
) -> BitsPrediction:
    """Convert torch model outputs back to the numpy prediction schema."""

    positions_np = _to_numpy(positions).astype(float, copy=False)
    yaws_np = _to_numpy(yaws).astype(float, copy=False)
    if availabilities is None:
        availabilities_np = np.ones(positions_np.shape[:2], dtype=bool)
    else:
        availabilities_np = _to_numpy(availabilities).astype(bool, copy=False)
    if scores is None:
        scores_np = np.ones(positions_np.shape[0], dtype=float)
    else:
        scores_np = _to_numpy(scores).astype(float, copy=False)

    return BitsPrediction(
        positions=positions_np,
        yaws=yaws_np,
        availabilities=availabilities_np,
        scores=scores_np,
    )


class TorchBitsPolicy(BitsPolicy):
    """Wrap a PyTorch module behind the standard BITS policy interface."""

    def __init__(
        self,
        module,
        device=None,
        dtype=None,
        include_optional: bool = True,
        module_input: str = "tensors",
        plan_scorer: Optional[BitsPlanScorer] = None,
        select_best_plan: bool = True,
        module_kwargs: Optional[Dict[str, object]] = None,
    ):
        if module_input not in {"tensors", "batch"}:
            raise ValueError("module_input must be either 'tensors' or 'batch'.")
        self.module = module
        self.device = device
        self.dtype = dtype
        self.include_optional = include_optional
        self.module_input = module_input
        self.plan_scorer = plan_scorer
        self.select_best_plan = select_best_plan
        self.module_kwargs = dict(module_kwargs or {})
        self.last_plan: Optional[BitsPlan] = None
        self.last_plan_scores = None
        self.last_selected_plan: Optional[BitsPlan] = None

    def predict_batch(self, batch: BitsBatch) -> BitsPrediction:
        """Run a torch module and return the normal numpy BITS prediction."""

        torch_batch = bits_batch_to_torch(
            batch,
            device=self.device,
            dtype=self.dtype,
            include_optional=self.include_optional,
        )
        module_arg = torch_batch if self.module_input == "batch" else torch_batch.tensors

        # Treat the torch module as a policy head: it maps BITS tensors to
        # trajectory tensors, then optional plans/predictions are collapsed by the
        # official multi-candidate closed-loop selection.
        with torch.no_grad():
            output = self.module(module_arg, **self.module_kwargs)

        if isinstance(output, dict) and "plan" in output and "predictions" in output:
            return self._prediction_from_bilevel_output(batch, output)
        return _prediction_from_module_output(output)

    def _prediction_from_bilevel_output(
        self,
        batch: BitsBatch,
        output: Dict[str, object],
    ) -> BitsPrediction:
        prediction = _squeeze_single_batch_prediction(_prediction_from_module_output(output["predictions"]))
        plan = _plan_from_torch_output(output["plan"], prediction)
        agent_prediction = _agent_prediction_from_torch_output(output["predictions"])
        scorer = self.plan_scorer or BitsPlanScorer()
        self.last_plan = plan
        self.last_plan_scores = scorer.score_batch(batch, plan, agent_prediction=agent_prediction)
        if self.select_best_plan:
            best_index = int(np.argmax(self.last_plan_scores.total))
            self.last_selected_plan = scorer.select_plan(plan, self.last_plan_scores)
            return BitsPrediction(
                positions=prediction.positions[[best_index]].copy(),
                yaws=prediction.yaws[[best_index]].copy(),
                availabilities=prediction.availabilities[[best_index]].copy(),
                scores=self.last_plan_scores.total[[best_index]].copy(),
            )

        self.last_selected_plan = BitsPlan(
            positions=plan.positions,
            yaws=plan.yaws,
            availabilities=plan.availabilities,
            scores=self.last_plan_scores.total,
        )
        return BitsPrediction(
            positions=prediction.positions,
            yaws=prediction.yaws,
            availabilities=prediction.availabilities,
            scores=self.last_plan_scores.total,
        )


def _as_tensor(value, dtype, device):
    return torch.as_tensor(value, dtype=dtype, device=device)


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _ensure_image_batch(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        return image.unsqueeze(0)
    return image


def _ensure_sequence_batch(sequence: torch.Tensor) -> torch.Tensor:
    if sequence.ndim == 2:
        return sequence.unsqueeze(0)
    return sequence


def _ensure_availability_batch(availability: torch.Tensor) -> torch.Tensor:
    if availability.ndim == 1:
        return availability.unsqueeze(0)
    return availability


def _ensure_vector_batch(vector: torch.Tensor) -> torch.Tensor:
    if vector.ndim == 1:
        return vector.unsqueeze(0)
    return vector


def _ensure_goal_batch(goal: torch.Tensor) -> torch.Tensor:
    if goal.ndim == 1:
        return goal.reshape(1, 1, goal.shape[0])
    if goal.ndim == 2:
        return goal.unsqueeze(1)
    return goal


def _ensure_scalar_batch(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 0:
        return value.reshape(1, 1)
    if value.ndim == 1:
        return value[:, None]
    return value


def _ensure_matrix_batch(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim == 2:
        return matrix.unsqueeze(0)
    return matrix


def _ensure_map_batch(map_: torch.Tensor) -> torch.Tensor:
    if map_.ndim == 2:
        return map_.unsqueeze(0)
    return map_


def _resolve_drivable_map(tensors: Dict[str, torch.Tensor]):
    drivable_map = tensors.get("drivable_map")
    if drivable_map is not None:
        return drivable_map
    image = tensors.get("image")
    if image is None:
        return None
    image = _ensure_image_batch(image)
    if image.shape[1] == 0:
        return None
    # The official implementation infers drivable area from raster input when no
    # drivable_map is provided. In Tactics2D rasters, the first of the last three
    # static channels is the drivable channel.
    static_start = max(0, image.shape[1] - 3)
    return image[:, static_start] > 0


def _current_scene_states(tensors: Dict[str, torch.Tensor]) -> tuple:
    ego_state = _ego_current_states(tensors)
    agent_states = _agent_current_states(tensors)
    agent_availability = tensors["all_other_agents_history_availability"]
    if agent_availability.ndim == 2:
        agent_availability = agent_availability.unsqueeze(0)
    agent_types = tensors["all_other_agents_types"]
    if agent_types.ndim == 1:
        agent_types = agent_types.unsqueeze(0)
    current_agent_availability = agent_availability[..., -1] & (agent_types > 0)
    ego_availability = torch.ones(
        ego_state.shape[0],
        1,
        dtype=torch.bool,
        device=ego_state.device,
    )
    current_availability = torch.cat(
        [ego_availability, current_agent_availability.to(ego_state.device)],
        dim=1,
    )
    positions = torch.cat([ego_state[:, None, 0:2], agent_states[..., 0:2]], dim=1)
    states = torch.cat([ego_state[:, None], agent_states], dim=1)
    states = states * current_availability.to(states).unsqueeze(-1)
    return positions, states, current_availability


def _ego_current_states(tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
    curr_speed = _ensure_scalar_batch(tensors["curr_speed"])
    history_yaws = _ensure_sequence_batch(tensors["history_yaws"]).to(curr_speed)
    return torch.cat(
        [
            torch.zeros(*curr_speed.shape[:-1], 2, dtype=curr_speed.dtype, device=curr_speed.device),
            curr_speed,
            history_yaws[..., -1, :],
        ],
        dim=-1,
    )


def _agent_current_states(tensors: Dict[str, torch.Tensor]) -> torch.Tensor:
    history_positions = tensors["all_other_agents_history_positions"]
    if history_positions.ndim == 3:
        history_positions = history_positions.unsqueeze(0)
    history_yaws = tensors["all_other_agents_history_yaws"]
    if history_yaws.ndim == 3:
        history_yaws = history_yaws.unsqueeze(0)
    curr_speed = tensors["all_other_agents_curr_speed"]
    if curr_speed.ndim == 1:
        curr_speed = curr_speed.unsqueeze(0)
    history_availability = tensors["all_other_agents_history_availability"]
    if history_availability.ndim == 2:
        history_availability = history_availability.unsqueeze(0)

    history_positions = history_positions.to(curr_speed)
    history_yaws = history_yaws.to(curr_speed)
    current_available = history_availability[..., -1].to(curr_speed).unsqueeze(-1)
    return torch.cat(
        [
            history_positions[..., -1, :] * current_available,
            curr_speed[..., None],
            history_yaws[..., -1, :] * current_available,
        ],
        dim=-1,
    )


def _scene_history_trajectories(
    tensors: Dict[str, torch.Tensor],
    reference: torch.Tensor,
) -> tuple:
    ego_positions = _ensure_sequence_batch(tensors["history_positions"]).to(reference)
    ego_yaws = _ensure_sequence_batch(tensors["history_yaws"]).to(reference)
    other_positions = tensors["all_other_agents_history_positions"]
    if other_positions.ndim == 3:
        other_positions = other_positions.unsqueeze(0)
    other_yaws = tensors["all_other_agents_history_yaws"]
    if other_yaws.ndim == 3:
        other_yaws = other_yaws.unsqueeze(0)
    positions = torch.cat([ego_positions[:, None], other_positions.to(reference)], dim=1)
    yaws = torch.cat([ego_yaws[:, None], other_yaws.to(reference)], dim=1)
    return positions, yaws


def _transform_agent_points(points: torch.Tensor, raster_from_agent: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(*points.shape[:-1], 1, dtype=points.dtype, device=points.device)
    homogeneous = torch.cat([points, ones], dim=-1)
    transform = raster_from_agent.to(device=points.device, dtype=points.dtype)
    return torch.matmul(homogeneous, transform.transpose(1, 2))[..., :2]


def _build_indexed_upright_rois(
    raster_points: torch.Tensor,
    context_size: int,
) -> torch.Tensor:
    """Build official-style [batch_index, x1, y1, x2, y2] ROI boxes."""

    batch_size, agent_count = raster_points.shape[:2]
    half = float(context_size) / 2.0
    x_center = raster_points[..., 0]
    y_center = raster_points[..., 1]
    boxes = torch.stack(
        [
            x_center - half,
            y_center - half,
            x_center + half,
            y_center + half,
        ],
        dim=-1,
    ).reshape(batch_size * agent_count, 4)
    batch_indices = (
        torch.arange(batch_size, dtype=raster_points.dtype, device=raster_points.device)
        .unsqueeze(1)
        .expand(-1, agent_count)
        .reshape(-1, 1)
    )
    return torch.cat([batch_indices, boxes], dim=1)


def _transform_raster_points(points: torch.Tensor, agent_from_raster: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(*points.shape[:-1], 1, dtype=points.dtype, device=points.device)
    homogeneous = torch.cat([points, ones], dim=-1)
    transform = agent_from_raster.to(device=points.device, dtype=points.dtype)
    return torch.matmul(homogeneous, transform.transpose(1, 2))[..., :2]


def _trajectory_losses(
    predictions: Dict[str, torch.Tensor],
    tensors: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if "scene_positions" in predictions and "scene_yaws" in predictions:
        return _scene_trajectory_losses(predictions, tensors)

    pred_positions = predictions["positions"]
    pred_yaws = predictions["yaws"]
    target_positions = _ensure_sequence_batch(tensors["target_positions"]).to(pred_positions)
    target_yaws = _ensure_sequence_batch(tensors["target_yaws"]).to(pred_yaws)
    availability = _ensure_availability_batch(tensors["target_availabilities"]).to(
        device=pred_positions.device,
        dtype=pred_positions.dtype,
    )
    if pred_positions.ndim == 3:
        pred_positions = pred_positions.unsqueeze(1)
    if pred_yaws.ndim == 3:
        pred_yaws = pred_yaws.unsqueeze(1)

    mode_errors = torch.linalg.norm(
        pred_positions - target_positions[:, None], dim=-1
    ) * availability[:, None]
    valid_steps = availability.sum(dim=-1).clamp_min(1.0)
    mode_ade = mode_errors.sum(dim=-1) / valid_steps[:, None]
    best_mode = torch.argmin(mode_ade, dim=-1)
    batch_indices = torch.arange(pred_positions.shape[0], device=pred_positions.device)
    best_positions = pred_positions[batch_indices, best_mode]
    best_yaws = pred_yaws[batch_indices, best_mode]

    yaw_delta = torch.atan2(
        torch.sin(best_yaws - target_yaws),
        torch.cos(best_yaws - target_yaws),
    )
    # Match TBSIM trajectory_loss by applying MSE to x/y/yaw as one vector.
    predicted_trajectory = torch.cat([best_positions, yaw_delta], dim=-1)
    target_trajectory = torch.cat(
        [target_positions, torch.zeros_like(target_yaws)],
        dim=-1,
    )
    prediction_loss = torch.mean(
        F.mse_loss(predicted_trajectory, target_trajectory, reduction="none")
        * availability.unsqueeze(-1)
    )
    goal_loss = _last_available_mse_loss(
        predicted_trajectory,
        target_trajectory,
        availability,
    )
    return {
        "prediction_loss": prediction_loss,
        "goal_loss": goal_loss,
    }


def _scene_trajectory_losses(
    predictions: Dict[str, torch.Tensor],
    tensors: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    pred_positions = predictions["scene_positions"]
    pred_yaws = predictions["scene_yaws"]
    target_positions, target_yaws, availability = _scene_targets(tensors)
    target_positions = target_positions.to(pred_positions)
    target_yaws = target_yaws.to(pred_yaws)
    availability = availability.to(device=pred_positions.device, dtype=pred_positions.dtype)

    mode_errors = torch.linalg.norm(
        pred_positions[:, :, 0] - target_positions[:, None, 0],
        dim=-1,
    ) * availability[:, None, 0]
    valid_steps = availability[:, 0].sum(dim=-1).clamp_min(1.0)
    mode_ade = mode_errors.sum(dim=-1) / valid_steps[:, None]
    best_mode = torch.argmin(mode_ade, dim=-1)
    batch_indices = torch.arange(pred_positions.shape[0], device=pred_positions.device)
    best_positions = pred_positions[batch_indices, best_mode]
    best_yaws = pred_yaws[batch_indices, best_mode]

    yaw_delta = torch.atan2(
        torch.sin(best_yaws - target_yaws),
        torch.cos(best_yaws - target_yaws),
    )
    predicted_trajectory = torch.cat([best_positions, yaw_delta], dim=-1)
    target_trajectory = torch.cat(
        [target_positions, torch.zeros_like(target_yaws)],
        dim=-1,
    )
    prediction_loss = torch.mean(
        F.mse_loss(predicted_trajectory, target_trajectory, reduction="none")
        * availability.unsqueeze(-1)
    )
    goal_loss = _last_available_mse_loss(
        predicted_trajectory,
        target_trajectory,
        availability,
    )
    return {
        "prediction_loss": prediction_loss,
        "goal_loss": goal_loss,
    }


def _scene_targets(
    tensors: Dict[str, torch.Tensor],
) -> tuple:
    ego_positions = _ensure_sequence_batch(tensors["target_positions"])
    ego_yaws = _ensure_sequence_batch(tensors["target_yaws"])
    ego_availability = _ensure_availability_batch(tensors["target_availabilities"])
    agent_positions = tensors["all_other_agents_future_positions"]
    if agent_positions.ndim == 3:
        agent_positions = agent_positions.unsqueeze(0)
    agent_yaws = tensors["all_other_agents_future_yaws"]
    if agent_yaws.ndim == 3:
        agent_yaws = agent_yaws.unsqueeze(0)
    agent_availability = tensors["all_other_agents_future_availability"]
    if agent_availability.ndim == 2:
        agent_availability = agent_availability.unsqueeze(0)
    return (
        torch.cat([ego_positions[:, None], agent_positions], dim=1),
        torch.cat([ego_yaws[:, None], agent_yaws], dim=1),
        torch.cat([ego_availability[:, None], agent_availability], dim=1),
    )


def _last_available_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    availabilities: torch.Tensor,
) -> torch.Tensor:
    valid_counts = availabilities.sum(dim=-1)
    if not torch.any(valid_counts > 0):
        return torch.mean(predictions * 0.0)

    last_indices = valid_counts.to(dtype=torch.long).clamp_min(1) - 1
    time_count = availabilities.shape[-1]
    step_index = torch.arange(time_count, device=availabilities.device)
    goal_mask = step_index.reshape(*([1] * (availabilities.ndim - 1)), time_count)
    goal_mask = goal_mask == last_indices[..., None]
    goal_mask = goal_mask & (valid_counts[..., None] > 0)
    return torch.mean(
        F.mse_loss(predictions, targets, reduction="none") * goal_mask.unsqueeze(-1).to(predictions)
    )


def _spatial_planner_losses(
    plan: Dict[str, torch.Tensor],
    goal_tensors: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    spatial_prediction = plan["spatial_prediction"]
    location_logits = plan.get("location_logits", spatial_prediction[:, 0])
    batch_size, channels, _height, _width = spatial_prediction.shape
    pixel_flat = goal_tensors["goal_position_pixel_flat"].long().reshape(batch_size)
    ce_loss = F.cross_entropy(location_logits.flatten(1), pixel_flat)

    gather_index = pixel_flat[:, None, None].expand(-1, channels, 1)
    local_pred = torch.gather(spatial_prediction.flatten(2), dim=2, index=gather_index)
    local_pred = local_pred.squeeze(-1)
    residual_pred = torch.sigmoid(local_pred[:, 1:3])
    yaw_pred = local_pred[:, 3:4]
    bce_loss = F.binary_cross_entropy_with_logits(
        location_logits,
        _ensure_map_batch(goal_tensors["goal_spatial_map"]).to(location_logits),
    )
    residual_loss = F.mse_loss(
        residual_pred,
        _ensure_vector_batch(goal_tensors["goal_position_residual"]).to(residual_pred),
    )
    yaw_loss = F.mse_loss(
        yaw_pred,
        _ensure_vector_batch(goal_tensors["goal_yaw"]).to(yaw_pred),
    )
    return {
        "pixel_bce_loss": bce_loss,
        "pixel_ce_loss": ce_loss,
        "pixel_res_loss": residual_loss,
        "pixel_yaw_loss": yaw_loss,
    }


def _has_spatial_goal_targets(goal_tensors: Dict[str, torch.Tensor]) -> bool:
    return all(
        goal_tensors.get(key) is not None
        for key in (
            "goal_position_pixel_flat",
            "goal_position_residual",
            "goal_spatial_map",
            "goal_yaw",
        )
    )


def _resolve_torch_loss_weights(
    config: Optional[BitsConfig],
    loss_weights: Optional[Dict[str, float]],
) -> Dict[str, float]:
    resolved = dict((config or BitsConfig()).torch_loss_weights)
    if loss_weights:
        resolved.update(loss_weights)
    return resolved


def _weighted_torch_loss_sum(
    losses: Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> torch.Tensor:
    total = None
    for name, loss in losses.items():
        weighted = loss * float(weights.get(name, 1.0))
        total = weighted if total is None else total + weighted
    if total is None:
        raise ValueError("No torch BITS losses were computed.")
    return total


def _prediction_from_module_output(output) -> BitsPrediction:
    if isinstance(output, BitsPrediction):
        return output

    if isinstance(output, dict):
        positions = output["positions"]
        yaws = output["yaws"]
        availabilities = output.get("availabilities")
        scores = output.get("scores")
    elif isinstance(output, (tuple, list)):
        if len(output) < 2 or len(output) > 4:
            raise ValueError("Torch BITS module output must contain 2 to 4 values.")
        positions = output[0]
        yaws = output[1]
        availabilities = output[2] if len(output) >= 3 else None
        scores = output[3] if len(output) >= 4 else None
    else:
        raise TypeError("Torch BITS module output must be a dict, tuple, or BitsPrediction.")

    return bits_prediction_from_torch(positions, yaws, availabilities, scores)


def _squeeze_single_batch_prediction(prediction: BitsPrediction) -> BitsPrediction:
    if prediction.positions.ndim != 4:
        return prediction
    if prediction.positions.shape[0] != 1:
        raise ValueError("TorchBitsPolicy only supports one BITS batch at a time.")
    return BitsPrediction(
        positions=prediction.positions[0],
        yaws=prediction.yaws[0],
        availabilities=prediction.availabilities[0],
        scores=prediction.scores.reshape(-1),
    )


def _plan_from_torch_output(plan_output, fallback_prediction: BitsPrediction) -> BitsPlan:
    if isinstance(plan_output, BitsPlan):
        return plan_output
    if not isinstance(plan_output, dict):
        raise TypeError("Torch BITS plan output must be a dict or BitsPlan.")

    scores = plan_output.get("log_likelihood", plan_output.get("scores"))

    if scores is None:
        plan_scores = np.asarray(fallback_prediction.scores, dtype=float).reshape(-1)
    else:
        scores_np = _to_numpy(scores).astype(float, copy=False)
        plan_scores = scores_np.reshape(-1)

    return BitsPlan(
        positions=fallback_prediction.positions,
        yaws=fallback_prediction.yaws,
        availabilities=fallback_prediction.availabilities,
        scores=plan_scores,
    )


def _agent_prediction_from_torch_output(prediction_output) -> Optional[BitsAgentPrediction]:
    if not isinstance(prediction_output, dict):
        return None
    if "agent_positions" in prediction_output and "agent_yaws" in prediction_output:
        positions = _to_numpy(prediction_output["agent_positions"])
        yaws = _to_numpy(prediction_output["agent_yaws"])
        availabilities = prediction_output.get("agent_availabilities")
        if availabilities is None:
            availabilities = prediction_output.get("scene_availabilities")
            if availabilities is not None:
                availabilities = _to_numpy(availabilities)[..., 1:, :]
        if availabilities is None:
            availabilities = np.ones(positions.shape[:-1], dtype=bool)
        else:
            availabilities = _to_numpy(availabilities).astype(bool, copy=False)
        if positions.ndim == 5 and positions.shape[0] == 1:
            positions = positions[0]
        if yaws.ndim == 5 and yaws.shape[0] == 1:
            yaws = yaws[0]
        if availabilities.ndim == 4 and availabilities.shape[0] == 1:
            availabilities = availabilities[0]
        return BitsAgentPrediction(
            positions=np.asarray(positions, dtype=float),
            yaws=np.asarray(yaws, dtype=float),
            availabilities=availabilities,
        )
    return None
