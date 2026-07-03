# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""High-level PyTorch model assembly for BITS-style bi-level imitation.

Contains the full BITS model classes (:class:`BitsBiLevelTorchModel`,
:class:`BitsSpatialPlannerModule`, :class:`BitsAgentAwareTrajectoryModule`),
ROI map encoders, output decoding, and the :class:`TorchBitsPolicy` wrapper.

Low-level building blocks (MLP, Attention, UNet, dynamics …) live in
:mod:`.torch_base`.
"""

from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import RoIAlign

from .config import BitsConfig
from .model import BitsAgentPrediction, BitsPlan, BitsPlanScorer, BitsPolicy, BitsPrediction
from .schema import BitsBatch
from .torch_base import (
    BitsMLP,
    BitsMLPTrajectoryDecoder,
    BitsRasterBackbone,
    BitsRNNTrajectoryEncoder,
    BitsSimpleTransformer,
    BitsTorchBatch,
    SharedRasterEncoder,
    SpatialGoalUNetDecoder,
)

# ---------------------------------------------------------------------------
# Batch key constants (used by batch-conversion helpers)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Batch conversion helpers
# ---------------------------------------------------------------------------


def bits_batch_to_torch(
    batch: BitsBatch, device=None, dtype=None, include_optional: bool = True
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
    batches: Iterable[BitsBatch], device=None, dtype=None, include_optional: bool = True
) -> BitsTorchBatch:
    """Stack same-shaped BITS batches into a torch mini-batch."""

    samples = [
        bits_batch_to_torch(batch, device=device, dtype=dtype, include_optional=include_optional)
        for batch in batches
    ]
    if not samples:
        return BitsTorchBatch(
            tensors={}, metadata={"ego_id": [], "frame": [], "agent_ids": [], "lane_id": []}
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


# ---------------------------------------------------------------------------
# ROI map encoders
# ---------------------------------------------------------------------------


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
            device=image.device, dtype=image.dtype
        )
        raster_points = _transform_agent_points(
            agent_positions.to(device=image.device, dtype=image.dtype), raster_from_agent
        )
        # Official BITS defaults to use_rotated_roi=False: build an axis-aligned
        # ROI around each agent raster position, then apply torchvision RoIAlign.
        rois = _build_indexed_upright_rois(raster_points, context_size=self.context_size)
        roi_features = self.roi_align(encoder_features[self.roi_layer_key], rois)
        batch_size, agent_count = raster_points.shape[:2]
        agent_features = self.agent_net(roi_features).reshape(
            batch_size, agent_count, self.agent_feature_dim
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
            image_channels, model_arch=model_arch, feature_dim=global_feature_dim
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
            device=image.device, dtype=image.dtype
        )
        raster_points = _transform_agent_points(
            agent_positions.to(device=image.device, dtype=image.dtype), raster_from_agent
        )
        # Official BITS defaults to use_rotated_roi=False: build an axis-aligned
        # ROI box around each agent raster position, then apply RoIAlign.
        rois = _build_indexed_upright_rois(raster_points, context_size=self.context_size)
        roi_features = self.roi_align(encoder_features[self.roi_layer_key], rois)
        batch_size, agent_count = raster_points.shape[:2]
        agent_features = self.agent_net(roi_features).reshape(
            batch_size, agent_count, self.agent_feature_dim
        )
        return agent_features, global_features, encoder_features


# ---------------------------------------------------------------------------
# Policy and predictor heads
# ---------------------------------------------------------------------------


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
        self, other_features: torch.Tensor, other_states: torch.Tensor, future_steps: int
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


# ---------------------------------------------------------------------------
# Full spatial-planner module
# ---------------------------------------------------------------------------


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
        spatial_prediction = self.spatial_goal_decoder(encoder_features, target_hw=image.shape[-2:])
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


# ---------------------------------------------------------------------------
# Full bi-level torch model
# ---------------------------------------------------------------------------


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
                image_channels=image_channels, model_arch=model_arch, feature_dim=global_feature_dim
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
            BitsSimpleTransformer(src_dim=agent_feature_dim_total) if use_transformer else None
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
        ego_decoded = self.policy_head(ego_features, ego_states, goal_positions, goal_yaws)
        ego_controls = ego_decoded["controls"]
        ego_positions = ego_decoded["positions"]
        ego_yaws = ego_decoded["yaws"]

        other_count = max(agent_count - 1, 0)
        other_features = all_features[:, 1:]
        other_states = current_states[:, 1:]
        agent_decoded = self.future_state_head(
            other_features, other_states, future_steps=self.future_steps
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
            -1, mode_count, -1, self.future_steps
        )
        scene_availabilities = torch.cat([ego_scene_availability, agent_scene_availability], dim=2)

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
            return (
                all_features,
                current_states.to(all_features),
                current_availability,
                encoder_features,
            )
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
            image_channels=image_channels, model_arch=model_arch, feature_dim=hidden_dim
        )
        self.planner = BitsSpatialPlannerModule(
            encoder_channels=self.shared_encoder.feature_channels
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
            tensors, return_encoder_features=True, encoder_features=encoder_features
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
            tensors, goal_positions, goal_yaws, feature_context=feature_context
        )
        return {"plan": plan, "predictions": predictions}


# ---------------------------------------------------------------------------
# Spatial prediction decoding
# ---------------------------------------------------------------------------


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
            flat_prob_map, num_samples=int(num_samples), replacement=True
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


# ---------------------------------------------------------------------------
# Torch-to-numpy prediction helper
# ---------------------------------------------------------------------------


def bits_prediction_from_torch(positions, yaws, availabilities=None, scores=None) -> BitsPrediction:
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
        positions=positions_np, yaws=yaws_np, availabilities=availabilities_np, scores=scores_np
    )


# ---------------------------------------------------------------------------
# TorchBitsPolicy — wraps a torch module behind the BitsPolicy interface
# ---------------------------------------------------------------------------


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
            batch, device=self.device, dtype=self.dtype, include_optional=self.include_optional
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
        self, batch: BitsBatch, output: Dict[str, object]
    ) -> BitsPrediction:
        prediction = _squeeze_single_batch_prediction(
            _prediction_from_module_output(output["predictions"])
        )
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


# ---------------------------------------------------------------------------
# Private helpers — tensor conversion, shape checks, coordinate transform
# ---------------------------------------------------------------------------


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
    ego_availability = torch.ones(ego_state.shape[0], 1, dtype=torch.bool, device=ego_state.device)
    current_availability = torch.cat(
        [ego_availability, current_agent_availability.to(ego_state.device)], dim=1
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
            torch.zeros(
                *curr_speed.shape[:-1], 2, dtype=curr_speed.dtype, device=curr_speed.device
            ),
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


def _scene_history_trajectories(tensors: Dict[str, torch.Tensor], reference: torch.Tensor) -> tuple:
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


def _build_indexed_upright_rois(raster_points: torch.Tensor, context_size: int) -> torch.Tensor:
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


def _transform_raster_points(points: torch.Tensor, agent_from_raster: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(*points.shape[:-1], 1, dtype=points.dtype, device=points.device)
    homogeneous = torch.cat([points, ones], dim=-1)
    transform = agent_from_raster.to(device=points.device, dtype=points.dtype)
    return torch.matmul(homogeneous, transform.transpose(1, 2))[..., :2]


# ---------------------------------------------------------------------------
# Module-output extraction helpers
# ---------------------------------------------------------------------------


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


__all__ = [
    "BitsAgentAwareTrajectoryModule",
    "BitsBiLevelTorchModel",
    "BitsRasterizeROIEncoder",
    "BitsRasterizeROIHead",
    "BitsSpatialPlannerModule",
    "GoalConditionalPolicyHead",
    "FutureStatePredictorHead",
    "TorchBitsPolicy",
    "bits_batch_to_torch",
    "bits_prediction_from_torch",
    "collate_bits_batches_to_torch",
    "decode_bits_spatial_prediction",
]
