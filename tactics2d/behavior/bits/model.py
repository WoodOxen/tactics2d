# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""BITS behavior model and neural-network modules for bi-level imitation learning.

Contains the full BITS model:

- :class:`BitsSpatialPlannerModule` — high-level spatial planner
- :class:`BitsAgentAwareTrajectoryModule` — low-level traffic model
- :class:`BitsBiLevelTorchModel` — full bi-level model combining both
- :class:`BitsBehaviorModel` — public entry point with checkpoint loading

Low-level building blocks (MLP, transformer, UNet, dynamics …) are imported
from :mod:`.blocks`, :mod:`.transformer`, :mod:`.unet`, and :mod:`.dynamics`.
ROI heads come from :mod:`.roi`, policy heads from :mod:`.heads`.
"""

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor

from tactics2d.behavior.base import BehaviorModelBase
from tactics2d.geometry import spatial
from tactics2d.map.element import Map
from tactics2d.participant.trajectory import State, Trajectory

from .config import BitsConfig
from .dataset import BitsBatchBuilder
from .heads import FutureStatePredictorHead, GoalConditionalPolicyHead
from .policy import BitsPolicy, TorchBitsPolicy
from .predictor import BitsPrediction
from .roi import ROIHead
from .schema import BitsBatch
from .scorer import BitsPlanScorer
from .transformer import Transformer as BitsSimpleTransformer
from .unet import BitsRasterBackbone
from .unet import GoalDecoder as SpatialGoalUNetDecoder
from .unet import SharedRasterEncoder

# ---------------------------------------------------------------------------
# Private tensor helpers
# ---------------------------------------------------------------------------


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


def _resolve_drivable_map(tensors):
    """Infer drivable-area mask from raster image when no explicit map exists."""
    drivable_map = tensors.get("drivable_map")
    if drivable_map is not None:
        return drivable_map
    image = tensors.get("image")
    if image is None:
        return None
    image = _add_batch_dim(image, 4)
    if image.shape[1] == 0:
        return None
    static_start = max(0, image.shape[1] - 3)
    return image[:, static_start] > 0


def _unpack_ego_init_states(tensors):
    """Build the ego unicycle initial state tensor [x=0, y=0, speed, yaw].

    The ego is always at the origin in its own frame, so x/y are zero.
    """
    curr_speed = _add_batch_dim(tensors["curr_speed"], 2)
    history_yaws = _add_batch_dim(tensors["history_yaws"], 3).to(curr_speed)
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


def _unpack_agent_init_states(tensors):
    """Build neighbour-agent unicycle initial state tensors."""
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


def _unpack_scene_states(tensors):
    """Unpack ego and agent current unicycle states from the tensor dict.

    Returns:
        A tuple ``(positions, states, availability)`` where:
        - ``positions`` has shape ``(batch, 1+N, 2)``
        - ``states`` has shape ``(batch, 1+N, 4)``
        - ``availability`` has shape ``(batch, 1+N)``
    """
    ego_state = _unpack_ego_init_states(tensors)
    agent_states = _unpack_agent_init_states(tensors)
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


def _unpack_history_trajectories(tensors, reference):
    """Unpack ego and agent history positions/yaws from the tensor dict."""
    ego_positions = _add_batch_dim(tensors["history_positions"], 3).to(reference)
    ego_yaws = _add_batch_dim(tensors["history_yaws"], 3).to(reference)
    other_positions = tensors["all_other_agents_history_positions"]
    if other_positions.ndim == 3:
        other_positions = other_positions.unsqueeze(0)
    other_yaws = tensors["all_other_agents_history_yaws"]
    if other_yaws.ndim == 3:
        other_yaws = other_yaws.unsqueeze(0)
    positions = torch.cat([ego_positions[:, None], other_positions.to(reference)], dim=1)
    yaws = torch.cat([ego_yaws[:, None], other_yaws.to(reference)], dim=1)
    return positions, yaws


def _ensure_goal_batch(goal: torch.Tensor) -> torch.Tensor:
    """Reshape a goal-position tensor to (batch, mode, dim)."""
    if goal.ndim == 1:
        return goal.reshape(1, 1, goal.shape[0])
    if goal.ndim == 2:
        return goal.unsqueeze(1)
    return goal


# ---------------------------------------------------------------------------
# Checkpoint key mapping utilities (TBSIM format support)
# ---------------------------------------------------------------------------


def _normalize_checkpoint_keys(state_dict: Mapping[str, object]) -> Dict[str, object]:
    """Translate legacy/official BITS keys to the shared-encoder module tree."""
    normalized: Dict[str, object] = {}
    replacements = (
        ("predictor.map_encoder.encoder_heads.", "shared_encoder.encoder_heads."),
        ("predictor.map_encoder.roi_head.", "predictor.roi_head."),
        ("predictor.map_encoder.agent_net.", "predictor.roi_head.agent_net."),
        ("predictor.goal_encoder.", "predictor.policy_head.goal_encoder."),
        ("predictor.ego_decoder.", "predictor.policy_head.ego_decoder."),
        ("predictor.agents_decoder.", "predictor.future_state_head.agents_decoder."),
        ("planner.raster_unet.decoder.", "planner.spatial_goal_decoder.decoder."),
        ("planner.raster_unet.encoder_heads.", "shared_encoder.encoder_heads."),
        ("planner.raster_unet.", "planner.spatial_goal_decoder."),
    )
    for key, value in state_dict.items():
        target_key = str(key)
        for old_prefix, new_prefix in replacements:
            if target_key.startswith(old_prefix):
                target_key = new_prefix + target_key[len(old_prefix) :]
                break
        normalized[target_key] = value
    return normalized


def _strip_lightning_prefix(key: str) -> str:
    return key[len("state_dict.") :] if key.startswith("state_dict.") else key


def _map_checkpoint_prefixes(
    state_dict: Mapping[str, object], prefix_map: Mapping[str, str]
) -> Dict[str, object]:
    mapped: Dict[str, object] = {}
    for key, value in state_dict.items():
        normalized_key = _strip_lightning_prefix(str(key))
        target_key = normalized_key
        for source_prefix, target_prefix in prefix_map.items():
            if normalized_key.startswith(source_prefix):
                target_key = target_prefix + normalized_key[len(source_prefix) :]
                break
        mapped[target_key] = value
    return mapped


def _state_dict_from_checkpoint(checkpoint, map_location=None) -> Mapping[str, object]:
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(checkpoint, map_location=map_location)
    if isinstance(checkpoint, Mapping):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], Mapping):
            return checkpoint["state_dict"]
        if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], Mapping):
            return checkpoint["model_state_dict"]
        return checkpoint
    raise TypeError("checkpoint must be a path or a mapping containing tensor weights.")


def _extract_planner_state(checkpoint, map_location=None) -> Dict[str, object]:
    """Extract only ``planner.*`` and ``shared_encoder.*`` tensors from a BITS checkpoint."""
    if isinstance(checkpoint, Mapping):
        payload = checkpoint
    else:
        payload = torch.load(checkpoint, map_location=map_location)
    state_dict = payload.get("model_state_dict", payload)
    normalized = _normalize_checkpoint_keys(state_dict)
    planner_state = {
        key: value
        for key, value in normalized.items()
        if str(key).startswith("planner.") or str(key).startswith("shared_encoder.")
    }
    if not planner_state:
        raise ValueError(
            "Tactics2D planner checkpoint does not contain planner.* or shared_encoder.* weights."
        )
    return planner_state


def _merge_tbsim_state_dicts(
    planner_state_dict: Optional[Mapping[str, object]] = None,
    predictor_state_dict: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Merge official planner/predictor state dicts into one BITS model state dict."""

    def _map_planner(sd):
        mapped = _map_checkpoint_prefixes(
            sd,
            prefix_map={
                "nets.policy.decoder.": "planner.spatial_goal_decoder.decoder.",
                "policy.decoder.": "planner.spatial_goal_decoder.decoder.",
                "nets.policy.encoder_heads.": "shared_encoder.encoder_heads.",
                "policy.encoder_heads.": "shared_encoder.encoder_heads.",
                "nets.policy.": "planner.spatial_goal_decoder.",
                "policy.": "planner.spatial_goal_decoder.",
            },
        )
        return _normalize_checkpoint_keys(mapped)

    def _map_predictor(sd):
        mapped = _map_checkpoint_prefixes(sd, prefix_map={"model.": "predictor."})
        return _normalize_checkpoint_keys(mapped)

    merged: Dict[str, object] = {}
    if planner_state_dict is not None:
        merged.update(_map_planner(planner_state_dict))
    if predictor_state_dict is not None:
        mapped_predictor = _map_predictor(predictor_state_dict)
        overlap = set(merged).intersection(mapped_predictor)
        non_encoder_overlap = [
            key for key in sorted(overlap) if not key.startswith("shared_encoder.encoder_heads.")
        ]
        if non_encoder_overlap:
            raise ValueError(f"Overlapping BITS checkpoint keys: {non_encoder_overlap[:3]}")
        merged.update(mapped_predictor)
    return merged


def _build_bits_model(metadata: dict, config: BitsConfig) -> "BitsBiLevelTorchModel":
    """Create a BitsBiLevelTorchModel from checkpoint metadata."""
    return BitsBiLevelTorchModel(
        image_channels=int(metadata.get("image_channels", 14)),
        future_steps=int(metadata.get("future_steps", 20)),
        hidden_dim=int(
            metadata.get("hidden_dim", metadata.get("schedule", {}).get("hidden_dim", 128))
        ),
        model_arch=str(
            metadata.get("model_arch", metadata.get("schedule", {}).get("model_arch", "resnet18"))
        ),
        context_size=int(
            metadata.get("context_size", metadata.get("schedule", {}).get("context_size", 30))
        ),
        roi_feature_size=int(
            metadata.get(
                "roi_feature_size", metadata.get("schedule", {}).get("roi_feature_size", 7)
            )
        ),
        roi_layer_key=str(
            metadata.get(
                "roi_layer_key", metadata.get("schedule", {}).get("roi_layer_key", "layer2")
            )
        ),
        history_conditioning=bool(
            metadata.get(
                "history_conditioning",
                metadata.get("schedule", {}).get("history_conditioning", False),
            )
        ),
        use_transformer=bool(
            metadata.get(
                "use_transformer", metadata.get("schedule", {}).get("use_transformer", False)
            )
        ),
        config=config,
    )


# ---------------------------------------------------------------------------
# Torch modules — spatial planner
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
        image = _add_batch_dim(tensors["image"], 4)
        if encoder_features is None:
            if self.standalone_encoder is None:
                raise ValueError("encoder_features are required for the shared-encoder planner.")
            encoder_features = self.standalone_encoder(image)
        spatial_prediction = self.spatial_goal_decoder(encoder_features, target_hw=image.shape[-2:])
        drivable_map = _resolve_drivable_map(tensors) if mask_drivable else None
        decoded = decode_bits_spatial_prediction(
            spatial_prediction=spatial_prediction,
            agent_from_raster=_add_batch_dim(tensors["agent_from_raster"], 3),
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
# Torch modules — bi-level traffic model
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
        self.roi_head = ROIHead(
            global_feature_dim=global_feature_dim,
            agent_feature_dim=agent_feature_dim,
            context_size=context_size,
            roi_feature_size=roi_feature_size,
            roi_layer_key=roi_layer_key,
            feature_channels=self.shared_encoder.feature_channels,
            feature_scales=self.shared_encoder.feature_scales,
        )
        history_feature_dim = 16 if self.history_conditioning else 0
        if self.history_conditioning:
            from .encoder import RNNEncoder

            self.history_encoder = RNNEncoder(
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
        agent_positions, current_states, current_availability = _unpack_scene_states(tensors)
        image = _add_batch_dim(tensors["image"], 4)
        if encoder_features is None:
            encoder_features = self.shared_encoder(image)
        agent_features, global_features = self.roi_head(tensors, agent_positions, encoder_features)
        global_features = global_features[:, None].expand(-1, agent_features.shape[1], -1)
        all_features = torch.cat([agent_features, global_features], dim=-1)
        if self.history_encoder is not None:
            history_positions, history_yaws = _unpack_history_trajectories(tensors, all_features)
            history_trajectory = torch.cat([history_positions, history_yaws], dim=-1)
            history_features = self.history_encoder(
                history_trajectory.reshape(-1, history_trajectory.shape[-2], 3)
            ).reshape(*history_trajectory.shape[:2], -1)
            all_features = torch.cat([all_features, history_features], dim=-1)
        if self.transformer is not None:
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


# ---------------------------------------------------------------------------
# Torch modules — full bi-level model
# ---------------------------------------------------------------------------


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
        image = _add_batch_dim(tensors["image"], 4)
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
            goal_positions = _add_batch_dim(goal_tensors["goal_position"], 2).unsqueeze(1)
            goal_yaws = _add_batch_dim(goal_tensors["goal_yaw"], 2).unsqueeze(1)
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
        drivable = _add_batch_dim(drivable_map, 3).to(device=prob_map.device, dtype=torch.bool)
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

    residual_map = torch.sigmoid(spatial_prediction[:, 1:3])
    yaw_map = spatial_prediction[:, 3:4]
    gather_index = flat_indices[:, None].expand(-1, 2, -1)
    residuals = torch.gather(residual_map.flatten(2), dim=2, index=gather_index).transpose(1, 2)
    yaws = torch.gather(yaw_map.flatten(2), dim=2, index=flat_indices[:, None]).transpose(1, 2)

    pixel_positions = torch.stack([cols, rows], dim=-1).to(residuals.dtype) + residuals
    positions = _homogeneous_transform(pixel_positions, agent_from_raster)
    return {
        "positions": positions,
        "yaws": yaws,
        "scores": scores,
        "location_prob_map": prob_map,
        "pixel_positions": pixel_positions,
    }


# ---------------------------------------------------------------------------
# BitsBehaviorModel — public entry point with checkpoint-aware constructors
# ---------------------------------------------------------------------------


class BitsBehaviorModel(BehaviorModelBase):
    """Public BITS-style behavior model entry point.

    Usage::

        # From a single merged checkpoint:
        model = BitsBehaviorModel.from_checkpoint("path/to/model.ckpt")

        # From a trained planner plus an official predictor:
        model = BitsBehaviorModel.from_trained_planner(
            planner_checkpoint="runtime/bits_planner/epoch_0050.ckpt",
            predictor_checkpoint="checkpoints/bits-3qx90/official_predictor/iter94000_ep6_valLoss0.06.ckpt",
        )
    """

    def __init__(
        self,
        config: Optional[BitsConfig] = None,
        policy: Optional[BitsPolicy] = None,
        builder: Optional[BitsBatchBuilder] = None,
        include_raster: bool = True,
        device=None,
        dtype=None,
    ):
        if policy is None:
            raise ValueError(
                "A BitsPolicy is required. Use from_checkpoint() or "
                "from_trained_planner() to construct a model from weights."
            )
        self.config = config or BitsConfig()
        self.policy = policy
        self.builder = builder or BitsBatchBuilder(self.config)
        self.include_raster = include_raster

    @classmethod
    def from_checkpoint(
        cls, path, *, map_location=None, device=None, dtype=None
    ) -> "BitsBehaviorModel":
        """Load a complete BITS checkpoint into a ready-to-use behavior model.

        The checkpoint must contain both planner and predictor weights along
        with metadata describing the model architecture.
        """
        payload = torch.load(path, map_location=map_location)
        metadata = payload["metadata"]
        config = BitsConfig(**metadata["config"])
        model = _build_bits_model(metadata, config)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        if device is not None:
            model.to(device)
        if dtype is not None:
            model.to(dtype=dtype)

        policy = TorchBitsPolicy(
            model,
            device=device,
            dtype=dtype,
            plan_scorer=BitsPlanScorer(config),
            module_kwargs={
                "use_ground_truth_goal": False,
                "num_samples": None,
                "mask_drivable": False,
            },
        )
        return cls(
            config=config,
            policy=policy,
            builder=BitsBatchBuilder(config),
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_trained_planner(
        cls,
        planner_checkpoint,
        predictor_checkpoint=None,
        *,
        image_channels: int = 14,
        future_steps: int = 20,
        hidden_dim: int = 128,
        model_arch: str = "resnet18",
        context_size: int = 30,
        roi_feature_size: int = 7,
        roi_layer_key: str = "layer2",
        history_conditioning: bool = False,
        use_transformer: bool = False,
        config: Optional[BitsConfig] = None,
        map_location=None,
        device=None,
        dtype=None,
        **extra_arch_kwargs,
    ) -> "BitsBehaviorModel":
        """Load a trained planner checkpoint merged with an official predictor.

        When ``planner_checkpoint`` is a Tactics2D-format checkpoint (saved by
        the behavior training pipeline), its metadata is used to infer
        architecture parameters and override explicit arguments.  When loading
        an official TBSIM checkpoint that has no metadata, explicit ``**``
        arguments serve as fallback.
        """
        resolved_config = config or BitsConfig(future_steps=int(future_steps))

        # --- Try to read architecture from planner metadata ---
        try:
            planner_payload = torch.load(planner_checkpoint, map_location=map_location)
            planner_meta = planner_payload.get("metadata", {})
            image_channels = int(planner_meta.get("image_channels", image_channels))
            future_steps = int(planner_meta.get("future_steps", future_steps))
            hidden_dim = int(
                planner_meta.get(
                    "hidden_dim", planner_meta.get("schedule", {}).get("hidden_dim", hidden_dim)
                )
            )
            model_arch = str(
                planner_meta.get(
                    "model_arch", planner_meta.get("schedule", {}).get("model_arch", model_arch)
                )
            )
            context_size = int(
                planner_meta.get(
                    "context_size",
                    planner_meta.get("schedule", {}).get("context_size", context_size),
                )
            )
            roi_feature_size = int(
                planner_meta.get(
                    "roi_feature_size",
                    planner_meta.get("schedule", {}).get("roi_feature_size", roi_feature_size),
                )
            )
            roi_layer_key = str(
                planner_meta.get(
                    "roi_layer_key",
                    planner_meta.get("schedule", {}).get("roi_layer_key", roi_layer_key),
                )
            )
            history_conditioning = bool(
                planner_meta.get(
                    "history_conditioning",
                    planner_meta.get("schedule", {}).get(
                        "history_conditioning", history_conditioning
                    ),
                )
            )
            use_transformer = bool(
                planner_meta.get(
                    "use_transformer",
                    planner_meta.get("schedule", {}).get("use_transformer", use_transformer),
                )
            )
            resolved_config = BitsConfig(
                **{**dict(planner_meta.get("config", {})), "future_steps": future_steps}
            )
        except Exception:
            pass

        model = BitsBiLevelTorchModel(
            image_channels=int(image_channels),
            future_steps=int(future_steps),
            hidden_dim=int(hidden_dim),
            model_arch=model_arch,
            context_size=int(context_size),
            roi_feature_size=int(roi_feature_size),
            roi_layer_key=roi_layer_key,
            history_conditioning=bool(history_conditioning),
            use_transformer=bool(use_transformer),
            config=resolved_config,
        )

        # --- Merge weights ---
        mapped_state_dict = _extract_planner_state(planner_checkpoint, map_location=map_location)

        if predictor_checkpoint is not None:
            predictor_sd = _state_dict_from_checkpoint(
                predictor_checkpoint, map_location=map_location
            )
            official_sd = _merge_tbsim_state_dicts(
                planner_state_dict=None, predictor_state_dict=predictor_sd
            )
            overlap = set(mapped_state_dict).intersection(official_sd)
            non_encoder_overlap = [
                key
                for key in sorted(overlap)
                if not key.startswith("shared_encoder.encoder_heads.")
            ]
            if non_encoder_overlap:
                raise ValueError(
                    f"Overlapping checkpoint keys (non-encoder): {non_encoder_overlap[:3]}"
                )
            mapped_state_dict.update(official_sd)

        model_state = model.state_dict()
        loadable = {k: mapped_state_dict[k] for k in mapped_state_dict if k in model_state}
        missing = set(model_state) - set(loadable)
        if missing:
            raise ValueError(
                f"Checkpoint is missing {len(missing)} keys from the model. "
                f"First 10: {sorted(missing)[:10]}"
            )
        model_state.update(loadable)
        model.load_state_dict(model_state)
        model.eval()
        if device is not None:
            model.to(device)
        if dtype is not None:
            model.to(dtype=dtype)

        policy = TorchBitsPolicy(
            model,
            device=device,
            dtype=dtype,
            plan_scorer=BitsPlanScorer(resolved_config),
            module_kwargs={"use_ground_truth_goal": False, "num_samples": 8, "mask_drivable": True},
        )
        return cls(
            config=resolved_config,
            policy=policy,
            builder=BitsBatchBuilder(resolved_config),
            device=device,
            dtype=dtype,
        )

    def predict(
        self,
        participants: Dict[object, object],
        map_: Optional[Map],
        frame: int,
        agent_ids: Optional[Iterable[object]] = None,
    ) -> Dict[object, Trajectory]:
        """Predict future trajectories for selected agents."""

        selected_ids = list(participants.keys()) if agent_ids is None else list(agent_ids)
        trajectories = {}
        for ego_id in selected_ids:
            if ego_id not in participants or not participants[ego_id].trajectory.has_state(frame):
                continue

            batch = self.builder.build(
                participants=participants,
                frame=frame,
                ego_id=ego_id,
                map_=map_,
                include_raster=self.include_raster,
            )
            prediction = self.policy.predict_batch(batch)
            trajectories[ego_id] = self._prediction_to_trajectory(ego_id, frame, batch, prediction)
        return trajectories

    def predict_batch(self, batch: BitsBatch) -> BitsPrediction:
        """Expose the policy-level batch API for training/evaluation code."""
        return self.policy.predict_batch(batch)

    def _prediction_to_trajectory(
        self, ego_id: object, frame: int, batch: BitsBatch, prediction: BitsPrediction
    ) -> Trajectory:
        best_index = int(np.argmax(prediction.scores))
        trajectory = Trajectory(id_=ego_id, fps=round(1.0 / self.config.dt, 3), stable_freq=True)
        for step in range(self.config.future_steps):
            if not bool(prediction.availabilities[best_index, step]):
                continue
            local_position = prediction.positions[best_index, step]
            local_yaw = float(prediction.yaws[best_index, step, 0])
            world_position = spatial.transform_point(local_position, batch.world_from_agent)
            world_yaw = spatial.normalize_angle(batch.yaw + local_yaw)
            previous_world_position = self._previous_world_position(
                prediction.positions[best_index], prediction.availabilities[best_index], step, batch
            )
            velocity = (world_position - previous_world_position) / self.config.dt
            state_frame = frame + self.config.step_ms * (step + 1)
            trajectory.add_state(
                State(
                    frame=state_frame,
                    x=float(world_position[0]),
                    y=float(world_position[1]),
                    heading=world_yaw,
                    vx=float(velocity[0]),
                    vy=float(velocity[1]),
                )
            )
        return trajectory

    def _previous_world_position(
        self, positions: np.ndarray, availabilities: np.ndarray, step: int, batch: BitsBatch
    ) -> np.ndarray:
        for previous_step in range(step - 1, -1, -1):
            if bool(availabilities[previous_step]):
                return spatial.transform_point(positions[previous_step], batch.world_from_agent)
        return np.asarray(batch.centroid, dtype=float)


__all__ = [
    "BitsAgentAwareTrajectoryModule",
    "BitsBiLevelTorchModel",
    "BitsSpatialPlannerModule",
    "BitsBehaviorModel",
    "decode_bits_spatial_prediction",
]
