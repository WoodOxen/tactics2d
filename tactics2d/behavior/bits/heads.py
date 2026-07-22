# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Goal-conditional policy and future-state prediction heads."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .config import BitsConfig
from .mlp import MLP, TrajectoryDecoder


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
        self.goal_encoder = MLP(3, goal_feature_dim, output_activation=nn.ReLU)
        self.ego_decoder = TrajectoryDecoder(
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
        self.agents_decoder = TrajectoryDecoder(
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
