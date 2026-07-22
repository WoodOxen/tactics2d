# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""LSTM-based trajectory encoder for history-conditioned BITS models."""

import torch
import torch.nn as nn

from .mlp import MLP


class RNNEncoder(nn.Module):
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
        self.mlp = MLP(
            input_dim=int(rnn_hidden_size),
            output_dim=int(feature_dim),
            layer_dims=mlp_layer_dims,
            output_activation=nn.ReLU,
        )

    def forward(self, input_trajectory: torch.Tensor) -> torch.Tensor:
        trajectory_feature = self.lstm(input_trajectory)[0][:, -1, :]
        return self.mlp(trajectory_feature)
