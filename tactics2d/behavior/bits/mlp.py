# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""MLP and trajectory decoder building blocks."""

from collections import OrderedDict
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .config import BitsConfig
from .dynamics import integrate_unicycle_controls


class MLP(nn.Module):
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


class SplitMLP(MLP):
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


class TrajectoryDecoder(nn.Module):
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
        self.mlp = SplitMLP(
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
