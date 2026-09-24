# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Adapted from SMART (github.com/rainmaker22/SMART), Apache-2.0.

"""Token types consumed by SMART encoders and decoders."""

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional

import numpy as np
import torch


class AgentType(IntEnum):
    """Motion codebook type index of a modelled participant."""

    VEHICLE = 0
    PEDESTRIAN = 1
    CYCLIST = 2


class PolygonType(IntEnum):
    """Map codebook polygon-type index, sized for the map decoder's ``Embedding(4)``."""

    VEHICLE = 0
    BIKE = 1
    BUS = 2
    PEDESTRIAN = 3


class PointType(IntEnum):
    """Map codebook point-type index, sized for the map decoder's ``Embedding(17)``.

    The order is load-bearing for the pretrained checkpoint and must never be
    renumbered.
    """

    DASH_SOLID_YELLOW = 0
    DASH_SOLID_WHITE = 1
    DASHED_WHITE = 2
    DASHED_YELLOW = 3
    DOUBLE_SOLID_YELLOW = 4
    DOUBLE_SOLID_WHITE = 5
    DOUBLE_DASH_YELLOW = 6
    DOUBLE_DASH_WHITE = 7
    SOLID_YELLOW = 8
    SOLID_WHITE = 9
    SOLID_DASH_WHITE = 10
    SOLID_DASH_YELLOW = 11
    EDGE = 12
    NONE = 13
    UNKNOWN = 14
    CROSSWALK = 15
    CENTERLINE = 16


class LightType(IntEnum):
    """Traffic-light state index, sized for the map decoder's ``Embedding(4)``."""

    STOP = 0
    GO = 1
    CAUTION = 2
    UNKNOWN = 3


@dataclass
class SmartAgentTokens:
    """Tokenized motion history for the jointly modelled agents.

    Tensors are shaped ``(A, S)``; each slot covers ``shift + 1`` history frames
    and overlaps the previous slot by one frame.

    Attributes:
        agent_ids (List[object]): Participant id of each tensor row.
        token_idx (torch.Tensor): Motion codebook index, shape ``(A, S)``, dtype ``int64``.
        token_pos (torch.Tensor): Contour centroid, shape ``(A, S, 2)``.
        token_heading (torch.Tensor): Heading from corner 0 to corner 3, shape ``(A, S)``.
        token_velocity (torch.Tensor): Finite-difference velocity, shape ``(A, S, 2)``.
        agent_valid_mask (torch.Tensor): Slot availability, shape ``(A, S)``, dtype ``bool``.
        agent_type (np.ndarray): Codebook type index per agent, shape ``(A,)``.
        agent_shape (torch.Tensor): Agent extent at the newest frame, shape ``(A, 3)``.
        eval_mask (torch.Tensor): Newest-frame observation flag, shape ``(A,)``, dtype ``bool``.
    """

    agent_ids: List[object]
    token_idx: torch.Tensor
    token_pos: torch.Tensor
    token_heading: torch.Tensor
    token_velocity: torch.Tensor
    agent_valid_mask: torch.Tensor
    agent_type: np.ndarray
    agent_shape: torch.Tensor
    eval_mask: torch.Tensor

    @property
    def num_agents(self) -> int:
        """Return the number of modelled agents."""

        return int(self.token_idx.shape[0])

    @property
    def num_slots(self) -> int:
        """Return the number of motion-token slots."""

        return int(self.token_idx.shape[1])


@dataclass
class SmartMapTokens:
    """Tokenized map polylines consumed by the map decoder.

    The first dimension is the number of point tokens ``N``, one per tokenized
    polyline window.

    Attributes:
        pt_position (torch.Tensor): Window start point, shape ``(N, 3)`` with a zero z-column.
        pt_orientation (torch.Tensor): Polyline heading, shape ``(N,)``.
        pt_token_idx (torch.Tensor): Map codebook index, shape ``(N,)``, dtype ``int64``.
        pt_type (torch.Tensor): Point-type index into the 17-class table, shape ``(N,)``.
        pl_type (torch.Tensor): Polygon-type index into the 4-class table, shape ``(N,)``.
        pt_side (torch.Tensor): Lane side index, shape ``(N,)``.
        light_type (torch.Tensor): Traffic-light state index, shape ``(N,)``.
        traj_pos (torch.Tensor): Resampled window points, shape ``(N, points_per_token, 2)``.
        traj_theta (torch.Tensor): Heading of each window, shape ``(N,)``.
        token2pl (torch.Tensor): ``(2, N)`` point-token to source-polyline index.
    """

    pt_position: torch.Tensor
    pt_orientation: torch.Tensor
    pt_token_idx: torch.Tensor
    pt_type: torch.Tensor
    pl_type: torch.Tensor
    pt_side: torch.Tensor
    light_type: torch.Tensor
    traj_pos: torch.Tensor
    traj_theta: torch.Tensor
    token2pl: torch.Tensor

    @property
    def num_tokens(self) -> int:
        """Return the number of point tokens."""

        return int(self.pt_token_idx.shape[0])


@dataclass
class SmartTokenBatch:
    """One scenario assembled into the model's heterogeneous input.

    Attributes:
        agents (Optional[SmartAgentTokens]): Tokenized agent history. Defaults to None.
        map_tokens (Optional[SmartMapTokens]): Tokenized map. Defaults to None.
        frame_ms (int): Timestamp of the newest observed frame. Defaults to 0.
    """

    agents: Optional[SmartAgentTokens] = None
    map_tokens: Optional[SmartMapTokens] = None
    frame_ms: int = 0
