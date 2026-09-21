# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Adapted from SMART (github.com/rainmaker22/SMART), Apache-2.0.

"""Inference data types for SMART-style joint behavior generation."""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional

import numpy as np
import torch

from tactics2d.participant.trajectory import State, Trajectory


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


@dataclass(frozen=True)
class SmartPrediction:
    """A joint SMART rollout for every modelled agent, in the world frame.

    Arrays are indexed ``[agent, step]`` in ``agent_ids`` order; the first
    predicted state sits at ``frame_ms0 + step_ms``.

    Attributes:
        agent_ids (List[object]): Modelled agent ids, in tensor-row order.
        positions (np.ndarray): World-frame positions, shape ``(A, K, 2)``.
        headings (np.ndarray): World-frame headings in radians, shape ``(A, K)``.
        availabilities (np.ndarray): Per-step validity, shape ``(A, K)``.
        frame_ms0 (int): Timestamp the rollout starts from.
        step_ms (int): Interval between consecutive predicted states.
        frames (Optional[List[int]]): Timestamp of each predicted step. Defaults
            to None, the uniform lattice implied by ``frame_ms0`` and ``step_ms``.
    """

    agent_ids: List[object]
    positions: np.ndarray
    headings: np.ndarray
    availabilities: np.ndarray
    frame_ms0: int
    step_ms: int
    frames: Optional[List[int]] = None

    def trajectory(self, agent_id: object) -> Trajectory:
        """Return one agent's rollout as a ``Trajectory``.

        Args:
            agent_id (object): A modelled agent id present in this rollout.

        Returns:
            The agent's predicted trajectory with one state per step.

        Raises:
            KeyError: If *agent_id* was not part of the joint rollout.
        """

        row = self.agent_ids.index(agent_id)
        frames = self.frames
        if frames is None:
            frames = [
                self.frame_ms0 + self.step_ms * (step + 1)
                for step in range(self.positions.shape[1])
            ]
        return build_trajectory(
            agent_id,
            self.positions[row],
            self.headings[row],
            self.availabilities[row],
            frames,
            self.step_ms,
        )


@dataclass
class SmartRollingResult:
    """Per-scenario closed-loop outcome and its collision metrics.

    Attributes:
        front_collisions (int): Head-on collision count. Defaults to 0.
        side_collisions (int): Lateral collision count. Defaults to 0.
        rear_collisions (int): Rear-end collision count. Defaults to 0.
        progress (float): Summed per-step displacement of the controlled agents, in metres.
        total_agents_controlled (int): Number of agents stepped jointly. Defaults to 0.
        collided (bool): Whether any collision occurred. Defaults to False.
        ego_id (object): The scenario's centre agent. Defaults to None.
        modelled_ids (List[object]): Agents modelled in this scenario.
        poses (Dict[object, np.ndarray]): Per-agent ``(steps, 4)`` poses; ``-1`` is unoccupied.
    """

    front_collisions: int = 0
    side_collisions: int = 0
    rear_collisions: int = 0
    progress: float = 0.0
    total_agents_controlled: int = 0
    collided: bool = False
    ego_id: object = None
    modelled_ids: List[object] = field(default_factory=list)
    poses: Dict[object, np.ndarray] = field(default_factory=dict)


def build_trajectory(
    agent_id: object,
    positions: np.ndarray,
    headings: np.ndarray,
    availabilities: np.ndarray,
    frames: List[int],
    step_ms: int,
) -> Trajectory:
    """Convert per-step poses into a ``Trajectory``.

    Args:
        agent_id (object): Identifier recorded on the trajectory.
        positions (np.ndarray): Positions of shape ``(K, 2)``.
        headings (np.ndarray): Headings in radians, shape ``(K,)``.
        availabilities (np.ndarray): Per-step validity, shape ``(K,)``.
        frames (List[int]): Timestamp of each step, in milliseconds.
        step_ms (int): Interval between consecutive steps, in milliseconds.

    Returns:
        A trajectory holding one state per valid step.
    """

    trajectory = Trajectory(id_=agent_id, fps=round(1000.0 / step_ms, 3), stable_freq=True)
    for step, frame_ms in enumerate(frames):
        if not bool(availabilities[step]):
            continue
        heading = float(headings[step])
        step_speed = 0.0
        if step > 0:
            step_speed = float(
                np.hypot(
                    positions[step, 0] - positions[step - 1, 0],
                    positions[step, 1] - positions[step - 1, 1],
                )
            ) / (step_ms / 1000.0)
        trajectory.add_state(
            State(
                frame=int(frame_ms),
                x=float(positions[step, 0]),
                y=float(positions[step, 1]),
                heading=heading,
                vx=step_speed * np.cos(heading),
                vy=step_speed * np.sin(heading),
            )
        )
    return trajectory
