# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""LimSim decision-search state and trajectory conversion."""

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from tactics2d.behavior.trajectory_processing import trajectory_from_poses
from tactics2d.geometry import spatial

from .action import LimSimAction


@dataclass(frozen=True)
class AgentDecisionState:
    """Compact state for one simulated traffic participant."""

    agent_id: object
    x: float
    y: float
    heading: float
    speed: float
    lane_id: Optional[str] = None
    lateral_offset: float = 0.0
    route_lane_ids: Tuple[str, ...] = ()
    route_progress: float = 0.0
    length: float = 4.8
    width: float = 1.9
    action: LimSimAction = LimSimAction.KS

    @property
    def location(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @property
    def footprint(self):
        """Return the oriented bounding box of this agent."""
        return spatial.oriented_box(self.x, self.y, self.heading, self.length, self.width)

    def with_updates(self, **kwargs) -> "AgentDecisionState":
        """Return a new instance with the given fields replaced."""
        return replace(self, **kwargs)


@dataclass(frozen=True)
class JointDecisionState:
    """Joint state used as a node payload in MCTS."""

    agents: Tuple[AgentDecisionState, ...]
    depth: int = 0
    trajectories: Tuple[Tuple[AgentDecisionState, ...], ...] = ()

    @property
    def agent_ids(self) -> Tuple[object, ...]:
        return tuple(agent.agent_id for agent in self.agents)

    def trajectory_dict(self) -> Dict[object, List[AgentDecisionState]]:
        """Return accumulated rollout states keyed by agent id."""

        result = {}
        for agent, states in zip(self.agents, self.trajectories):
            result[agent.agent_id] = list(states)
        return result


def states_to_trajectory(agent_id, states, start_frame: int, dt: float):
    """Convert predicted decision states to a native trajectory."""

    frames = [int(round(start_frame + (index + 1) * dt * 1000)) for index in range(len(states))]
    positions = np.asarray([(state.x, state.y) for state in states], dtype=float)
    headings = np.asarray([spatial.normalize_angle(state.heading) for state in states])
    return trajectory_from_poses(
        agent_id,
        positions,
        headings,
        frames,
        speeds=[state.speed for state in states],
        step_ms=int(round(dt * 1000)),
    )
