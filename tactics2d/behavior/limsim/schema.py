# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared data schemas for LimSim-style interaction planning."""

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from tactics2d.participant.trajectory import State, Trajectory

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

    def with_updates(self, **kwargs) -> "AgentDecisionState":
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


@dataclass
class PlanningResult:
    """Output of the LimSim behavior model."""

    trajectories: Dict[object, Trajectory] = field(default_factory=dict)
    actions: Dict[object, LimSimAction] = field(default_factory=dict)
    groups: List[List[object]] = field(default_factory=list)
    root_nodes: Dict[Tuple[object, ...], object] = field(default_factory=dict)
    roi_agent_ids: List[object] = field(default_factory=list)
    background_agent_ids: List[object] = field(default_factory=list)


def states_to_trajectory(agent_id: object, states: List[AgentDecisionState], start_frame: int, dt: float):
    """Convert predicted decision states to a Tactics2D trajectory."""

    trajectory = Trajectory(id_=agent_id, fps=round(1.0 / dt, 3), stable_freq=True)
    for index, state in enumerate(states):
        frame = int(round(start_frame + (index + 1) * dt * 1000))
        vx = state.speed * np.cos(state.heading)
        vy = state.speed * np.sin(state.heading)
        trajectory.add_state(
            State(frame=frame, x=state.x, y=state.y, heading=state.heading, vx=vx, vy=vy)
        )
    return trajectory
