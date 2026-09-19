# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared data schemas for LimSim-style interaction planning."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from tactics2d.geometry import spatial
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

    @property
    def footprint(self):
        """Return the oriented bounding box of this agent."""
        return spatial.oriented_box(self.x, self.y, self.heading, self.length, self.width)

    def with_updates(self, **kwargs) -> "AgentDecisionState":
        """Return a new instance with the given fields replaced."""
        return AgentDecisionState(
            agent_id=kwargs.pop("agent_id", self.agent_id),
            x=kwargs.pop("x", self.x),
            y=kwargs.pop("y", self.y),
            heading=kwargs.pop("heading", self.heading),
            speed=kwargs.pop("speed", self.speed),
            lane_id=kwargs.pop("lane_id", self.lane_id),
            lateral_offset=kwargs.pop("lateral_offset", self.lateral_offset),
            route_lane_ids=kwargs.pop("route_lane_ids", self.route_lane_ids),
            route_progress=kwargs.pop("route_progress", self.route_progress),
            length=kwargs.pop("length", self.length),
            width=kwargs.pop("width", self.width),
            action=kwargs.pop("action", self.action),
        )


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


@dataclass
class LimSimRollingResult:
    """Output of a receding-horizon replay of one vehicle."""

    ego_id: object = None
    # Frames to animate, in milliseconds: history up to the take-over, then the replayed future.
    frames: List[int] = field(default_factory=list)
    # The replayed vehicle's track: recorded history plus the committed future,
    # on the recorded frame grid.
    trajectory: Optional[Trajectory] = None
    # Per planning frame, the plan issued there as ``(frame, x, y)`` waypoints.
    plans: Dict[int, List[Tuple[int, float, float]]] = field(default_factory=dict)
    cycles: int = 0
    # Every vehicle re-simulated, the ego included. A caller measuring the closed
    # loop should read the committed futures back out of ``participants``.
    controlled_ids: List[object] = field(default_factory=list)


def states_to_trajectory(
    agent_id: object, states: List[AgentDecisionState], start_frame: int, dt: float
):
    """Convert predicted decision states to a Tactics2D trajectory."""

    trajectory = Trajectory(id_=agent_id, fps=round(1.0 / dt, 3), stable_freq=True)
    for index, state in enumerate(states):
        frame = int(round(start_frame + (index + 1) * dt * 1000))
        heading = spatial.normalize_angle(state.heading)
        vx = state.speed * np.cos(heading)
        vy = state.speed * np.sin(heading)
        trajectory.add_state(
            State(frame=frame, x=state.x, y=state.y, heading=heading, vx=vx, vy=vy)
        )
    return trajectory
