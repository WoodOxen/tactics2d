# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared data schemas for LimSim-style interaction planning."""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from tactics2d.geometry import spatial
from tactics2d.participant.trajectory import State, Trajectory

from .action import LimSimAction


@dataclass
class TrajectoryError:
    """Displacement error between a planned trajectory and a reference trajectory."""

    ade: float
    fde: float
    samples: int


def find_first_collision(
    trajectories: Dict[object, Trajectory],
    dimensions: Optional[Dict[object, Tuple[float, float]]] = None,
) -> Optional[Tuple[int, object, object]]:
    """Return the first colliding frame and agent ids, if any."""
    agent_ids = sorted(trajectories)
    if len(agent_ids) < 2:
        return None
    frame_sets = [set(trajectories[agent_id].frames) for agent_id in agent_ids]
    common_frames = sorted(set.intersection(*frame_sets)) if frame_sets else []
    for frame in common_frames:
        footprints = []
        for agent_id in agent_ids:
            length, width = (dimensions or {}).get(agent_id, (4.8, 1.9))
            state = trajectories[agent_id].get_state(frame)
            footprints.append(
                (agent_id, spatial.oriented_box(state.x, state.y, state.heading, length, width))
            )
        for i, (source_id, source_shape) in enumerate(footprints):
            for target_id, target_shape in footprints[i + 1 :]:
                if source_shape.intersects(target_shape):
                    return frame, source_id, target_id
    return None


def displacement_errors(
    planned: Dict[object, Trajectory], reference: Dict[object, Trajectory]
) -> Dict[object, TrajectoryError]:
    """Compute ADE/FDE on frames shared by planned and reference trajectories."""
    errors = {}
    for agent_id, trajectory in planned.items():
        reference_trajectory = reference.get(agent_id)
        if reference_trajectory is None:
            continue
        common_frames = sorted(set(trajectory.frames) & set(reference_trajectory.frames))
        if not common_frames:
            continue
        distances = []
        for frame in common_frames:
            planned_state = trajectory.get_state(frame)
            reference_state = reference_trajectory.get_state(frame)
            distances.append(
                spatial.euclidean_distance(planned_state.location, reference_state.location)
            )
        errors[agent_id] = TrajectoryError(
            ade=float(np.mean(distances)), fde=float(distances[-1]), samples=len(distances)
        )
    return errors


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
        """Return a new instance with the given fields replaced.

        Uses direct construction instead of :func:`dataclasses.replace` to
        avoid the intermediate dict allocation on every call (thousands per
        planning cycle).
        """
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


@dataclass
class LimSimEvaluation:
    """Compact metrics for one behavior planning result."""

    action_counts: Dict[str, int] = field(default_factory=dict)
    has_collision: bool = False
    first_collision: Optional[Tuple[int, object, object]] = None
    mean_ade: Optional[float] = None
    mean_fde: Optional[float] = None
    trajectory_errors: Dict[object, TrajectoryError] = field(default_factory=dict)


def evaluate_planning_result(
    result: PlanningResult,
    reference_trajectories: Optional[Dict[object, Trajectory]] = None,
    dimensions: Optional[Dict[object, Tuple[float, float]]] = None,
) -> LimSimEvaluation:
    """Evaluate actions, collisions, and optional displacement error."""

    action_counts = Counter(action.value for action in result.actions.values())
    first_collision = find_first_collision(result.trajectories, dimensions)
    trajectory_errors = {}
    if reference_trajectories is not None:
        trajectory_errors = displacement_errors(result.trajectories, reference_trajectories)

    mean_ade = None
    mean_fde = None
    if trajectory_errors:
        mean_ade = float(np.mean([error.ade for error in trajectory_errors.values()]))
        mean_fde = float(np.mean([error.fde for error in trajectory_errors.values()]))

    return LimSimEvaluation(
        action_counts=dict(action_counts),
        has_collision=first_collision is not None,
        first_collision=first_collision,
        mean_ade=mean_ade,
        mean_fde=mean_fde,
        trajectory_errors=trajectory_errors,
    )
