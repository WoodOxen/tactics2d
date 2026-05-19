# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Evaluation helpers for LimSim-style behavior planning."""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import Polygon

from tactics2d.geometry import euclidean_distance
from tactics2d.participant.trajectory import State, Trajectory

from .schema import PlanningResult


@dataclass
class TrajectoryError:
    """Displacement error between a planned trajectory and a reference trajectory."""

    ade: float
    fde: float
    samples: int


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


def dimensions_from_participants(participants: Dict[object, object]) -> Dict[object, Tuple[float, float]]:
    """Extract ``(length, width)`` from Tactics2D participants."""

    dimensions = {}
    for agent_id, participant in participants.items():
        length = getattr(participant, "length", None) or 4.8
        width = getattr(participant, "width", None) or 1.9
        dimensions[agent_id] = (float(length), float(width))
    return dimensions


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
            footprints.append((agent_id, state_footprint(state, length, width)))

        for i, (source_id, source_shape) in enumerate(footprints):
            for target_id, target_shape in footprints[i + 1 :]:
                if source_shape.intersects(target_shape):
                    return frame, source_id, target_id
    return None


def displacement_errors(
    planned: Dict[object, Trajectory],
    reference: Dict[object, Trajectory],
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
            distances.append(euclidean_distance(planned_state.location, reference_state.location))

        errors[agent_id] = TrajectoryError(
            ade=float(np.mean(distances)),
            fde=float(distances[-1]),
            samples=len(distances),
        )
    return errors


def state_footprint(state: State, length: float, width: float) -> Polygon:
    """Build an oriented rectangular footprint for a Tactics2D trajectory state."""

    half_length = length / 2.0
    half_width = width / 2.0
    cos_yaw = np.cos(state.heading)
    sin_yaw = np.sin(state.heading)
    corners = [
        (half_length, half_width),
        (half_length, -half_width),
        (-half_length, -half_width),
        (-half_length, half_width),
    ]
    points = [
        (
            state.x + local_x * cos_yaw - local_y * sin_yaw,
            state.y + local_x * sin_yaw + local_y * cos_yaw,
        )
        for local_x, local_y in corners
    ]
    return Polygon(points)
