# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared trajectory evaluation helpers for behavior models."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon

from tactics2d.geometry import euclidean_distance, oriented_box
from tactics2d.participant.trajectory import State, Trajectory


@dataclass
class TrajectoryError:
    """Displacement error between a planned trajectory and a reference trajectory."""

    ade: float
    fde: float
    samples: int


def dimensions_from_participants(
    participants: Dict[object, object],
) -> Dict[object, Tuple[float, float]]:
    """Extract ``(length, width)`` from Tactics2D participants."""

    dimensions = {}
    for agent_id, participant in participants.items():
        length = getattr(participant, "length", None) or 4.8
        width = getattr(participant, "width", None) or 1.9
        dimensions[agent_id] = (float(length), float(width))
    return dimensions


def rolling_distance_and_collisions(
    participants: Dict[object, object],
    frames,
    dimensions: Optional[Dict[object, Tuple[float, float]]] = None,
) -> Tuple[float, int, Optional[Tuple[int, object, object]]]:
    """Compute minimum center distance and footprint collisions over rolling frames."""

    min_distance = float("inf")
    collision_count = 0
    first_collision = None
    agent_ids = list(participants)
    for frame in frames:
        active = [
            agent_id
            for agent_id in agent_ids
            if participants[agent_id].trajectory.has_state(frame)
        ]
        for index, source_id in enumerate(active):
            source_state = participants[source_id].trajectory.get_state(frame)
            for target_id in active[index + 1 :]:
                target_state = participants[target_id].trajectory.get_state(frame)
                min_distance = min(
                    min_distance,
                    euclidean_distance(source_state.location, target_state.location),
                )

                source_length, source_width = (dimensions or {}).get(source_id, (4.8, 1.9))
                target_length, target_width = (dimensions or {}).get(target_id, (4.8, 1.9))
                source_shape = state_footprint(source_state, source_length, source_width)
                target_shape = state_footprint(target_state, target_length, target_width)
                if source_shape.intersects(target_shape):
                    collision_count += 1
                    if first_collision is None:
                        first_collision = (frame, source_id, target_id)
    return min_distance, collision_count, first_collision


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
            distances.append(euclidean_distance(planned_state.location, reference_state.location))

        errors[agent_id] = TrajectoryError(
            ade=float(np.mean(distances)),
            fde=float(distances[-1]),
            samples=len(distances),
        )
    return errors


def state_footprint(state: State, length: float, width: float) -> Polygon:
    """Build an oriented rectangular footprint for a Tactics2D trajectory state."""

    return oriented_box(state.x, state.y, state.heading, length, width)
