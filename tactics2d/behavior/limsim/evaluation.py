# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Evaluation helpers for LimSim-style behavior planning."""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon

from tactics2d.geometry import euclidean_distance, oriented_box
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


@dataclass
class RollingEvaluation:
    """Compact metrics for a rolling LimSim-style simulation."""

    action_counts: Dict[str, int] = field(default_factory=dict)
    min_distance: float = float("inf")
    collision_count: int = 0
    first_collision: Optional[Tuple[int, object, object]] = None
    action_switch_count: int = 0
    memory_hit_count: int = 0
    roi_sizes: Tuple[int, ...] = ()
    background_sizes: Tuple[int, ...] = ()


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


def evaluate_rolling_result(
    rolling_result,
    dimensions: Optional[Dict[object, Tuple[float, float]]] = None,
) -> RollingEvaluation:
    """Evaluate a rolling PDP simulation with safety and continuity metrics."""

    action_counts = Counter(
        action.value for result in rolling_result.results for action in result.actions.values()
    )
    min_distance, collision_count, first_collision = rolling_distance_and_collisions(
        rolling_result.participants,
        rolling_result.frames,
        dimensions,
    )
    return RollingEvaluation(
        action_counts=dict(action_counts),
        min_distance=min_distance,
        collision_count=collision_count,
        first_collision=first_collision,
        action_switch_count=count_action_switches(rolling_result.results),
        memory_hit_count=count_memory_hits(rolling_result),
        roi_sizes=tuple(len(result.roi_agent_ids) for result in rolling_result.results),
        background_sizes=tuple(len(result.background_agent_ids) for result in rolling_result.results),
    )


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


def count_action_switches(results) -> int:
    """Count high-level action changes for agents observed in consecutive PDP rounds."""

    previous_actions = {}
    switch_count = 0
    for result in results:
        for agent_id, action in result.actions.items():
            previous = previous_actions.get(agent_id)
            if previous is not None and previous != action:
                switch_count += 1
            previous_actions[agent_id] = action
    return switch_count


def count_memory_hits(rolling_result) -> int:
    """Count predictions that exactly reuse a previous round's remaining trajectory."""

    memory_hits = 0
    for index in range(1, len(rolling_result.predicted_trajectories)):
        frame = rolling_result.frames[index]
        previous_result = rolling_result.results[index - 1]
        predictions = rolling_result.predicted_trajectories[index]
        for agent_id, prediction in predictions.items():
            previous_trajectory = previous_result.trajectories.get(agent_id)
            if previous_trajectory is None:
                continue
            remaining_frames = [
                state_frame for state_frame in previous_trajectory.frames if state_frame > frame
            ]
            if prediction.frames == remaining_frames:
                memory_hits += 1
    return memory_hits


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
            ade=float(np.mean(distances)), fde=float(distances[-1]), samples=len(distances)
        )
    return errors


def state_footprint(state: State, length: float, width: float) -> Polygon:
    """Build an oriented rectangular footprint for a Tactics2D trajectory state."""

    return oriented_box(state.x, state.y, state.heading, length, width)
