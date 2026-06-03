# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Evaluation helpers for LimSim-style behavior planning."""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from tactics2d.behavior.trajectory_evaluation import (
    TrajectoryError,
    dimensions_from_participants,
    displacement_errors,
    find_first_collision,
    rolling_distance_and_collisions,
    state_footprint,
)
from tactics2d.participant.trajectory import Trajectory

from .schema import PlanningResult

__all__ = [
    "LimSimEvaluation",
    "RollingEvaluation",
    "TrajectoryError",
    "count_action_switches",
    "count_memory_hits",
    "dimensions_from_participants",
    "displacement_errors",
    "evaluate_planning_result",
    "evaluate_rolling_result",
    "find_first_collision",
    "rolling_distance_and_collisions",
    "state_footprint",
]


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
    rolling_result, dimensions: Optional[Dict[object, Tuple[float, float]]] = None
) -> RollingEvaluation:
    """Evaluate a rolling PDP simulation with safety and continuity metrics."""

    action_counts = Counter(
        action.value for result in rolling_result.results for action in result.actions.values()
    )
    min_distance, collision_count, first_collision = rolling_distance_and_collisions(
        rolling_result.participants, rolling_result.frames, dimensions
    )
    return RollingEvaluation(
        action_counts=dict(action_counts),
        min_distance=min_distance,
        collision_count=collision_count,
        first_collision=first_collision,
        action_switch_count=count_action_switches(rolling_result.results),
        memory_hit_count=count_memory_hits(rolling_result),
        roi_sizes=tuple(len(result.roi_agent_ids) for result in rolling_result.results),
        background_sizes=tuple(
            len(result.background_agent_ids) for result in rolling_result.results
        ),
    )


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
