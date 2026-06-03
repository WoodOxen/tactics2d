# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Evaluation helpers for BITS-style rolling simulation."""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from shapely.geometry import LinearRing, Point, Polygon

from tactics2d.behavior.trajectory_evaluation import (
    TrajectoryError,
    dimensions_from_participants,
    displacement_errors,
    rolling_distance_and_collisions,
)
from tactics2d.map.element import Map

from .rolling import BitsRollingResult


@dataclass(frozen=True)
class BitsRollingEvaluation:
    """Compact metrics for a rolling BITS simulation."""

    frame_count: int
    prediction_round_count: int
    min_distance: float
    collision_count: int
    first_collision: Optional[tuple]
    off_drivable_count: int = 0
    off_drivable_rate: float = 0.0
    first_off_drivable: Optional[tuple] = None
    mean_ade: Optional[float] = None
    mean_fde: Optional[float] = None
    trajectory_errors: Dict[object, TrajectoryError] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "frame_count": self.frame_count,
            "prediction_round_count": self.prediction_round_count,
            "min_distance": self.min_distance,
            "collision_count": self.collision_count,
            "first_collision": self.first_collision,
            "off_drivable_count": self.off_drivable_count,
            "off_drivable_rate": self.off_drivable_rate,
            "first_off_drivable": self.first_off_drivable,
            "mean_ade": self.mean_ade,
            "mean_fde": self.mean_fde,
            "trajectory_errors": self.trajectory_errors,
        }


def evaluate_bits_rolling_result(
    rolling_result: BitsRollingResult,
    reference_participants: Optional[Dict[object, object]] = None,
    dimensions: Optional[Dict[object, tuple]] = None,
    map_: Optional[Map] = None,
) -> BitsRollingEvaluation:
    """Evaluate a rolling BITS simulation with safety and optional log error metrics."""

    resolved_dimensions = dimensions or dimensions_from_participants(rolling_result.participants)
    min_distance, collision_count, first_collision = rolling_distance_and_collisions(
        rolling_result.participants,
        rolling_result.frames,
        resolved_dimensions,
    )
    trajectory_errors = {}
    if reference_participants is not None:
        reference_trajectories = {
            agent_id: participant.trajectory
            for agent_id, participant in reference_participants.items()
        }
        simulated_trajectories = {
            agent_id: participant.trajectory
            for agent_id, participant in rolling_result.participants.items()
        }
        trajectory_errors = displacement_errors(simulated_trajectories, reference_trajectories)

    mean_ade = None
    mean_fde = None
    if trajectory_errors:
        mean_ade = float(np.mean([error.ade for error in trajectory_errors.values()]))
        mean_fde = float(np.mean([error.fde for error in trajectory_errors.values()]))
    off_drivable_count, off_drivable_rate, first_off_drivable = (
        rolling_drivable_violations(rolling_result, map_) if map_ is not None else (0, 0.0, None)
    )

    return BitsRollingEvaluation(
        frame_count=len(rolling_result.frames),
        prediction_round_count=len(rolling_result.predicted_trajectories),
        min_distance=min_distance,
        collision_count=collision_count,
        first_collision=first_collision,
        off_drivable_count=off_drivable_count,
        off_drivable_rate=off_drivable_rate,
        first_off_drivable=first_off_drivable,
        mean_ade=mean_ade,
        mean_fde=mean_fde,
        trajectory_errors=trajectory_errors,
    )


def rolling_drivable_violations(
    rolling_result: BitsRollingResult,
    map_: Map,
) -> tuple:
    """Count states whose center point is outside the drivable map geometry."""

    drivable_geometries = _drivable_geometries(map_)
    if not drivable_geometries:
        return 0, 0.0, None

    total = 0
    violations = 0
    first_violation = None
    for frame in rolling_result.frames:
        for agent_id, participant in rolling_result.participants.items():
            if not participant.trajectory.has_state(frame):
                continue
            total += 1
            state = participant.trajectory.get_state(frame)
            point = Point(state.location)
            if any(geometry.covers(point) for geometry in drivable_geometries):
                continue
            violations += 1
            if first_violation is None:
                first_violation = (frame, agent_id)

    rate = float(violations / total) if total else 0.0
    return violations, rate, first_violation


def _drivable_geometries(map_: Map) -> list:
    geometries = []
    for lane in map_.lanes.values():
        geometry = _as_drivable_polygon(getattr(lane, "geometry", None))
        if geometry is not None:
            geometries.append(geometry)

    drivable_subtypes = {"drivable_area", "parking", "road_segment"}
    for area in map_.areas.values():
        if getattr(area, "subtype", None) not in drivable_subtypes:
            continue
        geometry = _as_drivable_polygon(getattr(area, "geometry", None))
        if geometry is not None:
            geometries.append(geometry)
    return geometries


def _as_drivable_polygon(geometry):
    if geometry is None:
        return None
    if isinstance(geometry, Polygon):
        return geometry
    if isinstance(geometry, LinearRing):
        return Polygon(geometry)
    if getattr(geometry, "geom_type", None) == "LinearRing":
        return Polygon(geometry)
    return geometry
