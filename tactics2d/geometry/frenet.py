# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Frenet coordinate helpers for reference-path based planning."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from shapely.geometry import LineString, Point

from .spatial import normalize_angle


@dataclass(frozen=True)
class FrenetPoint:
    """A point represented in reference-path Frenet coordinates."""

    s: float
    d: float


class ReferencePath:
    """A route reference path with Cartesian/Frenet conversion helpers."""

    def __init__(self, path: LineString, lane_ids: Tuple[str, ...] = (), lane_width: float = 0.0):
        self.path = path
        self.lane_ids = lane_ids
        self.lane_width = lane_width

    def cartesian_to_frenet(self, x: float, y: float) -> FrenetPoint:
        """Project a Cartesian point to Frenet coordinates on this reference path."""

        point = Point(x, y)
        s = float(self.path.project(point))
        ref_point = self.path.interpolate(s)
        heading = self.heading_at(s)
        dx = x - ref_point.x
        dy = y - ref_point.y
        d = float(-dx * np.sin(heading) + dy * np.cos(heading))
        return FrenetPoint(s=s, d=d)

    def frenet_to_cartesian(self, s: float, d: float) -> Tuple[float, float, float]:
        """Convert Frenet coordinates to Cartesian pose on this reference path."""

        s = float(np.clip(s, 0.0, self.path.length))
        point = self.path.interpolate(s)
        heading = self.heading_at(s)
        x = float(point.x - d * np.sin(heading))
        y = float(point.y + d * np.cos(heading))
        return x, y, heading

    def heading_at(self, s: float) -> float:
        """Return the local tangent heading along this reference path."""

        s0 = float(np.clip(s, 0.0, self.path.length))
        ahead = self.path.interpolate(min(s0 + 0.5, self.path.length))
        behind = self.path.interpolate(max(s0 - 0.5, 0.0))
        return normalize_angle(np.arctan2(ahead.y - behind.y, ahead.x - behind.x))


def align_path_with_heading(
    path_array: np.ndarray, x: float, y: float, heading: float
) -> np.ndarray:
    """Ensure a polyline path direction is consistent with a given heading.

    Projects the reference point *(x, y)* onto the path, samples the local
    tangent direction, and reverses the path array if the tangent opposes
    *heading*.

    Args:
        path_array: Array of path points with shape ``(N, 2)``.
        x: Reference x-coordinate.
        y: Reference y-coordinate.
        heading: Desired heading in radians.

    Returns:
        The path array, reversed if the local tangent opposes *heading*.
    """
    line = LineString(path_array)
    progress = float(line.project(Point(x, y)))
    point = line.interpolate(progress)
    ahead = line.interpolate(min(progress + 0.5, line.length))
    if ahead.distance(point) < 1e-6:
        ahead = point
        point = line.interpolate(max(progress - 0.5, 0.0))
    path_heading = normalize_angle(np.arctan2(ahead.y - point.y, ahead.x - point.x))
    if np.cos(normalize_angle(path_heading - heading)) < 0.0:
        return path_array[::-1].copy()
    return path_array
