# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Utility helpers for routing."""

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import LineString, Point

from tactics2d.map.element import Lane, Map


def get_lane_centerline(lane: Lane) -> Optional[np.ndarray]:
    """Get a lane centerline as an array of points.

    The parser may provide a centerline through ``custom_tags``. If not
    available, approximate the centerline by averaging lane boundaries.
    """

    if lane.custom_tags is not None and "centerline" in lane.custom_tags:
        centerline = np.asarray(lane.custom_tags["centerline"], dtype=float)
        if centerline.ndim == 2 and centerline.shape[1] == 2 and len(centerline) >= 2:
            return centerline

    if lane.left_side is None or lane.right_side is None:
        return None

    left_coords = np.asarray(lane.left_side.coords, dtype=float)
    right_coords = np.asarray(lane.right_side.coords, dtype=float)
    if len(left_coords) != len(right_coords) or len(left_coords) < 2:
        return None

    return (left_coords + right_coords) / 2.0


def get_lane_length(lane: Lane) -> float:
    """Estimate lane traversal length."""

    centerline = get_lane_centerline(lane)
    if centerline is not None:
        return float(LineString(centerline).length)

    if lane.geometry is not None:
        return float(lane.geometry.length) / 2.0

    return 0.0


def concatenate_centerlines(centerlines: Iterable[np.ndarray]) -> Optional[np.ndarray]:
    """Concatenate route centerlines into a single polyline."""

    merged: List[np.ndarray] = []
    for centerline in centerlines:
        if centerline is None or len(centerline) == 0:
            continue
        if not merged:
            merged.append(centerline.copy())
            continue
        previous = merged[-1]
        if np.allclose(previous[-1], centerline[0]):
            merged.append(centerline[1:].copy())
        else:
            merged.append(centerline.copy())

    if not merged:
        return None

    return np.vstack(merged)


def find_nearest_lane(
    map_: Map, point_xy: Sequence[float], candidate_lane_ids: Optional[Iterable[str]] = None
) -> Optional[str]:
    """Find the nearest lane to a point."""

    point = Point(point_xy[0], point_xy[1])
    best_lane_id = None
    best_distance = np.inf

    if candidate_lane_ids is None:
        lane_ids = list(map_.lanes.keys())
    else:
        lane_ids = list(candidate_lane_ids)

    for lane_id in lane_ids:
        lane = map_.lanes.get(lane_id)
        if lane is None or lane.geometry is None:
            continue

        centerline = get_lane_centerline(lane)
        if centerline is not None:
            distance = LineString(centerline).distance(point)
        else:
            distance = lane.geometry.distance(point)

        if distance < best_distance:
            best_distance = distance
            best_lane_id = lane_id

    return best_lane_id
