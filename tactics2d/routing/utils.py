# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Utility helpers for routing."""

from typing import Iterable, List, Optional, Sequence

import numpy as np
from shapely.geometry import LineString, Point

from tactics2d.map.element import Lane, Map


def get_lane_centerline(lane: Lane) -> Optional[np.ndarray]:
    """Get a lane centerline as an array of points.

    The parser may provide a centerline through ``custom_tags``. If not
    available, approximate the centerline by averaging lane boundaries.
    """

    centerline = lane.centerline()
    if centerline is None:
        return None
    return np.asarray(centerline.coords, dtype=float)


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
    map_: Map,
    point_xy: Sequence[float],
    candidate_lane_ids: Optional[Iterable[str]] = None,
    search_radius: float = 20.0,
) -> Optional[str]:
    """Find the nearest lane to a point.

    When ``candidate_lane_ids`` is not given, the map's spatial index narrows
    the search to lanes whose geometry lies within ``search_radius`` of the
    point; a full scan is used as a fallback when no index is available or no
    lane is that close (so a far-away nearest lane is still found).
    """

    point = Point(point_xy[0], point_xy[1])
    best_lane_id = None
    best_distance = np.inf

    if candidate_lane_ids is not None:
        lane_ids = list(candidate_lane_ids)
    else:
        candidates = map_.query_point(point_xy, buffer=search_radius)
        lane_ids = candidates if candidates else list(map_.lanes.keys())

    for lane_id in lane_ids:
        lane = map_.lanes.get(lane_id)
        if lane is None or lane.geometry is None:
            continue

        centerline = lane.centerline()
        if centerline is not None:
            distance = centerline.distance(point)
        else:
            distance = lane.geometry.distance(point)

        if distance < best_distance:
            best_distance = distance
            best_lane_id = lane_id

    return best_lane_id
