# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Routing utilities."""

from typing import Optional

from shapely.geometry import Point


def find_nearest_lane(map_, location) -> Optional[str]:
    """Return the id of the lane closest to *location*, or None if no lanes.

    This is a lightweight fallback used by batch-cache builders; for
    production-quality lane matching use ``tactics2d.routing.router``.
    """
    if map_ is None or not map_.lanes:
        return None

    point = Point(location[0], location[1])
    best_lane_id = None
    best_dist = float("inf")

    for lane_id, lane in map_.lanes.items():
        centerline = lane.centerline()
        if centerline is None:
            geometry = getattr(lane, "geometry", None)
            if geometry is None:
                continue
            dist = geometry.distance(point)
        else:
            dist = centerline.distance(point)
        if dist < best_dist:
            best_dist = dist
            best_lane_id = lane_id

    return best_lane_id
