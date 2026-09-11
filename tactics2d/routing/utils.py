# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Routing utilities."""

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import LineString, Point

from tactics2d.geometry import spatial
from tactics2d.map.element import Lane, Map

_MAX_GAP_SCAN_LANES = 2000


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


def find_lane_at_pose(
    map_: Map,
    x: float,
    y: float,
    heading: float,
    radius: float = 8.0,
    heading_tolerance_deg: float = 60.0,
) -> Optional[Tuple[object, float]]:
    """Match a pose to the closest heading-consistent lane.

    Unlike :func:`find_nearest_lane`, this respects the driving direction: the
    closest lane whose centerline passes within ``radius`` and whose local
    tangent agrees with ``heading`` within the tolerance is returned. This
    avoids snapping a forward-driving agent onto a same-corridor lane that
    points the other way.

    Returns:
        A tuple ``(lane_id, s0)`` with ``s0`` the arc offset of the pose along
        the lane centerline, or ``None`` when no lane matches.
    """

    if map_ is None:
        return None
    tolerance = heading_tolerance_deg * np.pi / 180.0
    best = None
    best_distance = radius
    query = Point(x, y)
    for lane_id, lane in map_.lanes.items():
        projection = lane.project_point(query)
        if projection is None:
            continue
        if projection.distance > best_distance:
            continue
        yaw_error = abs(spatial.normalize_angle(heading - projection.heading))
        if yaw_error > tolerance:
            continue
        best_distance = projection.distance
        best = (lane_id, projection.s)
    return best


def geometric_successor_links(
    map_: Map,
    max_gap: float = 2.5,
    max_heading_diff_deg: float = 45.0,
) -> List[Tuple[object, object]]:
    """Return ``(src_id, dst_id)`` lane links implied by geometry.

    A lane whose centerline ends within ``max_gap`` of another lane's start
    while keeping a heading-continuous tangent is linked forward even when the
    source topology omits the successor (common at intersection boundaries in
    exported maps). Explicit successors are never duplicated. The scan is
    quadratic and therefore skipped above a bounded lane count.
    """

    valid = []
    for lane_id, lane in map_.lanes.items():
        centerline = lane.centerline()
        if centerline is None or len(centerline.coords) < 2:
            continue
        points = np.asarray(centerline.coords, dtype=float)
        tangents = points[1:] - points[:-1]
        start_heading = np.arctan2(tangents[0][1], tangents[0][0])
        end_heading = np.arctan2(tangents[-1][1], tangents[-1][0])
        valid.append((lane_id, points[0], points[-1], start_heading, end_heading))
    if len(valid) > _MAX_GAP_SCAN_LANES or len(valid) < 2:
        return []

    ids = [item[0] for item in valid]
    starts = np.asarray([item[1] for item in valid], dtype=float)
    ends = np.asarray([item[2] for item in valid], dtype=float)
    head_start = np.asarray([item[3] for item in valid], dtype=float)
    head_end = np.asarray([item[4] for item in valid], dtype=float)

    distance = np.linalg.norm(ends[:, None, :] - starts[None, :, :], axis=2)
    diff = head_end[:, None] - head_start[None, :]
    aligned = np.abs(np.arctan2(np.sin(diff), np.cos(diff))) <= np.radians(max_heading_diff_deg)
    close = distance <= max_gap
    np.fill_diagonal(close, False)

    links = []
    for src_index in range(len(valid)):
        if not np.any(close[src_index]):
            continue
        src_id = ids[src_index]
        successors = map_.lanes[src_id].successors
        for dst_index in np.nonzero(close[src_index] & aligned[src_index])[0]:
            dst_id = ids[dst_index]
            if dst_id in successors:
                continue
            links.append((src_id, dst_id))
    return links


def augment_lane_successors(
    map_: Map,
    max_gap: float = 2.5,
    max_heading_diff_deg: float = 45.0,
) -> Map:
    """Add geometric gap successors onto map lanes in place.

    This is the map-level counterpart of :func:`geometric_successor_links`: the
    inferred links are written back into each lane's ``successors`` so that
    lane-following consumers (reference-path chains, routing) keep moving past
    connectivity gaps instead of stopping at a broken chain. Idempotent.
    """

    for src_id, dst_id in geometric_successor_links(map_, max_gap, max_heading_diff_deg):
        map_.lanes[src_id].successors.add(dst_id)
    return map_
