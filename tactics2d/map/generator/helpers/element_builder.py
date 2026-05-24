# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Map-element builders for map generators."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from shapely.geometry import LineString

from tactics2d.geometry import curvature_stats, has_self_intersection, polyline_length
from tactics2d.map.element import Junction, Lane, LaneRelationship, RoadLine


def as_id_list(ids: str | int | Iterable[str | int] | None) -> list[str]:
    """Convert one or more ids to a list of strings."""
    if ids is None:
        return []

    if isinstance(ids, (str, int)):
        return [str(ids)]

    return [str(id_) for id_ in ids]


def as_linestring(points: np.ndarray | LineString) -> LineString:
    """Convert polyline points to a LineString."""
    if isinstance(points, LineString):
        return points

    points = np.asarray(points, dtype=float)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2), got {points.shape}.")
    if len(points) < 2:
        raise ValueError("points must contain at least two points.")

    return LineString(points)


def merge_custom_tags(
    base_tags: dict[str, Any] | None = None, extra_tags: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Merge two custom-tag dictionaries."""
    tags = dict(base_tags or {})

    if extra_tags is not None:
        tags.update(extra_tags)

    return tags


def merge_marking_kwargs(
    marking_kwargs: dict[str, Any], custom_tags: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Merge RoadLine marking kwargs with extra custom tags."""
    kwargs = dict(marking_kwargs)
    kwargs["custom_tags"] = merge_custom_tags(kwargs.get("custom_tags"), custom_tags)
    return kwargs


def build_roadline_from_points(
    id_: str | int,
    points: np.ndarray | LineString,
    marking_kwargs: dict[str, Any] | None = None,
    custom_tags: dict[str, Any] | None = None,
    **roadline_kwargs,
) -> RoadLine:
    """Build a RoadLine from polyline points.

    Args:
        id_: RoadLine id.
        points: Polyline points or an existing LineString.
        marking_kwargs: Keyword arguments from lane-marking rules.
        custom_tags: Extra custom tags merged into marking kwargs.
        roadline_kwargs: Extra RoadLine constructor keyword arguments.

    Returns:
        A RoadLine instance.
    """
    kwargs = dict(marking_kwargs or {})
    kwargs.update(roadline_kwargs)
    kwargs["custom_tags"] = merge_custom_tags(kwargs.get("custom_tags"), custom_tags)

    return RoadLine(id_=str(id_), geometry=as_linestring(points), **kwargs)


def build_optional_roadline_from_points(
    id_counter: int,
    points: np.ndarray | LineString,
    marking_kwargs: dict[str, Any] | None = None,
    custom_tags: dict[str, Any] | None = None,
    min_points: int = 2,
    **roadline_kwargs,
) -> tuple[RoadLine | None, int]:
    """Build a RoadLine when enough points exist.

    This is useful for fork/merge/ramp cases where a visible roadline segment
    may disappear after clipping.

    Args:
        id_counter: Current id counter.
        points: Polyline points or LineString.
        marking_kwargs: Keyword arguments from lane-marking rules.
        custom_tags: Extra custom tags.
        min_points: Minimum number of points required.
        roadline_kwargs: Extra RoadLine constructor keyword arguments.

    Returns:
        A tuple ``(roadline, next_id_counter)``.
    """
    if isinstance(points, LineString):
        point_num = len(points.coords)
    else:
        point_num = len(points)

    if point_num < min_points:
        return None, id_counter

    roadline = build_roadline_from_points(
        id_=id_counter,
        points=points,
        marking_kwargs=marking_kwargs,
        custom_tags=custom_tags,
        **roadline_kwargs,
    )

    return roadline, id_counter + 1


def build_lane_from_boundaries(
    id_: str | int,
    left_points: np.ndarray | LineString,
    right_points: np.ndarray | LineString,
    left_roadline_ids: str | int | Iterable[str | int] | None,
    right_roadline_ids: str | int | Iterable[str | int] | None,
    *,
    speed_limit: float | None = None,
    speed_limit_unit: str = "km/h",
    subtype: str = "road",
    custom_tags: dict[str, Any] | None = None,
    **lane_kwargs,
) -> Lane:
    """Build a Lane from left and right boundary points.

    ``Lane`` already builds its polygon geometry from ``left_side`` and
    ``right_side``, so road generators should not manually assemble lane
    polygons here.

    Args:
        id_: Lane id.
        left_points: Left boundary points or LineString.
        right_points: Right boundary points or LineString.
        left_roadline_ids: RoadLine ids on the left side.
        right_roadline_ids: RoadLine ids on the right side.
        speed_limit: Lane speed limit.
        speed_limit_unit: Speed-limit unit.
        subtype: Lane subtype.
        custom_tags: Lane custom tags.
        lane_kwargs: Extra Lane constructor keyword arguments.

    Returns:
        A Lane instance.
    """
    return Lane(
        id_=str(id_),
        left_side=as_linestring(left_points),
        right_side=as_linestring(right_points),
        subtype=subtype,
        speed_limit=speed_limit,
        speed_limit_unit=speed_limit_unit,
        line_ids={"left": as_id_list(left_roadline_ids), "right": as_id_list(right_roadline_ids)},
        custom_tags=dict(custom_tags or {}),
        **lane_kwargs,
    )


def link_lanes(predecessor: Lane, successor: Lane) -> None:
    """Link two lanes as predecessor and successor."""
    predecessor.add_related_lane(successor.id_, LaneRelationship.SUCCESSOR)
    successor.add_related_lane(predecessor.id_, LaneRelationship.PREDECESSOR)


def add_ordered_lane_neighbors(
    lanes: list[Lane],
    *,
    left_relationship: LaneRelationship = LaneRelationship.LEFT_NEIGHBOR,
    right_relationship: LaneRelationship = LaneRelationship.RIGHT_NEIGHBOR,
) -> None:
    """Add neighbor relationships for ordered lanes.

    Args:
        lanes: Lanes ordered from left to right in their driving direction.
        left_relationship: Relationship assigned to the previous lane.
        right_relationship: Relationship assigned to the next lane.
    """
    for i, lane in enumerate(lanes):
        if i > 0:
            lane.add_related_lane(lanes[i - 1].id_, left_relationship)
        if i < len(lanes) - 1:
            lane.add_related_lane(lanes[i + 1].id_, right_relationship)


def lane_ids(lanes: Iterable[Lane]) -> tuple[str, ...]:
    """Return lane ids as a tuple."""
    return tuple(lane.id_ for lane in lanes)


def build_module_quality(
    module: str,
    centerline: np.ndarray,
    *,
    accepted_reason_name: str = "self_intersection",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build common quality metadata for a single-reference-line module."""
    self_intersection = has_self_intersection(centerline)
    quality = {
        "module": module,
        "length": polyline_length(centerline),
        "self_intersection": self_intersection,
        "accepted_reasons": [accepted_reason_name] if self_intersection else [],
        "accepted": not self_intersection,
        **curvature_stats(centerline),
    }

    if extra is not None:
        quality.update(extra)

    return quality


def build_pavement_junction(
    id_: str | int,
    shape_points,
    *,
    center: np.ndarray | None = None,
    junction_type: str,
    sumo_type: str = "priority",
    custom_tags: dict[str, Any] | None = None,
) -> Junction:
    """Build a pavement-style Junction used as road surface fill."""
    tags = {
        "sumo_id": f"junction_{id_}",
        "type": "road",
        "junction_type": junction_type,
        "sumo_type": sumo_type,
        "shape": shape_points,
    }

    if center is not None:
        center = np.asarray(center, dtype=float)
        if center.shape != (2,):
            raise ValueError(f"center must have shape (2,), got {center.shape}.")
        tags["x"] = str(float(center[0]))
        tags["y"] = str(float(center[1]))

    tags.update(custom_tags or {})

    return Junction(id_=str(id_), custom_tags=tags)
