# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Map-element builders for road segment generators."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from shapely.geometry import LineString

from tactics2d.geometry import polyline_length, sample_by_s
from tactics2d.map.element import Junction, Lane, LaneRelationship, RoadLine
from tactics2d.map.generator.rules.module_types import RoadPort, make_port


def boundary_offset(boundary_index: int, lane_num: int, lane_width: float) -> float:
    """Return signed lateral offset of a boundary from the road reference line.

    Args:
        boundary_index: 0 = leftmost boundary, ``lane_num`` = rightmost boundary.
        lane_num: Total number of lanes.
        lane_width: Lane width in metres.

    Returns:
        Signed offset in metres (positive = left of the reference line).
    """
    return lane_num * lane_width / 2.0 - boundary_index * lane_width


def as_id_list(ids: str | int | Iterable[str | int] | None) -> list[str]:
    """Convert one or more ids to a flat list of strings.

    Args:
        ids: A single id, an iterable of ids, or ``None``.

    Returns:
        List of string ids. Returns ``[]`` when ``ids`` is ``None``.
    """
    if ids is None:
        return []

    if isinstance(ids, (str, int)):
        return [str(ids)]

    return [str(id_) for id_ in ids]


def as_linestring(points: np.ndarray | LineString) -> LineString:
    """Convert polyline points to a Shapely LineString.

    Args:
        points: Array of shape ``(N, 2)`` or an existing :class:`LineString`.

    Returns:
        A :class:`LineString` instance.

    Raises:
        ValueError: If ``points`` is not a 2-D array with 2 columns, or
            contains fewer than 2 points.
    """
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
    """Merge two custom-tag dictionaries, with extra tags taking priority.

    Args:
        base_tags: Base tag dict. ``None`` is treated as empty.
        extra_tags: Tags that override any matching keys in ``base_tags``.
            ``None`` is treated as empty.

    Returns:
        New dict containing all entries from both, with ``extra_tags`` winning.
    """
    tags = dict(base_tags or {})

    if extra_tags is not None:
        tags.update(extra_tags)

    return tags


def merge_marking_kwargs(
    marking_kwargs: dict[str, Any], custom_tags: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Merge RoadLine marking kwargs with extra custom tags.

    Args:
        marking_kwargs: Marking style dict as returned by lane-marking rule
            functions. The original is not mutated.
        custom_tags: Additional tags that extend the ``"custom_tags"`` entry of
            ``marking_kwargs``. ``None`` is a no-op.

    Returns:
        New dict with the merged ``"custom_tags"`` entry.
    """
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
    """Link two lanes as predecessor → successor.

    Args:
        predecessor: Lane whose traffic continues into ``successor``.
        successor: Lane that follows ``predecessor``.
    """
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
    """Return lane ids as a tuple of strings.

    Args:
        lanes: Iterable of :class:`Lane` objects.

    Returns:
        Tuple of id strings in iteration order.
    """
    return tuple(lane.id_ for lane in lanes)


def build_pavement_junction(
    id_: str | int,
    shape_points: list[tuple[float, float]],
    *,
    center: np.ndarray | None = None,
    junction_type: str,
    sumo_type: str = "priority",
    custom_tags: dict[str, Any] | None = None,
) -> Junction:
    """Build a pavement-style Junction used as road surface fill.

    Args:
        id_: Junction id.
        shape_points: Polygon vertices as a list of ``(x, y)`` tuples.
        center: Optional junction centre ``(x, y)`` in world coordinates.
        junction_type: Semantic junction type (e.g. ``"intersection"``,
            ``"roundabout"``).
        sumo_type: SUMO junction type string. Defaults to ``"priority"``.
        custom_tags: Additional tags merged into the junction.

    Returns:
        A :class:`Junction` instance with type metadata in ``custom_tags``.
    """
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


def build_segmented_roadline(
    pts: np.ndarray,
    gap_interval: tuple[float, float] | None,
    marking_kwargs: dict,
    id_counter: int,
    step_size: float,
    custom_tags: dict,
    global_s_offset: float = 0.0,
) -> tuple[list[RoadLine], list[str], int]:
    """Build a RoadLine, optionally split around a single gap interval.

    When ``gap_interval`` is ``None`` a single continuous RoadLine is returned.
    When a gap is provided the polyline is split into two segments — one before
    and one after the gap — each built as a separate RoadLine.  Segments shorter
    than ``step_size`` are silently skipped.

    Args:
        pts: Polyline points for the boundary.
        gap_interval: ``(start_s, end_s)`` arc-length range to leave empty, or
            ``None`` for a continuous line.
        marking_kwargs: Marking style dict passed to ``build_roadline_from_points``.
        id_counter: Current id counter; incremented for each created RoadLine.
        step_size: Sampling interval used when slicing segments.
        custom_tags: Extra ``custom_tags`` dict merged into each created RoadLine.
        global_s_offset: Arc-length origin offset added to ``dash_offset`` tags.

    Returns:
        Tuple of ``(roadlines, roadline_ids, updated_id_counter)``.
    """
    total_length = polyline_length(pts)

    if gap_interval is None:
        t = dict(custom_tags)
        t["dash_offset"] = float(global_s_offset)
        roadline = build_roadline_from_points(
            id_=id_counter, points=pts, marking_kwargs=marking_kwargs, custom_tags=t
        )
        return [roadline], [roadline.id_], id_counter + 1

    gap_start = float(np.clip(gap_interval[0], 0.0, total_length))
    gap_end = float(np.clip(gap_interval[1], gap_start, total_length))

    roadlines: list[RoadLine] = []
    roadline_ids: list[str] = []

    for seg_start, seg_end in [(0.0, gap_start), (gap_end, total_length)]:
        seg_len = seg_end - seg_start
        if seg_len <= max(step_size, 1e-3):
            continue

        n = max(2, int(seg_len / max(step_size, 1e-3)) + 1)
        seg_pts, _, _ = sample_by_s(pts, seg_start, seg_end, n)
        t = dict(custom_tags)
        t["dash_offset"] = float(global_s_offset + seg_start)
        roadline = build_roadline_from_points(
            id_=id_counter, points=seg_pts, marking_kwargs=marking_kwargs, custom_tags=t
        )
        id_counter += 1
        roadlines.append(roadline)
        roadline_ids.append(roadline.id_)

    return roadlines, roadline_ids, id_counter


def build_junction_arm_ports(
    normalized_arms: list[dict[str, Any]],
    outgoing_lane_ids: list[list[str]],
    incoming_lane_ids: list[list[str]],
    module_name: str,
) -> dict[str, RoadPort]:
    """Build arm port dict for intersection-style junctions.

    For each arm, creates an inward port (``arm_{i}_in``) carrying outgoing
    lane ids from the junction's perspective, and an outward port
    (``arm_{i}_out``) carrying incoming lane ids.

    Args:
        normalized_arms: Arm records with keys ``point``, ``heading_inward``,
            ``heading_outward``, ``lane_num``, ``lane_width``, ``speed_limit``.
        outgoing_lane_ids: Per-arm list of lane ids that leave the junction
            toward the connected road.
        incoming_lane_ids: Per-arm list of lane ids that enter the junction
            from the connected road.
        module_name: Junction module name, e.g. ``"intersection"`` or
            ``"roundabout"``. Used as the ``kind`` prefix and ``"module"``
            metadata key.

    Returns:
        Dict of named :class:`RoadPort` instances keyed
        ``"arm_{i}_in"`` / ``"arm_{i}_out"``.
    """
    ports: dict[str, RoadPort] = {}

    for arm_idx, arm in enumerate(normalized_arms):
        inward_port = RoadPort(
            point=np.asarray(arm["point"], dtype=float),
            heading=float(arm["heading_inward"]),
            lane_num=int(arm["lane_num"]),
            lane_width=float(arm["lane_width"]),
            speed_limit=float(arm["speed_limit"]),
        )
        outward_port = RoadPort(
            point=np.asarray(arm["point"], dtype=float),
            heading=float(arm["heading_outward"]),
            lane_num=int(arm["lane_num"]),
            lane_width=float(arm["lane_width"]),
            speed_limit=float(arm["speed_limit"]),
        )

        ports[f"arm_{arm_idx}_in"] = make_port(
            inward_port,
            kind=f"{module_name}_in",
            name=f"arm_{arm_idx}_in",
            lane_ids=tuple(outgoing_lane_ids[arm_idx]),
            metadata={
                "module": module_name,
                "arm_index": arm_idx,
                "direction": f"into_{module_name}",
            },
        )
        ports[f"arm_{arm_idx}_out"] = make_port(
            outward_port,
            kind=f"{module_name}_out",
            name=f"arm_{arm_idx}_out",
            lane_ids=tuple(incoming_lane_ids[arm_idx]),
            metadata={
                "module": module_name,
                "arm_index": arm_idx,
                "direction": f"out_of_{module_name}",
            },
        )

    return ports
