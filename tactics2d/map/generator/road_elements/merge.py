# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Socket-driven merge road generator."""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Point

from tactics2d.map.element import Lane, LaneRelationship, RoadLine

from ..geometry.geometry_utils import offset_polyline
from ..geometry.module_geometry import (
    curvature_stats,
    fit_reference_line,
    has_self_intersection,
    polyline_length,
)
from ..rules.lane_marking_rules import one_way_mark, ramp_mark, roadline_render_kwargs
from ..rules.module_types import RoadModuleResult, RoadPort, make_port, ports_to_interfaces


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _polyline_cumulative_s(points: np.ndarray) -> np.ndarray:
    """Return cumulative arc length of a polyline."""
    points = np.asarray(points, dtype=float)

    if len(points) < 2:
        return np.zeros(len(points), dtype=float)

    seg_lens = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_lens)])


def _interp_at_s(points: np.ndarray, cum_s: np.ndarray, s: float) -> np.ndarray:
    """Interpolate a point at arc-length coordinate s."""
    points = np.asarray(points, dtype=float)

    if len(points) < 2:
        return points[0]

    s = float(np.clip(s, 0.0, cum_s[-1]))
    idx = int(np.searchsorted(cum_s, s, side="right") - 1)
    idx = max(0, min(idx, len(points) - 2))

    s0 = cum_s[idx]
    s1 = cum_s[idx + 1]
    denom = max(s1 - s0, 1e-9)
    t = (s - s0) / denom

    return points[idx] * (1.0 - t) + points[idx + 1] * t


def _sample_polyline(points: np.ndarray, s: float) -> np.ndarray:
    """Sample a point on a polyline by arc length."""
    cum_s = _polyline_cumulative_s(points)
    return _interp_at_s(points, cum_s, s)


def _point_heading_at_s(points: np.ndarray, s: float) -> tuple[np.ndarray, float]:
    """Sample point and heading at arc-length coordinate s."""
    points = np.asarray(points, dtype=float)

    if len(points) < 2:
        return points[0], 0.0

    cum_s = _polyline_cumulative_s(points)
    total = cum_s[-1]
    s = float(np.clip(s, 0.0, total))

    idx = int(np.searchsorted(cum_s, s, side="right") - 1)
    idx = max(0, min(idx, len(points) - 2))

    point = _interp_at_s(points, cum_s, s)
    tangent = points[idx + 1] - points[idx]
    heading = float(np.arctan2(tangent[1], tangent[0]))

    return point, heading


def _project_s_to_polyline(points: np.ndarray, query: np.ndarray) -> float:
    """Project a point to a polyline and return the arc-length coordinate."""
    points = np.asarray(points, dtype=float)
    query = np.asarray(query, dtype=float)

    if len(points) < 2:
        return 0.0

    cum_s = _polyline_cumulative_s(points)
    best_dist = float("inf")
    best_s = 0.0

    for i in range(len(points) - 1):
        a = points[i]
        b = points[i + 1]
        ab = b - a
        denom = float(np.dot(ab, ab))

        if denom < 1e-9:
            continue

        t = float(np.clip(np.dot(query - a, ab) / denom, 0.0, 1.0))
        proj = a + t * ab
        dist = float(np.linalg.norm(query - proj))

        if dist < best_dist:
            best_dist = dist
            best_s = cum_s[i] + t * (cum_s[i + 1] - cum_s[i])

    return best_s


def _resample_polyline(points: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline to n points."""
    points = np.asarray(points, dtype=float)

    if len(points) == 0:
        return np.empty((0, 2), dtype=float)
    if len(points) == 1 or n <= 1:
        return points[:1]

    cum_s = _polyline_cumulative_s(points)
    total = cum_s[-1]

    if total < 1e-9:
        return np.repeat(points[:1], n, axis=0)

    samples = np.linspace(0.0, total, n)
    return np.asarray([_interp_at_s(points, cum_s, s) for s in samples], dtype=float)


def _boundary_offset(boundary_index: int, lane_num: int, lane_width: float) -> float:
    """Return boundary offset from road reference line."""
    half_width = lane_num * lane_width / 2.0
    return half_width - boundary_index * lane_width


def _boundary_token(boundary_index: int, boundary_num: int) -> str:
    """Return active-standard token for a main one-way road boundary."""
    lane_num = boundary_num - 1

    if boundary_index == 0:
        return one_way_mark(0, lane_num, "left")

    if boundary_index == boundary_num - 1:
        return one_way_mark(lane_num - 1, lane_num, "right")

    return one_way_mark(boundary_index - 1, lane_num, "right")


def _branch_boundary_token(boundary_index: int, boundary_num: int, merge_side: str) -> str:
    """Return active-standard token for a merge branch/ramp boundary.

    Branch boundaries are indexed from left to right in branch driving direction.
    MUTCD:
      - left ramp edge: yellow edge
      - right ramp edge: white ramp edge
      - interior ramp dividers: dashed white ramp
    GB mapping is handled by lane_marking_rules.ramp_mark().
    """
    if boundary_index == 0:
        return ramp_mark("left_edge")

    if boundary_index == boundary_num - 1:
        return ramp_mark("right_edge")

    return ramp_mark("interior")


def _make_roadline(
    id_counter: int, points: np.ndarray, token: str, custom_tags: dict
) -> tuple[RoadLine | None, int]:
    """Create a RoadLine if the geometry has at least two points."""
    if len(points) < 2:
        return None, id_counter

    roadline = RoadLine(
        id_=str(id_counter),
        geometry=LineString(points),
        **roadline_render_kwargs(token, custom_tags),
    )

    return roadline, id_counter + 1


def _choose_merge_s(
    main_center: np.ndarray, branch_point: np.ndarray, taper_length: float, branch_length: float
) -> float:
    """Choose the branch merge section on the main reference line."""
    total = polyline_length(main_center)
    projected_s = _project_s_to_polyline(main_center, branch_point)

    advance = max(float(taper_length), float(branch_length) * 0.45)

    lower = total * 0.18
    upper = total * 0.85

    if upper <= lower:
        return total * 0.5

    return float(np.clip(projected_s + advance, lower, upper))


def _target_lane_indices(main_lane_num: int, branch_lane_num: int, merge_side: str) -> list[int]:
    """Return main-lane indices targeted by the branch."""
    if branch_lane_num > main_lane_num:
        raise ValueError("branch_lane_num cannot exceed main_lane_num in merge v1.")

    if merge_side == "right":
        start = main_lane_num - branch_lane_num
        return list(range(start, main_lane_num))

    return list(range(branch_lane_num))


def _target_boundary_indices(
    main_lane_num: int, branch_lane_num: int, merge_side: str
) -> list[int]:
    """Return main-boundary indices occupied by the merging branch section."""
    if branch_lane_num > main_lane_num:
        raise ValueError("branch_lane_num cannot exceed main_lane_num in merge v1.")

    if merge_side == "right":
        start = main_lane_num - branch_lane_num
        return list(range(start, main_lane_num + 1))

    return list(range(0, branch_lane_num + 1))


def _branch_end_from_outer_boundary(
    main_boundaries: list[np.ndarray],
    target_boundaries: list[int],
    merge_s: float,
    main_length: float,
    merge_side: str,
    branch_n: int,
    lane_w: float,
    merge_heading: float,
) -> np.ndarray:
    """Return the branch reference-line end point."""
    outer_idx = target_boundaries[-1] if merge_side == "right" else target_boundaries[0]
    boundary = main_boundaries[outer_idx]
    boundary_total = polyline_length(boundary)

    if main_length < 1e-9:
        boundary_s = 0.0
    else:
        boundary_s = merge_s / main_length * boundary_total

    outer_pt = _sample_polyline(boundary, boundary_s)

    if merge_side == "right":
        inward_normal = np.array([-np.sin(merge_heading), np.cos(merge_heading)], dtype=float)
    else:
        inward_normal = np.array([np.sin(merge_heading), -np.cos(merge_heading)], dtype=float)

    return outer_pt + inward_normal * (branch_n * lane_w / 2.0)


def _intersection_points(geometry) -> list[Point]:
    """Extract representative Point objects from a Shapely intersection geometry."""
    if geometry.is_empty:
        return []

    geom_type = geometry.geom_type

    if geom_type == "Point":
        return [geometry]

    if geom_type == "MultiPoint":
        return [point for point in geometry.geoms]

    if geom_type == "LineString":
        coords = list(geometry.coords)
        if len(coords) == 0:
            return []
        return [Point(coords[0]), Point(coords[-1])]

    if geom_type == "MultiLineString":
        points: list[Point] = []
        for line in geometry.geoms:
            coords = list(line.coords)
            if len(coords) > 0:
                points.append(Point(coords[0]))
                points.append(Point(coords[-1]))
        return points

    if geom_type == "GeometryCollection":
        points: list[Point] = []
        for sub_geometry in geometry.geoms:
            points.extend(_intersection_points(sub_geometry))
        return points

    return []


def _find_intersection_point(
    line1: LineString, line2: LineString, *, pick: str = "last_on_line1"
) -> Point | None:
    """Find a representative Point intersection between two LineStrings."""
    points = _intersection_points(line1.intersection(line2))

    if len(points) == 0:
        return None

    if pick == "first_on_line1":
        return min(points, key=lambda pt: float(line1.project(pt)))
    if pick == "last_on_line1":
        return max(points, key=lambda pt: float(line1.project(pt)))
    if pick == "first_on_line2":
        return min(points, key=lambda pt: float(line2.project(pt)))
    if pick == "last_on_line2":
        return max(points, key=lambda pt: float(line2.project(pt)))

    return points[0]


def _cut_polyline(line: LineString, pt: Point, keep: str) -> np.ndarray:
    """Truncate a polyline precisely at a given Point."""
    coords = np.array(line.coords, dtype=float)
    dist = float(line.project(pt))

    dists = np.zeros(len(coords), dtype=float)
    dists[1:] = np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))

    idx = int(np.searchsorted(dists, dist))
    if idx == 0:
        idx = 1

    pt_coords = np.array([[pt.x, pt.y]], dtype=float)

    if keep == "before":
        return np.vstack([coords[:idx], pt_coords])
    return np.vstack([pt_coords, coords[idx:]])


def _branch_centerlines_from_boundaries(branch_boundaries: list[np.ndarray]) -> list[np.ndarray]:
    """Build approximate branch centerlines from adjacent boundary pairs."""
    centerlines: list[np.ndarray] = []

    for i in range(len(branch_boundaries) - 1):
        left = branch_boundaries[i]
        right = branch_boundaries[i + 1]
        n = max(2, min(len(left), len(right)))
        left_r = _resample_polyline(left, n)
        right_r = _resample_polyline(right, n)
        centerlines.append((left_r + right_r) * 0.5)

    return centerlines


def merge(
    main_in: RoadPort,
    branch_in: RoadPort,
    main_out: RoadPort,
    *,
    merge_side: str = "right",
    main_lane_num: int | None = None,
    branch_lane_num: int | None = None,
    lane_width: float | None = None,
    speed_limit: float | None = None,
    taper_length: float = 65.0,
    branch_length: float = 85.0,
    merge_s_ratio: float | None = None,
    step_size: float = 0.1,
    id_offset: int = 0,
) -> RoadModuleResult:
    """Generate a lane-level merge module."""
    if merge_side not in ("left", "right"):
        raise ValueError("merge_side must be 'left' or 'right'.")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive.")
    if taper_length <= 0.0:
        raise ValueError("taper_length must be positive.")
    if branch_length <= 0.0:
        raise ValueError("branch_length must be positive.")

    main_n = int(main_lane_num if main_lane_num is not None else main_in.lane_num)
    branch_n = int(branch_lane_num if branch_lane_num is not None else branch_in.lane_num)
    lane_w = float(lane_width if lane_width is not None else main_in.lane_width)
    speed = float(speed_limit if speed_limit is not None else main_in.speed_limit)

    if main_n < 1:
        raise ValueError("main_lane_num must be >= 1.")
    if branch_n < 1:
        raise ValueError("branch_lane_num must be >= 1.")
    if branch_n > main_n:
        raise ValueError("merge v1 requires branch_lane_num <= main_lane_num.")
    if lane_w <= 0.0:
        raise ValueError("lane_width must be positive.")

    main_center = fit_reference_line(
        main_in.point, main_in.heading, main_out.point, main_out.heading, step_size
    )

    main_length = polyline_length(main_center)
    if merge_s_ratio is not None:
        merge_s = float(np.clip(merge_s_ratio, 0.0, 1.0)) * main_length
    else:
        merge_s = _choose_merge_s(
            main_center, np.asarray(branch_in.point, dtype=float), taper_length, branch_length
        )
    merge_point, merge_heading = _point_heading_at_s(main_center, merge_s)

    main_boundaries: list[np.ndarray] = []
    for boundary_idx in range(main_n + 1):
        main_boundaries.append(
            offset_polyline(main_center, _boundary_offset(boundary_idx, main_n, lane_w))
        )

    target_lanes = _target_lane_indices(main_n, branch_n, merge_side)
    target_boundaries = _target_boundary_indices(main_n, branch_n, merge_side)

    branch_end = _branch_end_from_outer_boundary(
        main_boundaries=main_boundaries,
        target_boundaries=target_boundaries,
        merge_s=merge_s,
        main_length=main_length,
        merge_side=merge_side,
        branch_n=branch_n,
        lane_w=lane_w,
        merge_heading=merge_heading,
    )

    _branch_chord = branch_end - np.asarray(branch_in.point, dtype=float)
    _branch_chord_len = float(np.linalg.norm(_branch_chord))
    if _branch_chord_len > 1e-6:
        branch_depart_heading = float(np.arctan2(_branch_chord[1], _branch_chord[0]))
    else:
        branch_depart_heading = float(branch_in.heading)

    branch_center = fit_reference_line(
        np.asarray(branch_in.point, dtype=float),
        branch_depart_heading,
        branch_end,
        merge_heading,
        step_size,
    )

    branch_boundaries: list[np.ndarray] = []
    for local_boundary_idx in range(branch_n + 1):
        branch_boundaries.append(
            offset_polyline(branch_center, _boundary_offset(local_boundary_idx, branch_n, lane_w))
        )

    if merge_side == "right":
        main_outside_boundary_idx = main_n
        branch_outside_boundary_idx = branch_n
        branch_inside_boundary_idx = 0
    else:
        main_outside_boundary_idx = 0
        branch_outside_boundary_idx = 0
        branch_inside_boundary_idx = branch_n

    main_outside_boundary = main_boundaries[main_outside_boundary_idx]
    main_out_line = LineString(main_outside_boundary)

    branch_cut_lines: list[np.ndarray] = []
    nose_pt: Point | None = None
    merge_pt: Point | None = None

    for local_idx, b_pts in enumerate(branch_boundaries):
        b_line = LineString(b_pts)
        intersect = _find_intersection_point(b_line, main_out_line, pick="last_on_line1")

        if intersect is None:
            branch_cut_lines.append(b_pts)
        else:
            branch_cut_lines.append(_cut_polyline(b_line, intersect, "before"))
            if local_idx == branch_inside_boundary_idx:
                nose_pt = intersect
            if local_idx == branch_outside_boundary_idx:
                merge_pt = intersect

    main_before_pts = main_outside_boundary
    main_after_pts = np.empty((0, 2), dtype=float)
    main_outside_gap_start_s = 0.0
    main_outside_gap_end_s = 0.0

    if nose_pt is not None and merge_pt is not None:
        s_nose = float(main_out_line.project(nose_pt))
        s_merge = float(main_out_line.project(merge_pt))

        if s_merge < s_nose:
            pt_first = merge_pt
            pt_second = nose_pt
            main_outside_gap_start_s = s_merge
            main_outside_gap_end_s = s_nose
        else:
            pt_first = nose_pt
            pt_second = merge_pt
            main_outside_gap_start_s = s_nose
            main_outside_gap_end_s = s_merge

        main_before_pts = _cut_polyline(main_out_line, pt_first, "before")
        main_rest = _cut_polyline(main_out_line, pt_first, "after")
        main_after_pts = _cut_polyline(LineString(main_rest), pt_second, "after")

    branch_marking_end_s = 0.0
    if nose_pt is not None:
        branch_marking_end_s = float(
            LineString(branch_boundaries[branch_inside_boundary_idx]).project(nose_pt)
        )

    branch_centerlines = _branch_centerlines_from_boundaries(branch_boundaries)

    lanes: list[Lane] = []
    roadlines: list[RoadLine] = []
    id_counter = id_offset

    main_boundary_line_ids: list[list[str]] = []

    for boundary_idx, boundary_pts in enumerate(main_boundaries):
        token = _boundary_token(boundary_idx, main_n + 1)
        ids: list[str] = []

        if boundary_idx == main_outside_boundary_idx:
            before_roadline, id_counter = _make_roadline(
                id_counter,
                main_before_pts,
                token,
                {
                    "module": "merge",
                    "submodule": "main",
                    "boundary_index": boundary_idx,
                    "merge_side": merge_side,
                    "segment": "before_branch_opening",
                    "opening_hidden": True,
                },
            )
            if before_roadline is not None:
                ids.append(before_roadline.id_)
                roadlines.append(before_roadline)

            after_roadline, id_counter = _make_roadline(
                id_counter,
                main_after_pts,
                token,
                {
                    "module": "merge",
                    "submodule": "main",
                    "boundary_index": boundary_idx,
                    "merge_side": merge_side,
                    "segment": "after_branch_opening",
                    "opening_hidden": True,
                },
            )
            if after_roadline is not None:
                ids.append(after_roadline.id_)
                roadlines.append(after_roadline)

        else:
            roadline, id_counter = _make_roadline(
                id_counter,
                boundary_pts,
                token,
                {
                    "module": "merge",
                    "submodule": "main",
                    "boundary_index": boundary_idx,
                    "merge_side": merge_side,
                    "kept_on_main": True,
                },
            )
            if roadline is not None:
                ids.append(roadline.id_)
                roadlines.append(roadline)

        main_boundary_line_ids.append(ids)

    main_lanes: list[Lane] = []

    for lane_idx in range(main_n):
        left_line = LineString(main_boundaries[lane_idx])
        right_line = LineString(main_boundaries[lane_idx + 1])

        lane = Lane(
            id_=str(id_counter),
            left_side=left_line,
            right_side=right_line,
            subtype="road",
            speed_limit=speed,
            speed_limit_unit="km/h",
            line_ids={
                "left": main_boundary_line_ids[lane_idx],
                "right": main_boundary_line_ids[lane_idx + 1],
            },
            custom_tags={
                "module": "merge",
                "submodule": "main",
                "lane_index": lane_idx,
                "merge_side": merge_side,
            },
        )
        id_counter += 1

        lanes.append(lane)
        main_lanes.append(lane)

    for i, lane in enumerate(main_lanes):
        if i > 0:
            lane.add_related_lane(main_lanes[i - 1].id_, LaneRelationship.LEFT_NEIGHBOR)
        if i < len(main_lanes) - 1:
            lane.add_related_lane(main_lanes[i + 1].id_, LaneRelationship.RIGHT_NEIGHBOR)

    branch_boundary_line_ids: list[list[str]] = []

    for local_idx, boundary_pts in enumerate(branch_boundaries):
        token = _branch_boundary_token(local_idx, branch_n + 1, merge_side)

        if local_idx == branch_outside_boundary_idx:
            visibility_rule = "outside_edge_until_merge"
        elif local_idx == branch_inside_boundary_idx:
            visibility_rule = "inside_edge_until_merge"
        else:
            visibility_rule = "interior_divider_until_midpoint"

        visible_boundary = branch_cut_lines[local_idx]

        roadline, id_counter = _make_roadline(
            id_counter,
            visible_boundary,
            token,
            {
                "module": "merge",
                "submodule": "branch",
                "boundary_index": local_idx,
                "merge_side": merge_side,
                "target_main_boundary_index": target_boundaries[local_idx],
                "visibility_rule": visibility_rule,
                "branch_marking_end_s": float(branch_marking_end_s),
            },
        )

        ids = []
        if roadline is not None:
            ids.append(roadline.id_)
            roadlines.append(roadline)

        branch_boundary_line_ids.append(ids)

    branch_lanes: list[Lane] = []

    for lane_idx in range(branch_n):
        left_line = LineString(branch_boundaries[lane_idx])
        right_line = LineString(branch_boundaries[lane_idx + 1])
        target_lane_idx = target_lanes[lane_idx]

        lane = Lane(
            id_=str(id_counter),
            left_side=left_line,
            right_side=right_line,
            subtype="road",
            speed_limit=min(speed, float(branch_in.speed_limit)),
            speed_limit_unit="km/h",
            line_ids={
                "left": branch_boundary_line_ids[lane_idx],
                "right": branch_boundary_line_ids[lane_idx + 1],
            },
            custom_tags={
                "module": "merge",
                "submodule": "branch",
                "lane_index": lane_idx,
                "target_main_lane_index": target_lane_idx,
                "target_main_lane_id": main_lanes[target_lane_idx].id_,
                "merge_side": merge_side,
            },
        )
        id_counter += 1

        lanes.append(lane)
        branch_lanes.append(lane)

    for i, lane in enumerate(branch_lanes):
        if i > 0:
            lane.add_related_lane(branch_lanes[i - 1].id_, LaneRelationship.LEFT_NEIGHBOR)
        if i < len(branch_lanes) - 1:
            lane.add_related_lane(branch_lanes[i + 1].id_, LaneRelationship.RIGHT_NEIGHBOR)

    for local_idx, branch_lane in enumerate(branch_lanes):
        target_lane_idx = target_lanes[local_idx]
        main_lane = main_lanes[target_lane_idx]
        branch_lane.add_related_lane(main_lane.id_, LaneRelationship.SUCCESSOR)
        main_lane.add_related_lane(branch_lane.id_, LaneRelationship.PREDECESSOR)

    main_lane_ids = tuple(lane.id_ for lane in main_lanes)
    branch_lane_ids = tuple(lane.id_ for lane in branch_lanes)

    main_in_base = RoadPort(
        point=np.asarray(main_in.point, dtype=float),
        heading=float(main_in.heading),
        lane_num=main_n,
        lane_width=lane_w,
        speed_limit=speed,
    )
    branch_in_base = RoadPort(
        point=np.asarray(branch_in.point, dtype=float),
        heading=float(branch_in.heading),
        lane_num=branch_n,
        lane_width=lane_w,
        speed_limit=float(branch_in.speed_limit),
    )
    main_out_base = RoadPort(
        point=np.asarray(main_out.point, dtype=float),
        heading=float(main_out.heading),
        lane_num=main_n,
        lane_width=lane_w,
        speed_limit=speed,
    )

    ports = {
        "main_in": make_port(
            main_in_base,
            kind="merge_main_in",
            name="main_in",
            lane_ids=main_lane_ids,
            metadata={"module": "merge", "merge_side": merge_side},
        ),
        "branch_in": make_port(
            branch_in_base,
            kind="merge_branch_in",
            name="branch_in",
            lane_ids=branch_lane_ids,
            metadata={"module": "merge", "merge_side": merge_side},
        ),
        "main_out": make_port(
            main_out_base,
            kind="merge_main_out",
            name="main_out",
            lane_ids=main_lane_ids,
            metadata={"module": "merge", "merge_side": merge_side},
        ),
    }

    main_stats = curvature_stats(main_center)
    main_self_intersection = has_self_intersection(main_center)

    branch_total_length = 0.0
    branch_max_curvature = 0.0
    branch_max_curvature_rate = 0.0
    branch_self_intersection = False

    for centerline in branch_centerlines:
        stats = curvature_stats(centerline)
        branch_total_length += polyline_length(centerline)
        branch_max_curvature = max(branch_max_curvature, stats["max_abs_curvature"])
        branch_max_curvature_rate = max(branch_max_curvature_rate, stats["max_abs_curvature_rate"])
        branch_self_intersection = branch_self_intersection or has_self_intersection(centerline)

    accepted_reasons: list[str] = []
    if main_self_intersection:
        accepted_reasons.append("main_self_intersection")
    if branch_self_intersection:
        accepted_reasons.append("branch_self_intersection")

    quality = {
        "module": "merge",
        "merge_side": merge_side,
        "main_lane_num": main_n,
        "branch_lane_num": branch_n,
        "main_length": main_length,
        "branch_length": branch_total_length,
        "merge_s": float(merge_s),
        "merge_point": merge_point.tolist(),
        "merge_heading": float(merge_heading),
        "branch_end": branch_end.tolist(),
        "target_lane_indices": target_lanes,
        "target_boundary_indices": target_boundaries,
        "main_outside_boundary_index": int(main_outside_boundary_idx),
        "branch_inside_boundary_index": int(branch_inside_boundary_idx),
        "branch_outside_boundary_index": int(branch_outside_boundary_idx),
        "main_outside_gap_start_s": float(main_outside_gap_start_s),
        "main_outside_gap_end_s": float(main_outside_gap_end_s),
        "branch_marking_end_s": float(branch_marking_end_s),
        "branch_angle_delta": float(_wrap_angle(merge_heading - float(branch_in.heading))),
        "main_self_intersection": main_self_intersection,
        "branch_self_intersection": branch_self_intersection,
        "main_max_abs_curvature": main_stats["max_abs_curvature"],
        "main_max_abs_curvature_rate": main_stats["max_abs_curvature_rate"],
        "branch_max_abs_curvature": branch_max_curvature,
        "branch_max_abs_curvature_rate": branch_max_curvature_rate,
        "accepted_reasons": accepted_reasons,
        "accepted": len(accepted_reasons) == 0,
    }

    return RoadModuleResult(
        lanes=lanes,
        roadlines=roadlines,
        ports=ports,
        interfaces=ports_to_interfaces(ports),
        quality=quality,
        id_counter=id_counter,
    )
