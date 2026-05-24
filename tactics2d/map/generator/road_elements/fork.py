# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Socket-driven fork road generator."""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Point

from tactics2d.geometry import (
    curvature_stats,
    cut_polyline,
    find_intersection_point,
    has_self_intersection,
    nearest_s,
    normalize_angle,
    offset_polyline,
    point_at_s,
    point_heading_at_s,
    polyline_length,
    resample_polyline,
)
from tactics2d.map.element import Lane, RoadLine
from tactics2d.map.generator.helpers.element_builder import (
    add_ordered_lane_neighbors,
    build_lane_from_boundaries,
    build_optional_roadline_from_points,
    link_lanes,
)
from tactics2d.map.generator.helpers.reference_line import fit_reference_line

from ..rules.lane_marking_rules import one_way_mark, ramp_mark, roadline_render_kwargs
from ..rules.module_types import RoadModuleResult, RoadPort, make_port, ports_to_interfaces


def _boundary_offset(boundary_index: int, lane_num: int, lane_width: float) -> float:
    """Return boundary offset from road reference line."""
    half_width = lane_num * lane_width / 2.0
    return half_width - boundary_index * lane_width


def _boundary_token(boundary_index: int, boundary_num: int) -> str:
    """Return active-standard token for a main one-way road boundary.

    MUTCD:
      - one-way left edge: solid yellow edge
      - one-way right edge: solid white edge
      - same-direction interior lane divider: dashed white

    GB mapping is handled by lane_marking_rules.one_way_mark().
    """
    lane_num = boundary_num - 1

    if boundary_index == 0:
        return one_way_mark(0, lane_num, "left")

    if boundary_index == boundary_num - 1:
        return one_way_mark(lane_num - 1, lane_num, "right")

    return one_way_mark(boundary_index - 1, lane_num, "right")


def _branch_boundary_token(boundary_index: int, boundary_num: int, fork_side: str) -> str:
    """Return active-standard token for a fork branch/ramp boundary.

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


def _choose_diverge_s(
    main_center: np.ndarray, branch_point: np.ndarray, taper_length: float, branch_length: float
) -> float:
    """Choose the branch diverge section on the main reference line."""
    total = polyline_length(main_center)
    projected_s = nearest_s(main_center, branch_point)

    backoff = max(float(taper_length), float(branch_length) * 0.45)

    lower = total * 0.15
    upper = total * 0.82

    if upper <= lower:
        return total * 0.5

    return float(np.clip(projected_s - backoff, lower, upper))


def _source_lane_indices(main_lane_num: int, branch_lane_num: int, fork_side: str) -> list[int]:
    """Return main-lane indices used by the branch."""
    if branch_lane_num > main_lane_num:
        raise ValueError("branch_lane_num cannot exceed main_lane_num in fork v1.")

    if fork_side == "right":
        start = main_lane_num - branch_lane_num
        return list(range(start, main_lane_num))

    return list(range(branch_lane_num))


def _source_boundary_indices(main_lane_num: int, branch_lane_num: int, fork_side: str) -> list[int]:
    """Return main-boundary indices used by the branch section."""
    if branch_lane_num > main_lane_num:
        raise ValueError("branch_lane_num cannot exceed main_lane_num in fork v1.")

    if fork_side == "right":
        start = main_lane_num - branch_lane_num
        return list(range(start, main_lane_num + 1))

    return list(range(0, branch_lane_num + 1))


def _branch_start_from_outer_boundary(
    main_boundaries: list[np.ndarray],
    source_boundaries: list[int],
    diverge_s: float,
    main_length: float,
    fork_side: str,
    branch_n: int,
    lane_w: float,
    diverge_heading: float,
) -> np.ndarray:
    """Return the branch reference-line start point."""
    outer_idx = source_boundaries[-1] if fork_side == "right" else source_boundaries[0]
    boundary = main_boundaries[outer_idx]
    boundary_total = polyline_length(boundary)

    if main_length < 1e-9:
        boundary_s = 0.0
    else:
        boundary_s = diverge_s / main_length * boundary_total

    outer_pt = point_at_s(boundary, boundary_s)

    if fork_side == "right":
        inward_normal = np.array([-np.sin(diverge_heading), np.cos(diverge_heading)], dtype=float)
    else:
        inward_normal = np.array([np.sin(diverge_heading), -np.cos(diverge_heading)], dtype=float)

    return outer_pt + inward_normal * (branch_n * lane_w / 2.0)


def _branch_centerlines_from_boundaries(branch_boundaries: list[np.ndarray]) -> list[np.ndarray]:
    """Build approximate branch centerlines from adjacent boundary pairs."""
    centerlines: list[np.ndarray] = []

    for i in range(len(branch_boundaries) - 1):
        left = branch_boundaries[i]
        right = branch_boundaries[i + 1]
        n = max(2, min(len(left), len(right)))
        left_r = resample_polyline(left, n)
        right_r = resample_polyline(right, n)
        centerlines.append((left_r + right_r) * 0.5)

    return centerlines


def fork(
    main_in: RoadPort,
    main_out: RoadPort,
    branch_out: RoadPort,
    *,
    fork_side: str = "right",
    main_lane_num: int | None = None,
    branch_lane_num: int | None = None,
    lane_width: float | None = None,
    speed_limit: float | None = None,
    taper_length: float = 65.0,
    branch_length: float = 85.0,
    diverge_s_ratio: float | None = None,
    step_size: float = 0.1,
    id_offset: int = 0,
) -> RoadModuleResult:
    """Generate a lane-level fork module."""
    if fork_side not in ("left", "right"):
        raise ValueError("fork_side must be 'left' or 'right'.")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive.")
    if taper_length <= 0.0:
        raise ValueError("taper_length must be positive.")
    if branch_length <= 0.0:
        raise ValueError("branch_length must be positive.")

    main_n = int(main_lane_num if main_lane_num is not None else main_in.lane_num)
    branch_n = int(branch_lane_num if branch_lane_num is not None else branch_out.lane_num)
    lane_w = float(lane_width if lane_width is not None else main_in.lane_width)
    speed = float(speed_limit if speed_limit is not None else main_in.speed_limit)

    if main_n < 1:
        raise ValueError("main_lane_num must be >= 1.")
    if branch_n < 1:
        raise ValueError("branch_lane_num must be >= 1.")
    if branch_n > main_n:
        raise ValueError("fork v1 requires branch_lane_num <= main_lane_num.")
    if lane_w <= 0.0:
        raise ValueError("lane_width must be positive.")

    main_center = fit_reference_line(
        main_in.point, main_in.heading, main_out.point, main_out.heading, step_size
    )

    main_length = polyline_length(main_center)
    if diverge_s_ratio is not None:
        diverge_s = float(np.clip(diverge_s_ratio, 0.0, 1.0)) * main_length
    else:
        diverge_s = _choose_diverge_s(
            main_center, np.asarray(branch_out.point, dtype=float), taper_length, branch_length
        )
    diverge_point, diverge_heading = point_heading_at_s(main_center, diverge_s)

    main_boundaries: list[np.ndarray] = []
    for boundary_idx in range(main_n + 1):
        main_boundaries.append(
            offset_polyline(main_center, _boundary_offset(boundary_idx, main_n, lane_w))
        )

    source_lanes = _source_lane_indices(main_n, branch_n, fork_side)
    source_boundaries = _source_boundary_indices(main_n, branch_n, fork_side)

    branch_start = _branch_start_from_outer_boundary(
        main_boundaries=main_boundaries,
        source_boundaries=source_boundaries,
        diverge_s=diverge_s,
        main_length=main_length,
        fork_side=fork_side,
        branch_n=branch_n,
        lane_w=lane_w,
        diverge_heading=diverge_heading,
    )

    # Start the branch tangent to the main road.
    # This avoids a hard chord-angle insertion at the fork opening.
    branch_depart_heading = float(diverge_heading)

    branch_center = fit_reference_line(
        branch_start,
        branch_depart_heading,
        np.asarray(branch_out.point, dtype=float),
        float(branch_out.heading),
        step_size,
    )

    branch_boundaries: list[np.ndarray] = []
    for local_boundary_idx in range(branch_n + 1):
        branch_boundaries.append(
            offset_polyline(branch_center, _boundary_offset(local_boundary_idx, branch_n, lane_w))
        )

    if fork_side == "right":
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
    diverge_pt: Point | None = None

    for local_idx, b_pts in enumerate(branch_boundaries):
        b_line = LineString(b_pts)
        intersect = find_intersection_point(b_line, main_out_line, pick="first_on_line1")

        if intersect is None:
            branch_cut_lines.append(b_pts)
        else:
            branch_cut_lines.append(cut_polyline(b_line, intersect, "after"))
            if local_idx == branch_inside_boundary_idx:
                nose_pt = intersect
            if local_idx == branch_outside_boundary_idx:
                diverge_pt = intersect

    main_before_pts = main_outside_boundary
    main_after_pts = np.empty((0, 2), dtype=float)
    main_outside_gap_start_s = 0.0
    main_outside_gap_end_s = 0.0

    if nose_pt is not None and diverge_pt is not None:
        s_nose = float(main_out_line.project(nose_pt))
        s_diverge = float(main_out_line.project(diverge_pt))

        if s_diverge < s_nose:
            pt_first = diverge_pt
            pt_second = nose_pt
            main_outside_gap_start_s = s_diverge
            main_outside_gap_end_s = s_nose
        else:
            pt_first = nose_pt
            pt_second = diverge_pt
            main_outside_gap_start_s = s_nose
            main_outside_gap_end_s = s_diverge

        main_before_pts = cut_polyline(main_out_line, pt_first, "before")
        main_rest = cut_polyline(main_out_line, pt_first, "after")
        main_after_pts = cut_polyline(LineString(main_rest), pt_second, "after")

    branch_marking_start_s = 0.0
    if nose_pt is not None:
        branch_marking_start_s = float(
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
            before_roadline, id_counter = build_optional_roadline_from_points(
                id_counter,
                main_before_pts,
                marking_kwargs=roadline_render_kwargs(
                    token,
                    {
                        "module": "fork",
                        "submodule": "main",
                        "boundary_index": boundary_idx,
                        "fork_side": fork_side,
                        "segment": "before_branch_opening",
                        "opening_hidden": True,
                    },
                ),
            )
            if before_roadline is not None:
                ids.append(before_roadline.id_)
                roadlines.append(before_roadline)

            after_roadline, id_counter = build_optional_roadline_from_points(
                id_counter,
                main_after_pts,
                marking_kwargs=roadline_render_kwargs(
                    token,
                    {
                        "module": "fork",
                        "submodule": "main",
                        "boundary_index": boundary_idx,
                        "fork_side": fork_side,
                        "segment": "after_branch_opening",
                        "opening_hidden": True,
                    },
                ),
            )
            if after_roadline is not None:
                ids.append(after_roadline.id_)
                roadlines.append(after_roadline)

        else:
            roadline, id_counter = build_optional_roadline_from_points(
                id_counter,
                boundary_pts,
                marking_kwargs=roadline_render_kwargs(
                    token,
                    {
                        "module": "fork",
                        "submodule": "main",
                        "boundary_index": boundary_idx,
                        "fork_side": fork_side,
                        "kept_on_main": True,
                    },
                ),
            )
            if roadline is not None:
                ids.append(roadline.id_)
                roadlines.append(roadline)

        main_boundary_line_ids.append(ids)

    main_lanes: list[Lane] = []

    for lane_idx in range(main_n):
        lane = build_lane_from_boundaries(
            id_=id_counter,
            left_points=main_boundaries[lane_idx],
            right_points=main_boundaries[lane_idx + 1],
            left_roadline_ids=main_boundary_line_ids[lane_idx],
            right_roadline_ids=main_boundary_line_ids[lane_idx + 1],
            speed_limit=speed,
            custom_tags={
                "module": "fork",
                "submodule": "main",
                "lane_index": lane_idx,
                "fork_side": fork_side,
            },
        )
        id_counter += 1

        lanes.append(lane)
        main_lanes.append(lane)

    add_ordered_lane_neighbors(main_lanes)

    branch_boundary_line_ids: list[list[str]] = []

    for local_idx, boundary_pts in enumerate(branch_boundaries):
        token = _branch_boundary_token(local_idx, branch_n + 1, fork_side)

        if local_idx == branch_outside_boundary_idx:
            visibility_rule = "outside_edge_from_origin"
        elif local_idx == branch_inside_boundary_idx:
            visibility_rule = "inside_edge_from_origin"
        else:
            visibility_rule = "interior_divider_from_midpoint"

        visible_boundary = branch_cut_lines[local_idx]

        roadline, id_counter = build_optional_roadline_from_points(
            id_counter,
            visible_boundary,
            marking_kwargs=roadline_render_kwargs(
                token,
                {
                    "module": "fork",
                    "submodule": "branch",
                    "boundary_index": local_idx,
                    "fork_side": fork_side,
                    "source_main_boundary_index": source_boundaries[local_idx],
                    "visibility_rule": visibility_rule,
                    "branch_marking_start_s": float(branch_marking_start_s),
                },
            ),
        )

        ids = []
        if roadline is not None:
            ids.append(roadline.id_)
            roadlines.append(roadline)

        branch_boundary_line_ids.append(ids)

    branch_lanes: list[Lane] = []

    for lane_idx in range(branch_n):
        source_lane_idx = source_lanes[lane_idx]

        lane = build_lane_from_boundaries(
            id_=id_counter,
            left_points=branch_boundaries[lane_idx],
            right_points=branch_boundaries[lane_idx + 1],
            left_roadline_ids=branch_boundary_line_ids[lane_idx],
            right_roadline_ids=branch_boundary_line_ids[lane_idx + 1],
            speed_limit=min(speed, float(branch_out.speed_limit)),
            custom_tags={
                "module": "fork",
                "submodule": "branch",
                "lane_index": lane_idx,
                "source_main_lane_index": source_lane_idx,
                "source_main_lane_id": main_lanes[source_lane_idx].id_,
                "fork_side": fork_side,
            },
        )
        id_counter += 1

        lanes.append(lane)
        branch_lanes.append(lane)

    add_ordered_lane_neighbors(branch_lanes)

    for local_idx, branch_lane in enumerate(branch_lanes):
        source_lane_idx = source_lanes[local_idx]
        main_lane = main_lanes[source_lane_idx]
        link_lanes(main_lane, branch_lane)

    main_lane_ids = tuple(lane.id_ for lane in main_lanes)
    branch_lane_ids = tuple(lane.id_ for lane in branch_lanes)

    main_in_base = RoadPort(
        point=np.asarray(main_in.point, dtype=float),
        heading=float(main_in.heading),
        lane_num=main_n,
        lane_width=lane_w,
        speed_limit=speed,
    )
    main_out_base = RoadPort(
        point=np.asarray(main_out.point, dtype=float),
        heading=float(main_out.heading),
        lane_num=main_n,
        lane_width=lane_w,
        speed_limit=speed,
    )
    branch_out_base = RoadPort(
        point=np.asarray(branch_out.point, dtype=float),
        heading=float(branch_out.heading),
        lane_num=branch_n,
        lane_width=lane_w,
        speed_limit=float(branch_out.speed_limit),
    )

    ports = {
        "main_in": make_port(
            main_in_base,
            kind="fork_main_in",
            name="main_in",
            lane_ids=main_lane_ids,
            metadata={"module": "fork", "fork_side": fork_side},
        ),
        "main_out": make_port(
            main_out_base,
            kind="fork_main_out",
            name="main_out",
            lane_ids=main_lane_ids,
            metadata={"module": "fork", "fork_side": fork_side},
        ),
        "branch_out": make_port(
            branch_out_base,
            kind="fork_branch_out",
            name="branch_out",
            lane_ids=branch_lane_ids,
            metadata={"module": "fork", "fork_side": fork_side},
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
        "module": "fork",
        "fork_side": fork_side,
        "main_lane_num": main_n,
        "branch_lane_num": branch_n,
        "main_length": main_length,
        "branch_length": branch_total_length,
        "diverge_s": float(diverge_s),
        "diverge_point": diverge_point.tolist(),
        "diverge_heading": float(diverge_heading),
        "branch_start": branch_start.tolist(),
        "branch_depart_heading": float(branch_depart_heading),
        "source_lane_indices": source_lanes,
        "source_boundary_indices": source_boundaries,
        "main_outside_boundary_index": int(main_outside_boundary_idx),
        "branch_inside_boundary_index": int(branch_inside_boundary_idx),
        "branch_outside_boundary_index": int(branch_outside_boundary_idx),
        "main_outside_gap_start_s": float(main_outside_gap_start_s),
        "main_outside_gap_end_s": float(main_outside_gap_end_s),
        "branch_marking_start_s": float(branch_marking_start_s),
        "branch_angle_delta": float(normalize_angle(float(branch_out.heading) - diverge_heading)),
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
