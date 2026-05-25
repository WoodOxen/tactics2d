# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Socket-driven merge road generator."""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Point

from tactics2d.geometry import (
    curvature_stats,
    cut_polyline,
    find_intersection_point,
    has_self_intersection,
    normalize_angle,
    offset_polyline,
    point_heading_at_s,
    polyline_length,
)
from tactics2d.map.element import Lane, RoadLine
from tactics2d.map.generator.helpers.element_builder import (
    add_ordered_lane_neighbors,
    build_lane_from_boundaries,
    build_optional_roadline_from_points,
    link_lanes,
)
from tactics2d.map.generator.helpers.reference_line import fit_reference_line

from ..rules.lane_marking_rules import roadline_render_kwargs
from ..rules.module_types import RoadModuleResult, RoadPort, make_port, ports_to_interfaces
from ._fork_merge_helpers import accumulate_branch_stats as _accumulate_branch_stats
from ._fork_merge_helpers import boundary_offset as _boundary_offset
from ._fork_merge_helpers import branch_boundary_token as _shared_branch_boundary_token
from ._fork_merge_helpers import (
    branch_centerlines_from_boundaries as _branch_centerlines_from_boundaries,
)
from ._fork_merge_helpers import branch_outer_point as _branch_outer_point
from ._fork_merge_helpers import build_main_road_section as _build_main_road_section
from ._fork_merge_helpers import choose_merge_s as _choose_merge_s
from ._fork_merge_helpers import side_boundary_indices as _side_boundary_indices
from ._fork_merge_helpers import side_lane_indices as _side_lane_indices


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
    """Generate a lane-level merge module.

    Args:
        main_in: Upstream main-road socket. Its lane metadata is used as defaults.
        branch_in: Upstream branch socket that merges into the main road.
        main_out: Downstream main-road socket.
        merge_side: Side of the main road where the branch enters. Must be
            ``"left"`` or ``"right"``.
        main_lane_num: Number of main-road lanes. Defaults to ``main_in.lane_num``.
        branch_lane_num: Number of branch lanes. Defaults to ``branch_in.lane_num``.
        lane_width: Lane width in metres. Defaults to ``main_in.lane_width``.
        speed_limit: Main-road speed limit. Defaults to ``main_in.speed_limit``.
        taper_length: Minimum longitudinal distance reserved for the merge opening.
        branch_length: Nominal branch length used when choosing the merge point.
        merge_s_ratio: Optional normalized merge position on the main reference
            line. When omitted, the merge position is inferred from ``branch_in``.
        step_size: Reference-line sampling interval.
        id_offset: First id used by generated map elements.

    Returns:
        A ``RoadModuleResult`` containing generated main/branch lanes, roadlines,
        ports, interfaces, and the next id counter. The ``quality`` dictionary
        includes ``module``, ``merge_side``, lane counts, ``main_length``,
        ``branch_length``, ``merge_s``, ``merge_point``, ``merge_heading``,
        ``branch_end``, target lane/boundary indices, hidden-opening
        arc-length fields, branch marking end position, branch angle delta,
        self-intersection flags, curvature statistics, ``accepted_reasons``,
        and ``accepted``.

    Raises:
        ValueError: If ``merge_side`` is not ``"left"`` or ``"right"``;
            ``step_size``, ``taper_length``, ``branch_length``, or ``lane_width``
            is non-positive; a lane count is smaller than one; or
            ``branch_lane_num`` exceeds ``main_lane_num``.
    """
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
    merge_point, merge_heading = point_heading_at_s(main_center, merge_s)

    main_boundaries: list[np.ndarray] = []
    for boundary_idx in range(main_n + 1):
        main_boundaries.append(
            offset_polyline(main_center, _boundary_offset(boundary_idx, main_n, lane_w))
        )

    target_lanes = _side_lane_indices(main_n, branch_n, merge_side, module_name="merge")
    target_boundaries = _side_boundary_indices(main_n, branch_n, merge_side, module_name="merge")

    branch_end = _branch_outer_point(
        main_boundaries=main_boundaries,
        side_boundaries=target_boundaries,
        s_on_main=merge_s,
        main_length=main_length,
        side=merge_side,
        branch_n=branch_n,
        lane_w=lane_w,
        heading=merge_heading,
    )

    branch_chord = branch_end - np.asarray(branch_in.point, dtype=float)
    branch_chord_len = float(np.linalg.norm(branch_chord))
    if branch_chord_len > 1e-6:
        branch_depart_heading = float(np.arctan2(branch_chord[1], branch_chord[0]))
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
        intersect = find_intersection_point(b_line, main_out_line, pick="last_on_line1")

        if intersect is None:
            branch_cut_lines.append(b_pts)
        else:
            branch_cut_lines.append(cut_polyline(b_line, intersect, "before"))
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

        main_before_pts = cut_polyline(main_out_line, pt_first, "before")
        main_rest = cut_polyline(main_out_line, pt_first, "after")
        main_after_pts = cut_polyline(LineString(main_rest), pt_second, "after")

    branch_marking_end_s = 0.0
    if nose_pt is not None:
        branch_marking_end_s = float(
            LineString(branch_boundaries[branch_inside_boundary_idx]).project(nose_pt)
        )

    branch_centerlines = _branch_centerlines_from_boundaries(branch_boundaries)

    lanes: list[Lane] = []
    roadlines: list[RoadLine] = []
    id_counter = id_offset

    main_lanes, main_roadlines, _, id_counter = _build_main_road_section(
        main_n=main_n,
        main_boundaries=main_boundaries,
        outside_boundary_idx=main_outside_boundary_idx,
        before_pts=main_before_pts,
        after_pts=main_after_pts,
        speed=speed,
        module="merge",
        side_key="merge_side",
        side_value=merge_side,
        id_counter=id_counter,
    )
    lanes.extend(main_lanes)
    roadlines.extend(main_roadlines)

    branch_boundary_line_ids: list[list[str]] = []

    for local_idx, boundary_pts in enumerate(branch_boundaries):
        token = _shared_branch_boundary_token(local_idx, branch_n + 1)

        if local_idx == branch_outside_boundary_idx:
            visibility_rule = "outside_edge_until_merge"
        elif local_idx == branch_inside_boundary_idx:
            visibility_rule = "inside_edge_until_merge"
        else:
            visibility_rule = "interior_divider_until_midpoint"

        visible_boundary = branch_cut_lines[local_idx]

        roadline, id_counter = build_optional_roadline_from_points(
            id_counter,
            visible_boundary,
            marking_kwargs=roadline_render_kwargs(
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
            ),
        )

        ids = []
        if roadline is not None:
            ids.append(roadline.id_)
            roadlines.append(roadline)

        branch_boundary_line_ids.append(ids)

    branch_lanes: list[Lane] = []

    for lane_idx in range(branch_n):
        target_lane_idx = target_lanes[lane_idx]

        lane = build_lane_from_boundaries(
            id_=id_counter,
            left_points=branch_boundaries[lane_idx],
            right_points=branch_boundaries[lane_idx + 1],
            left_roadline_ids=branch_boundary_line_ids[lane_idx],
            right_roadline_ids=branch_boundary_line_ids[lane_idx + 1],
            speed_limit=min(speed, float(branch_in.speed_limit)),
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

    add_ordered_lane_neighbors(branch_lanes)

    for local_idx, branch_lane in enumerate(branch_lanes):
        target_lane_idx = target_lanes[local_idx]
        main_lane = main_lanes[target_lane_idx]
        link_lanes(branch_lane, main_lane)

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

    (
        branch_total_length,
        branch_max_curvature,
        branch_max_curvature_rate,
        branch_self_intersection,
    ) = _accumulate_branch_stats(branch_centerlines)

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
        "branch_angle_delta": float(normalize_angle(merge_heading - float(branch_in.heading))),
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
