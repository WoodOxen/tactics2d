# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared helpers for fork and merge road generators."""

from __future__ import annotations

import numpy as np

from tactics2d.geometry import (
    curvature_stats,
    has_self_intersection,
    nearest_s,
    point_at_s,
    polyline_length,
    resample_polyline,
)
from tactics2d.map.element import Lane, RoadLine
from tactics2d.map.generator.helpers.element_builder import (
    add_ordered_lane_neighbors,
    build_lane_from_boundaries,
    build_optional_roadline_from_points,
)

from ..rules.lane_marking_rules import one_way_mark, ramp_mark, roadline_render_kwargs


def boundary_offset(boundary_index: int, lane_num: int, lane_width: float) -> float:
    """Return boundary offset from road reference line."""
    half_width = lane_num * lane_width / 2.0
    return half_width - boundary_index * lane_width


def boundary_token(boundary_index: int, boundary_num: int) -> str:
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


def branch_boundary_token(boundary_index: int, boundary_num: int) -> str:
    """Return active-standard token for a branch/ramp boundary.

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


def choose_diverge_s(
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


def choose_merge_s(
    main_center: np.ndarray, branch_point: np.ndarray, taper_length: float, branch_length: float
) -> float:
    """Choose the branch merge section on the main reference line."""
    total = polyline_length(main_center)
    projected_s = nearest_s(main_center, branch_point)

    advance = max(float(taper_length), float(branch_length) * 0.45)

    lower = total * 0.18
    upper = total * 0.85

    if upper <= lower:
        return total * 0.5

    return float(np.clip(projected_s + advance, lower, upper))


def side_lane_indices(
    main_lane_num: int, branch_lane_num: int, side: str, *, module_name: str
) -> list[int]:
    """Return main-lane indices touched by a side branch."""
    if branch_lane_num > main_lane_num:
        raise ValueError(f"branch_lane_num cannot exceed main_lane_num in {module_name} v1.")

    if side == "right":
        start = main_lane_num - branch_lane_num
        return list(range(start, main_lane_num))

    return list(range(branch_lane_num))


def side_boundary_indices(
    main_lane_num: int, branch_lane_num: int, side: str, *, module_name: str
) -> list[int]:
    """Return main-boundary indices touched by a side branch."""
    if branch_lane_num > main_lane_num:
        raise ValueError(f"branch_lane_num cannot exceed main_lane_num in {module_name} v1.")

    if side == "right":
        start = main_lane_num - branch_lane_num
        return list(range(start, main_lane_num + 1))

    return list(range(0, branch_lane_num + 1))


def branch_outer_point(
    main_boundaries: list[np.ndarray],
    side_boundaries: list[int],
    s_on_main: float,
    main_length: float,
    side: str,
    branch_n: int,
    lane_w: float,
    heading: float,
) -> np.ndarray:
    """Return the branch reference-line anchor on the outer main boundary.

    Used by both fork (as the branch start point) and merge (as the branch
    end point).  The returned point sits at the branch centreline position on
    the outer boundary of the affected lanes, offset inward by half the branch
    width.

    Args:
        main_boundaries: Precomputed main-road boundary polylines indexed left
            to right.
        side_boundaries: Boundary indices on the branch side (from
            ``side_boundary_indices()``).
        s_on_main: Arc-length coordinate of the diverge/merge point on the
            main reference line.
        main_length: Total arc length of the main reference line.
        side: ``"right"`` or ``"left"`` side of the main road.
        branch_n: Number of branch lanes.
        lane_w: Lane width in metres.
        heading: Tangent heading at the diverge/merge point.

    Returns:
        2-D world coordinate of the branch anchor with shape ``(2,)``.
    """
    outer_idx = side_boundaries[-1] if side == "right" else side_boundaries[0]
    boundary = main_boundaries[outer_idx]
    boundary_total = polyline_length(boundary)
    boundary_s = 0.0 if main_length < 1e-9 else s_on_main / main_length * boundary_total
    outer_pt = point_at_s(boundary, boundary_s)

    if side == "right":
        inward_normal = np.array([-np.sin(heading), np.cos(heading)], dtype=float)
    else:
        inward_normal = np.array([np.sin(heading), -np.cos(heading)], dtype=float)

    return outer_pt + inward_normal * (branch_n * lane_w / 2.0)


def build_main_road_section(
    *,
    main_n: int,
    main_boundaries: list[np.ndarray],
    outside_boundary_idx: int,
    before_pts: np.ndarray,
    after_pts: np.ndarray,
    speed: float,
    module: str,
    side_key: str,
    side_value: str,
    id_counter: int,
) -> tuple[list[Lane], list[RoadLine], list[list[str]], int]:
    """Build the main-road boundary roadlines and lane objects for fork or merge.

    Both fork and merge lay out their main road with an identical pattern: a set
    of parallel boundary polylines where the outer (branch-side) edge is split
    into ``before`` and ``after`` segments around the branch opening, and all
    other boundaries are kept continuous.

    Args:
        main_n: Number of main-road lanes.
        main_boundaries: Precomputed boundary polylines indexed left to right
            (index 0 is leftmost, index ``main_n`` is rightmost).
        outside_boundary_idx: Index of the outer boundary on the branch side
            (the boundary that is split around the branch opening).
        before_pts: Outer-boundary points before the branch opening.
        after_pts: Outer-boundary points after the branch opening. May be an
            empty ``(0, 2)`` array when no visible post-opening segment
            remains.
        speed: Main-road speed limit in km/h.
        module: Module tag written into every element's ``custom_tags``
            (``"fork"`` or ``"merge"``).
        side_key: Key name for the side tag in ``custom_tags``
            (``"fork_side"`` or ``"merge_side"``).
        side_value: Side value (``"left"`` or ``"right"``).
        id_counter: Starting element id counter.

    Returns:
        A tuple ``(main_lanes, roadlines, boundary_line_ids, id_counter)``
        where ``boundary_line_ids[i]`` lists the roadline id(s) for boundary
        index ``i``, and ``id_counter`` is updated after all allocations.
    """
    main_boundary_line_ids: list[list[str]] = []
    roadlines: list[RoadLine] = []

    for boundary_idx, boundary_pts in enumerate(main_boundaries):
        token = boundary_token(boundary_idx, main_n + 1)
        ids: list[str] = []

        if boundary_idx == outside_boundary_idx:
            before_rl, id_counter = build_optional_roadline_from_points(
                id_counter,
                before_pts,
                marking_kwargs=roadline_render_kwargs(
                    token,
                    {
                        "module": module,
                        "submodule": "main",
                        "boundary_index": boundary_idx,
                        side_key: side_value,
                        "segment": "before_branch_opening",
                        "opening_hidden": True,
                    },
                ),
            )
            if before_rl is not None:
                ids.append(before_rl.id_)
                roadlines.append(before_rl)

            after_rl, id_counter = build_optional_roadline_from_points(
                id_counter,
                after_pts,
                marking_kwargs=roadline_render_kwargs(
                    token,
                    {
                        "module": module,
                        "submodule": "main",
                        "boundary_index": boundary_idx,
                        side_key: side_value,
                        "segment": "after_branch_opening",
                        "opening_hidden": True,
                    },
                ),
            )
            if after_rl is not None:
                ids.append(after_rl.id_)
                roadlines.append(after_rl)
        else:
            rl, id_counter = build_optional_roadline_from_points(
                id_counter,
                boundary_pts,
                marking_kwargs=roadline_render_kwargs(
                    token,
                    {
                        "module": module,
                        "submodule": "main",
                        "boundary_index": boundary_idx,
                        side_key: side_value,
                        "kept_on_main": True,
                    },
                ),
            )
            if rl is not None:
                ids.append(rl.id_)
                roadlines.append(rl)

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
                "module": module,
                "submodule": "main",
                "lane_index": lane_idx,
                side_key: side_value,
            },
        )
        id_counter += 1
        main_lanes.append(lane)

    add_ordered_lane_neighbors(main_lanes)
    return main_lanes, roadlines, main_boundary_line_ids, id_counter


def accumulate_branch_stats(
    branch_centerlines: list[np.ndarray],
) -> tuple[float, float, float, bool]:
    """Accumulate branch-road quality statistics across all centerlines.

    Args:
        branch_centerlines: List of branch lane centerline polylines.

    Returns:
        A tuple ``(total_length, max_curvature, max_curvature_rate,
        self_intersection)`` aggregated over all input centerlines.
    """
    total_length = 0.0
    max_curvature = 0.0
    max_curvature_rate = 0.0
    self_intersection = False

    for centerline in branch_centerlines:
        stats = curvature_stats(centerline)
        total_length += polyline_length(centerline)
        max_curvature = max(max_curvature, stats["max_abs_curvature"])
        max_curvature_rate = max(max_curvature_rate, stats["max_abs_curvature_rate"])
        self_intersection = self_intersection or has_self_intersection(centerline)

    return total_length, max_curvature, max_curvature_rate, self_intersection


def branch_centerlines_from_boundaries(branch_boundaries: list[np.ndarray]) -> list[np.ndarray]:
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
