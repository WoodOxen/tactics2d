# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fork and merge road segment generator implementations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple

import numpy as np
from shapely.geometry import LineString, Point

from tactics2d.geometry import polyline
from tactics2d.map.element import Lane, RoadLine
from tactics2d.map.generator.rules.lane_marking_rules import (
    one_way_boundary_token,
    ramp_mark,
    roadline_render_kwargs,
)
from tactics2d.map.generator.rules.module_types import RoadModuleResult, RoadPort, build_port

from .element_builder import (
    add_ordered_lane_neighbors,
    boundary_offset,
    build_lane_from_boundaries,
    build_optional_roadline_from_points,
    link_lanes,
)
from .reference_line import fit_reference_line
from .road_segment import RoadSegment

_DIVERGE_LOWER: float = 0.15
_DIVERGE_UPPER: float = 0.82
_MERGE_LOWER: float = 0.18
_MERGE_UPPER: float = 0.85
_BRANCH_OFFSET_FACTOR: float = 0.45


class _BranchKind(NamedTuple):
    """Static configuration that distinguishes fork from merge geometry.

    Attributes:
        name: Module name (``"fork"`` or ``"merge"``), used in tags and errors.
        cut_dir: Polyline cut direction after the intersection point.
        pick: Which intersection to use when a branch boundary crosses the main
            outer boundary (``"first_on_line1"`` for fork, ``"last_on_line1"``
            for merge).
        meta_prefix: Prefix for cross-reference custom-tag keys
            (``"source_"`` or ``"target_"``).
        side_key: Custom-tag key for the branch side (``"fork_side"`` or
            ``"merge_side"``).
        marking_s_key: Custom-tag key for the branch-opening arc-length
            annotation.
        edge_suffix: Suffix appended to ``"outside_edge_"`` and
            ``"inside_edge_"`` visibility-rule tags.
        interior_suffix: Suffix appended to ``"interior_divider_"`` tags.
        port_kind_main_in: ``RoadPort.kind`` for the ``"main_in"`` port.
        port_kind_main_out: ``RoadPort.kind`` for the ``"main_out"`` port.
        port_kind_branch: ``RoadPort.kind`` for the branch port.
        port_name_branch: Key used to store the branch port in the result dict.
        diverges: ``True`` for fork (traffic splits off); ``False`` for merge
            (traffic joins).
        choose_s: Function that picks the branch-opening arc-length position.
    """

    name: str
    cut_dir: str
    pick: str
    meta_prefix: str
    side_key: str
    marking_s_key: str
    edge_suffix: str
    interior_suffix: str
    port_kind_main_in: str
    port_kind_main_out: str
    port_kind_branch: str
    port_name_branch: str
    diverges: bool
    choose_s: Callable


def _branch_boundary_token(boundary_index: int, boundary_num: int) -> str:
    """Return the active-standard marking token for a branch or ramp boundary.

    Args:
        boundary_index: 0 = leftmost boundary, ``boundary_num - 1`` = rightmost.
        boundary_num: Total number of boundaries (``branch_lane_num + 1``).

    Returns:
        Marking token string.
    """
    if boundary_index == 0:
        return ramp_mark("left_edge")
    if boundary_index == boundary_num - 1:
        return ramp_mark("right_edge")
    return ramp_mark("interior")


def _choose_branch_s(
    diverges: bool,
    main_center: np.ndarray,
    branch_point: np.ndarray,
    taper_length: float,
    branch_length: float,
) -> float:
    """Choose the branch diverge or merge arc-length position on the main reference line.

    For diverge (fork), backs off from the nearest point; for converge (merge),
    advances past it.  The result is clamped to the corresponding safe interval.

    Args:
        diverges: ``True`` for fork (back off), ``False`` for merge (advance).
        main_center: Sampled main-road centreline points.
        branch_point: Nominal branch-socket position in world coordinates.
        taper_length: Minimum longitudinal taper length reserved for the opening.
        branch_length: Nominal branch length used to estimate the offset distance.

    Returns:
        Arc-length position on the main centreline in metres.

    Note:
        The valid s-range is controlled by four module-level fraction constants
        (expressed as a proportion of the total main-road length):

        * ``_DIVERGE_LOWER`` = 0.15, ``_DIVERGE_UPPER`` = 0.82 — for fork, the
          diverge point is constrained to the middle 67 % of the main road.
          The 15 % head room prevents the opening from starting at the very
          beginning of the segment; the 18 % tail room keeps the branch nose
          visible before the exit socket.
        * ``_MERGE_LOWER`` = 0.18, ``_MERGE_UPPER`` = 0.85 — the symmetric bounds
          for merge; slightly tighter at the head end so the branch has room to
          approach from upstream.
        * ``_BRANCH_OFFSET_FACTOR`` = 0.45 — scales ``branch_length`` into a
          longitudinal upstream/downstream offset so the inferred s-position
          lands upstream (fork) or downstream (merge) of the branch socket's
          nearest point on the main centreline, rather than right at it.
    """
    total = polyline.arc_length(main_center)
    offset = max(float(taper_length), float(branch_length) * _BRANCH_OFFSET_FACTOR)
    if diverges:
        lower, upper = total * _DIVERGE_LOWER, total * _DIVERGE_UPPER
        s = polyline.project_arc_length(main_center, branch_point) - offset
    else:
        lower, upper = total * _MERGE_LOWER, total * _MERGE_UPPER
        s = polyline.project_arc_length(main_center, branch_point) + offset
    if upper <= lower:
        return total * 0.5
    return float(np.clip(s, lower, upper))


_FORK = _BranchKind(
    name="fork",
    cut_dir="after",
    pick="first_on_line1",
    meta_prefix="source_",
    side_key="fork_side",
    marking_s_key="branch_marking_start_s",
    edge_suffix="from_origin",
    interior_suffix="from_midpoint",
    port_kind_main_in="fork_main_in",
    port_kind_main_out="fork_main_out",
    port_kind_branch="fork_branch_out",
    port_name_branch="branch_out",
    diverges=True,
    choose_s=partial(_choose_branch_s, True),
)

_MERGE = _BranchKind(
    name="merge",
    cut_dir="before",
    pick="last_on_line1",
    meta_prefix="target_",
    side_key="merge_side",
    marking_s_key="branch_marking_end_s",
    edge_suffix="until_merge",
    interior_suffix="until_midpoint",
    port_kind_main_in="merge_main_in",
    port_kind_main_out="merge_main_out",
    port_kind_branch="merge_branch_in",
    port_name_branch="branch_in",
    diverges=False,
    choose_s=partial(_choose_branch_s, False),
)


@dataclass
class _BranchConfig:
    """Resolved runtime parameters for a fork or merge branch segment.

    Bundles all non-port, non-id arguments so :func:`_build_branch_segment`
    has a single config argument instead of a flat list of scalars.

    Attributes:
        bk: Structural configuration distinguishing fork from merge.
        side: Side of the main road where the branch connects
            (``"left"`` or ``"right"``).
        main_n: Resolved main-road lane count.
        branch_n: Resolved branch lane count.
        lane_w: Resolved lane width in metres.
        speed: Resolved main-road speed limit in km/h.
        taper_length: Minimum longitudinal distance reserved for the opening.
        branch_length: Nominal branch length for s-position inference in metres.
        step_size: Reference-line sampling interval in metres.
        s_ratio: Optional normalised branch position in ``[0, 1]``.
    """

    bk: _BranchKind
    side: str
    main_n: int
    branch_n: int
    lane_w: float
    speed: float
    taper_length: float
    branch_length: float
    step_size: float
    s_ratio: float | None = None


def _side_indices(
    main_lane_num: int, branch_lane_num: int, side: str
) -> tuple[list[int], list[int]]:
    """Return lane and boundary indices for the side branch on the main road.

    Args:
        main_lane_num: Number of lanes on the main road.
        branch_lane_num: Number of lanes on the branch.
        side: ``"left"`` or ``"right"``.

    Returns:
        A tuple ``(lane_indices, boundary_indices)`` where boundary indices
        include one extra entry on the far side of the branch lanes.

    Raises:
        ValueError: If ``branch_lane_num`` exceeds ``main_lane_num``.
    """
    if branch_lane_num > main_lane_num:
        raise ValueError("branch_lane_num cannot exceed main_lane_num.")
    if side == "right":
        start = main_lane_num - branch_lane_num
        lane_idxs = list(range(start, main_lane_num))
    else:
        lane_idxs = list(range(branch_lane_num))
    return lane_idxs, lane_idxs + [lane_idxs[-1] + 1]


def _branch_outer_point(
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

    Projects ``s_on_main`` proportionally onto the outer boundary polyline,
    then offsets inward by half the branch-road width so the branch centre
    line starts at the correct lateral position.

    Args:
        main_boundaries: Offset boundary polylines indexed 0 (left) to
            ``lane_num`` (right).
        side_boundaries: Boundary indices belonging to the branch side,
            as returned by :func:`_side_indices`.
        s_on_main: Diverge/merge arc-length position on the main centreline.
        main_length: Total arc length of the main centreline in metres.
        side: ``"right"`` or ``"left"``.
        branch_n: Number of branch lanes.
        lane_w: Lane width in metres.
        heading: Main-road heading at the diverge/merge point in radians.

    Returns:
        Anchor point ``(x, y)`` for the branch centreline.
    """
    outer_idx = side_boundaries[-1] if side == "right" else side_boundaries[0]
    boundary = main_boundaries[outer_idx]
    boundary_total = polyline.arc_length(boundary)
    boundary_s = 0.0 if main_length < 1e-9 else s_on_main / main_length * boundary_total
    outer_pt = polyline.point_at_arc_length(boundary, boundary_s)

    if side == "right":
        inward_normal = np.array([-np.sin(heading), np.cos(heading)], dtype=float)
    else:
        inward_normal = np.array([np.sin(heading), -np.cos(heading)], dtype=float)

    return outer_pt + inward_normal * (branch_n * lane_w / 2.0)


def _build_main_road_section(
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
    """Build main-road boundary roadlines and lanes for a fork or merge module.

    The outer boundary adjacent to the branch opening is split into a
    ``before`` segment and an ``after`` segment around the gap; all other
    boundaries are kept as continuous polylines.

    Args:
        main_n: Number of main-road lanes.
        main_boundaries: Full-length offset boundary polylines (length
            ``main_n + 1``, indexed left to right).
        outside_boundary_idx: Index of the boundary adjacent to the branch side.
        before_pts: Points of the outer boundary segment before the branch gap.
        after_pts: Points of the outer boundary segment after the branch gap.
        speed: Speed limit in km/h applied to all generated lanes.
        module: Module tag written into ``custom_tags`` (``"fork"`` or
            ``"merge"``).
        side_key: Custom-tag key for the branch side (``"fork_side"`` or
            ``"merge_side"``).
        side_value: Value for ``side_key`` (``"left"`` or ``"right"``).
        id_counter: Starting id counter.

    Returns:
        Tuple ``(main_lanes, roadlines, boundary_line_ids, updated_id_counter)``.
    """
    main_boundary_line_ids: list[list[str]] = []
    roadlines: list[RoadLine] = []

    for boundary_idx, boundary_pts in enumerate(main_boundaries):
        token = one_way_boundary_token(boundary_idx, main_n + 1)
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


def _build_branch_segment(
    cfg: _BranchConfig,
    main_in: RoadPort,
    main_out: RoadPort,
    branch_port: RoadPort,
    id_offset: int = 0,
) -> RoadModuleResult:
    """Build the shared geometry for fork (diverge) and merge (converge) modules.

    Args:
        cfg: Resolved branch parameters (kind, side, lane geometry, sampling).
        main_in: Upstream main-road socket.
        main_out: Downstream main-road socket.
        branch_port: Branch socket — ``branch_out`` for fork, ``branch_in`` for
            merge.
        id_offset: Starting element id counter.

    Returns:
        :class:`RoadModuleResult` containing all lanes, roadlines, and ports.
    """
    bk = cfg.bk
    side = cfg.side
    main_n = cfg.main_n
    branch_n = cfg.branch_n
    lane_w = cfg.lane_w
    speed = cfg.speed
    taper_length = cfg.taper_length
    branch_length = cfg.branch_length
    step_size = cfg.step_size
    s_ratio = cfg.s_ratio

    main_center = fit_reference_line(
        main_in.point, main_in.heading, main_out.point, main_out.heading, step_size
    )
    main_length = polyline.arc_length(main_center)

    if s_ratio is not None:
        s_on_main = float(np.clip(s_ratio, 0.0, 1.0)) * main_length
    else:
        s_on_main = bk.choose_s(
            main_center, np.asarray(branch_port.point, dtype=float), taper_length, branch_length
        )
    junction_heading = polyline.heading_at_arc_length(main_center, s_on_main)

    main_boundaries: list[np.ndarray] = [
        polyline.offset(main_center, boundary_offset(i, main_n, lane_w)) for i in range(main_n + 1)
    ]

    side_lanes, side_boundaries = _side_indices(main_n, branch_n, side)

    branch_anchor = _branch_outer_point(
        main_boundaries=main_boundaries,
        side_boundaries=side_boundaries,
        s_on_main=s_on_main,
        main_length=main_length,
        side=side,
        branch_n=branch_n,
        lane_w=lane_w,
        heading=junction_heading,
    )

    if bk.diverges:
        branch_center = fit_reference_line(
            branch_anchor,
            float(junction_heading),
            np.asarray(branch_port.point, dtype=float),
            float(branch_port.heading),
            step_size,
        )
    else:
        branch_chord = branch_anchor - np.asarray(branch_port.point, dtype=float)
        branch_chord_len = float(np.linalg.norm(branch_chord))
        if branch_chord_len > 1e-6:
            branch_depart_heading = float(np.arctan2(branch_chord[1], branch_chord[0]))
        else:
            branch_depart_heading = float(branch_port.heading)
        branch_center = fit_reference_line(
            np.asarray(branch_port.point, dtype=float),
            branch_depart_heading,
            branch_anchor,
            junction_heading,
            step_size,
        )

    branch_boundaries: list[np.ndarray] = [
        polyline.offset(branch_center, boundary_offset(i, branch_n, lane_w))
        for i in range(branch_n + 1)
    ]

    if side == "right":
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
    junction_pt: Point | None = None

    for local_idx, b_pts in enumerate(branch_boundaries):
        b_line = LineString(b_pts)
        intersect = polyline.find_intersection_point(b_line, main_out_line, pick=bk.pick)
        if intersect is None:
            branch_cut_lines.append(b_pts)
        else:
            branch_cut_lines.append(polyline.cut(b_line, intersect, bk.cut_dir))
            if local_idx == branch_inside_boundary_idx:
                nose_pt = intersect
            if local_idx == branch_outside_boundary_idx:
                junction_pt = intersect

    main_before_pts = main_outside_boundary
    main_after_pts = np.empty((0, 2), dtype=float)

    if nose_pt is not None and junction_pt is not None:
        s_nose = float(main_out_line.project(nose_pt))
        s_junction = float(main_out_line.project(junction_pt))
        if s_junction < s_nose:
            pt_first, pt_second = junction_pt, nose_pt
        else:
            pt_first, pt_second = nose_pt, junction_pt
        main_before_pts = polyline.cut(main_out_line, pt_first, "before")
        main_rest = polyline.cut(main_out_line, pt_first, "after")
        main_after_pts = polyline.cut(LineString(main_rest), pt_second, "after")

    marking_s = 0.0
    if nose_pt is not None:
        marking_s = float(
            LineString(branch_boundaries[branch_inside_boundary_idx]).project(nose_pt)
        )

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
        module=bk.name,
        side_key=bk.side_key,
        side_value=side,
        id_counter=id_counter,
    )
    lanes.extend(main_lanes)
    roadlines.extend(main_roadlines)

    branch_boundary_line_ids: list[list[str]] = []

    for local_idx, _ in enumerate(branch_boundaries):
        token = _branch_boundary_token(local_idx, branch_n + 1)
        if local_idx == branch_outside_boundary_idx:
            visibility_rule = f"outside_edge_{bk.edge_suffix}"
        elif local_idx == branch_inside_boundary_idx:
            visibility_rule = f"inside_edge_{bk.edge_suffix}"
        else:
            visibility_rule = f"interior_divider_{bk.interior_suffix}"

        roadline, id_counter = build_optional_roadline_from_points(
            id_counter,
            branch_cut_lines[local_idx],
            marking_kwargs=roadline_render_kwargs(
                token,
                {
                    "module": bk.name,
                    "submodule": "branch",
                    "boundary_index": local_idx,
                    bk.side_key: side,
                    f"{bk.meta_prefix}main_boundary_index": side_boundaries[local_idx],
                    "visibility_rule": visibility_rule,
                    bk.marking_s_key: float(marking_s),
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
        side_lane_idx = side_lanes[lane_idx]
        lane = build_lane_from_boundaries(
            id_=id_counter,
            left_points=branch_boundaries[lane_idx],
            right_points=branch_boundaries[lane_idx + 1],
            left_roadline_ids=branch_boundary_line_ids[lane_idx],
            right_roadline_ids=branch_boundary_line_ids[lane_idx + 1],
            speed_limit=min(speed, float(branch_port.speed_limit)),
            custom_tags={
                "module": bk.name,
                "submodule": "branch",
                "lane_index": lane_idx,
                f"{bk.meta_prefix}main_lane_index": side_lane_idx,
                f"{bk.meta_prefix}main_lane_id": main_lanes[side_lane_idx].id_,
                bk.side_key: side,
            },
        )
        id_counter += 1
        lanes.append(lane)
        branch_lanes.append(lane)

    add_ordered_lane_neighbors(branch_lanes)

    for local_idx, branch_lane in enumerate(branch_lanes):
        if bk.diverges:
            link_lanes(main_lanes[side_lanes[local_idx]], branch_lane)
        else:
            link_lanes(branch_lane, main_lanes[side_lanes[local_idx]])

    main_lane_ids = tuple(lane.id_ for lane in main_lanes)
    branch_lane_ids = tuple(lane.id_ for lane in branch_lanes)
    _meta = {"module": bk.name, bk.side_key: side}

    ports = {
        "main_in": build_port(
            RoadPort(main_in.point, main_in.heading, main_n, lane_w, speed),
            kind=bk.port_kind_main_in,
            name="main_in",
            lane_ids=main_lane_ids,
            metadata=_meta,
        ),
        "main_out": build_port(
            RoadPort(main_out.point, main_out.heading, main_n, lane_w, speed),
            kind=bk.port_kind_main_out,
            name="main_out",
            lane_ids=main_lane_ids,
            metadata=_meta,
        ),
        bk.port_name_branch: build_port(
            RoadPort(
                branch_port.point, branch_port.heading, branch_n, lane_w, branch_port.speed_limit
            ),
            kind=bk.port_kind_branch,
            name=bk.port_name_branch,
            lane_ids=branch_lane_ids,
            metadata=_meta,
        ),
    }

    return RoadModuleResult(lanes=lanes, roadlines=roadlines, ports=ports, id_counter=id_counter)


class Fork(RoadSegment):
    """One-way lane-level fork generator where traffic diverges from the main road.

    The branch geometry is computed by intersecting offset boundary polylines
    with the main road's outer edge, producing a nose point (where the branch
    inside boundary meets the main outer boundary) and a junction point (where
    the branch outside boundary meets the main outer boundary).  The main road's
    outer boundary is split around the opening; the branch lane runs from the
    junction anchor to ``branch_out``.

    Attributes:
        fork_side: Side of the main road where the branch leaves.
        taper_length: Minimum longitudinal distance reserved for the fork
            opening in metres.
        branch_length: Nominal branch length used for diverge-point inference
            in metres.
        step_size: Reference-line sampling interval in metres.
    """

    def __init__(
        self,
        fork_side: str = "right",
        taper_length: float = 65.0,
        branch_length: float = 85.0,
        step_size: float = 0.1,
    ) -> None:
        """Initialise the generator.

        Args:
            fork_side: Side where the branch leaves. Must be ``"left"`` or ``"right"``.
            taper_length: Minimum longitudinal distance reserved for the fork opening.
            branch_length: Nominal branch length for diverge-point inference.
            step_size: Reference-line sampling interval in metres.

        Raises:
            ValueError: If ``fork_side`` is not ``"left"`` or ``"right"``, or
                any length/interval parameter is non-positive.
        """
        if fork_side not in ("left", "right"):
            raise ValueError("fork_side must be 'left' or 'right'.")
        if taper_length <= 0.0:
            raise ValueError("taper_length must be positive.")
        if branch_length <= 0.0:
            raise ValueError("branch_length must be positive.")
        if step_size <= 0.0:
            raise ValueError("step_size must be positive.")
        self.fork_side = fork_side
        self.taper_length = taper_length
        self.branch_length = branch_length
        self.step_size = step_size

    def build(
        self,
        main_in: RoadPort,
        main_out: RoadPort,
        branch_out: RoadPort,
        *,
        main_lane_num: int | None = None,
        branch_lane_num: int | None = None,
        lane_width: float | None = None,
        speed_limit: float | None = None,
        diverge_s_ratio: float | None = None,
        id_offset: int = 0,
    ) -> RoadModuleResult:
        """Build a fork module from the given port sockets.

        Args:
            main_in: Upstream main-road socket.
            main_out: Downstream main-road socket.
            branch_out: Downstream branch socket.
            main_lane_num: Number of main-road lanes. Defaults to ``main_in.lane_num``.
            branch_lane_num: Number of branch lanes. Defaults to
                ``branch_out.lane_num``.
            lane_width: Lane width in metres. Defaults to ``main_in.lane_width``.
            speed_limit: Main-road speed limit in km/h. Defaults to
                ``main_in.speed_limit``.
            diverge_s_ratio: Optional normalised diverge position on the main
                reference line. When omitted, inferred from ``branch_out``.
            id_offset: First element id.

        Returns:
            :class:`RoadModuleResult` with ports ``"main_in"``, ``"main_out"``,
            and ``"branch_out"``.

        Raises:
            ValueError: If ``branch_lane_num`` exceeds ``main_lane_num``, or
                any lane count or width parameter is invalid.
        """
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

        return _build_branch_segment(
            _BranchConfig(
                bk=_FORK,
                side=self.fork_side,
                main_n=main_n,
                branch_n=branch_n,
                lane_w=lane_w,
                speed=speed,
                taper_length=self.taper_length,
                branch_length=self.branch_length,
                step_size=self.step_size,
                s_ratio=diverge_s_ratio,
            ),
            main_in,
            main_out,
            branch_out,
            id_offset=id_offset,
        )


class Merge(RoadSegment):
    """One-way lane-level merge generator where traffic converges onto the main road.

    The branch geometry mirrors :class:`Fork`: branch boundaries are intersected
    with the main outer edge to find the nose and junction points, the main
    outer boundary is split, and the branch lane runs from ``branch_in`` to the
    junction anchor.

    Attributes:
        merge_side: Side of the main road where the branch enters.
        taper_length: Minimum longitudinal distance reserved for the merge
            opening in metres.
        branch_length: Nominal branch length used for merge-point inference
            in metres.
        step_size: Reference-line sampling interval in metres.
    """

    def __init__(
        self,
        merge_side: str = "right",
        taper_length: float = 65.0,
        branch_length: float = 85.0,
        step_size: float = 0.1,
    ) -> None:
        """Initialise the generator.

        Args:
            merge_side: Side where the branch enters. Must be ``"left"`` or ``"right"``.
            taper_length: Minimum longitudinal distance reserved for the merge opening.
            branch_length: Nominal branch length for merge-point inference.
            step_size: Reference-line sampling interval in metres.

        Raises:
            ValueError: If ``merge_side`` is not ``"left"`` or ``"right"``, or
                any length/interval parameter is non-positive.
        """
        if merge_side not in ("left", "right"):
            raise ValueError("merge_side must be 'left' or 'right'.")
        if taper_length <= 0.0:
            raise ValueError("taper_length must be positive.")
        if branch_length <= 0.0:
            raise ValueError("branch_length must be positive.")
        if step_size <= 0.0:
            raise ValueError("step_size must be positive.")
        self.merge_side = merge_side
        self.taper_length = taper_length
        self.branch_length = branch_length
        self.step_size = step_size

    def build(
        self,
        main_in: RoadPort,
        branch_in: RoadPort,
        main_out: RoadPort,
        *,
        main_lane_num: int | None = None,
        branch_lane_num: int | None = None,
        lane_width: float | None = None,
        speed_limit: float | None = None,
        merge_s_ratio: float | None = None,
        id_offset: int = 0,
    ) -> RoadModuleResult:
        """Build a merge module from the given port sockets.

        Args:
            main_in: Upstream main-road socket.
            branch_in: Upstream branch socket.
            main_out: Downstream main-road socket.
            main_lane_num: Number of main-road lanes. Defaults to ``main_in.lane_num``.
            branch_lane_num: Number of branch lanes. Defaults to
                ``branch_in.lane_num``.
            lane_width: Lane width in metres. Defaults to ``main_in.lane_width``.
            speed_limit: Main-road speed limit in km/h. Defaults to
                ``main_in.speed_limit``.
            merge_s_ratio: Optional normalised merge position on the main
                reference line. When omitted, inferred from ``branch_in``.
            id_offset: First element id.

        Returns:
            :class:`RoadModuleResult` with ports ``"main_in"``, ``"branch_in"``,
            and ``"main_out"``.

        Raises:
            ValueError: If ``branch_lane_num`` exceeds ``main_lane_num``, or
                any lane count or width parameter is invalid.
        """
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

        return _build_branch_segment(
            _BranchConfig(
                bk=_MERGE,
                side=self.merge_side,
                main_n=main_n,
                branch_n=branch_n,
                lane_w=lane_w,
                speed=speed,
                taper_length=self.taper_length,
                branch_length=self.branch_length,
                step_size=self.step_size,
                s_ratio=merge_s_ratio,
            ),
            main_in,
            main_out,
            branch_in,
            id_offset=id_offset,
        )
