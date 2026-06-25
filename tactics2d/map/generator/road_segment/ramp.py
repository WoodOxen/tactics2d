# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Socket-driven entrance and exit ramp generators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon

from tactics2d.geometry import (
    as_point,
    nearest_s,
    normalize_angle,
    offset_polyline,
    polyline_length,
    sample_by_s,
)
from tactics2d.map.element import Area, Lane, LaneRelationship, RoadLine
from tactics2d.map.generator.rules.lane_marking_rules import (
    one_way_boundary_token,
    ramp_mark,
    roadline_render_kwargs,
    two_way_backward,
    two_way_centerline,
    two_way_forward,
)
from tactics2d.map.generator.rules.module_types import (
    MainRoadType,
    RampKind,
    RampSide,
    RoadModuleResult,
    RoadPort,
    build_port,
)

from .element_builder import (
    add_ordered_lane_neighbors,
    boundary_offset,
    build_lane_from_boundaries,
    build_roadline_from_points,
    build_segmented_roadline,
    link_lanes,
)
from .reference_line import bezier_connection, fit_reference_line
from .road_segment import RoadSegment


@dataclass
class MainRoadSection:
    """Internal geometry result from main-road builders used by ramp generators.

    Attributes:
        lanes: All generated Lane objects (forward + backward).
        roadlines: All generated RoadLine objects.
        forward_lanes: Forward-direction lanes only.
        backward_lanes: Backward-direction lanes; empty for freeway type.
        attach_boundary: Edge boundary polyline adjacent to the ramp.
        attach_lane: The forward lane directly adjacent to the ramp.
        id_counter: Next available id counter.
    """

    lanes: list[Lane]
    roadlines: list[RoadLine]
    forward_lanes: list[Lane]
    backward_lanes: list[Lane]
    attach_boundary: np.ndarray
    attach_lane: Lane
    id_counter: int


def _repair_heading_towards_chord(
    start_point: np.ndarray, end_point: np.ndarray, heading: float, max_error: float = np.pi / 3.0
) -> float:
    """Repair a ramp endpoint heading that is inconsistent with the chord direction.

    When the angular deviation between ``heading`` and the chord exceeds
    ``max_error``, the chord heading is substituted to prevent extreme Bezier
    curves in the connector lane.

    Args:
        start_point: Connector start point ``(x, y)``.
        end_point: Connector end point ``(x, y)``.
        heading: Original heading in radians.
        max_error: Maximum tolerated angular deviation in radians.

    Returns:
        Corrected heading in radians.
    """
    start_point = as_point(start_point)
    end_point = as_point(end_point)
    chord = end_point - start_point

    if np.linalg.norm(chord) < 1e-6:
        return float(heading)

    chord_heading = float(np.arctan2(chord[1], chord[0]))
    if abs(normalize_angle(float(heading) - chord_heading)) > max_error:
        return chord_heading

    return float(heading)


def _build_freeway_main_road(
    center_pts: np.ndarray,
    lane_num: int,
    lane_width: float,
    speed_limit: float,
    id_counter: int,
    step_size: float,
    ramp_side: RampSide,
    edge_gap: tuple[float, float] | None = None,
) -> MainRoadSection:
    """Build a one-way freeway mainline from a carriageway reference line.

    Args:
        center_pts: Sampled carriageway centreline points.
        lane_num: Number of forward lanes.
        lane_width: Width per lane in metres.
        speed_limit: Speed limit in km/h.
        id_counter: Starting id counter.
        step_size: Sampling interval used for segmented boundary roadlines.
        ramp_side: Side where the ramp auxiliary lane attaches; the boundary on
            that side receives the ``edge_gap`` opening.
        edge_gap: ``(start_s, end_s)`` gap interval on the ramp-side outer edge,
            or ``None`` for a solid edge.

    Returns:
        :class:`MainRoadSection` where ``backward_lanes`` is always empty.
    """
    if lane_num < 1:
        raise ValueError("lane_num must be greater than or equal to 1.")

    lanes: list[Lane] = []
    roadlines: list[RoadLine] = []

    boundary_num = lane_num + 1
    boundary_pts = [
        offset_polyline(center_pts, boundary_offset(i, lane_num, lane_width))
        for i in range(boundary_num)
    ]
    boundary_line_ids: list[list[str]] = []

    for boundary_idx, pts in enumerate(boundary_pts):
        is_left_edge = boundary_idx == 0
        is_right_edge = boundary_idx == lane_num

        marking_kwargs = roadline_render_kwargs(one_way_boundary_token(boundary_idx, boundary_num))
        if is_left_edge:
            side = "left"
            gap = edge_gap if ramp_side == "left" else None
        elif is_right_edge:
            side = "right"
            gap = edge_gap if ramp_side == "right" else None
        else:
            side = "interior"
            gap = None

        generated_roadlines, generated_ids, id_counter = build_segmented_roadline(
            pts,
            gap,
            marking_kwargs,
            id_counter,
            step_size,
            {
                "module": "ramp",
                "submodule": "main",
                "main_road_type": "freeway",
                "direction": "forward",
                "boundary_index": boundary_idx,
                "side": side,
            },
        )
        roadlines.extend(generated_roadlines)
        boundary_line_ids.append(generated_ids)

    for i in range(lane_num):
        lane = build_lane_from_boundaries(
            id_=id_counter,
            left_points=boundary_pts[i],
            right_points=boundary_pts[i + 1],
            left_roadline_ids=boundary_line_ids[i],
            right_roadline_ids=boundary_line_ids[i + 1],
            speed_limit=speed_limit,
            custom_tags={
                "module": "ramp",
                "submodule": "main",
                "main_road_type": "freeway",
                "direction": "forward",
                "lane_index": i,
            },
        )
        id_counter += 1
        lanes.append(lane)

    add_ordered_lane_neighbors(lanes)

    if ramp_side == "right":
        attach_boundary = boundary_pts[-1]
        attach_lane = lanes[-1]
    else:
        attach_boundary = boundary_pts[0]
        attach_lane = lanes[0]

    return MainRoadSection(
        lanes=lanes,
        roadlines=roadlines,
        forward_lanes=lanes,
        backward_lanes=[],
        attach_boundary=attach_boundary,
        attach_lane=attach_lane,
        id_counter=id_counter,
    )


def _build_urban_main_road(
    center_pts: np.ndarray,
    forward_lane_num: int,
    backward_lane_num: int,
    lane_width: float,
    speed_limit: float,
    id_counter: int,
    step_size: float,
    ramp_side: RampSide,
    edge_gap: tuple[float, float] | None = None,
) -> MainRoadSection:
    """Build a two-way urban main road from a centerline.

    Args:
        center_pts: Sampled carriageway centreline points.
        forward_lane_num: Number of forward (ramp-side) lanes.
        backward_lane_num: Number of backward lanes on the opposite side.
        lane_width: Width per lane in metres.
        speed_limit: Speed limit in km/h.
        id_counter: Starting id counter.
        step_size: Sampling interval used for segmented boundary roadlines.
        ramp_side: Side where the ramp auxiliary lane attaches. Currently only
            ``"right"`` is supported.
        edge_gap: ``(start_s, end_s)`` gap interval on the ramp-side forward
            outer edge, or ``None`` for a solid edge.

    Returns:
        :class:`MainRoadSection` with forward and backward lanes populated.
    """
    if ramp_side != "right":
        raise ValueError("urban ramp currently supports only ramp_side='right'.")

    lanes: list[Lane] = []
    roadlines: list[RoadLine] = []

    center_roadline = build_roadline_from_points(
        id_=id_counter,
        points=center_pts,
        marking_kwargs=roadline_render_kwargs(
            two_way_centerline(forward_lane_num, backward_lane_num, no_passing=True)
        ),
        custom_tags={
            "module": "ramp",
            "submodule": "main",
            "main_road_type": "urban",
            "marking_role": "centerline",
            "boundary_index": 0,
        },
    )
    id_counter += 1
    roadlines.append(center_roadline)

    forward_boundary_pts = [
        offset_polyline(center_pts, -i * lane_width) for i in range(forward_lane_num + 1)
    ]
    forward_boundary_line_ids: list[list[str]] = [[center_roadline.id_]]

    for boundary_idx in range(1, forward_lane_num + 1):
        pts = forward_boundary_pts[boundary_idx]
        is_outer_edge = boundary_idx == forward_lane_num
        marking_kwargs = roadline_render_kwargs(
            two_way_forward(boundary_idx - 1, forward_lane_num, "right")
        )

        generated_roadlines, generated_ids, id_counter = build_segmented_roadline(
            pts,
            edge_gap if is_outer_edge else None,
            marking_kwargs,
            id_counter,
            step_size,
            {
                "module": "ramp",
                "submodule": "main",
                "main_road_type": "urban",
                "direction": "forward",
                "boundary_index": boundary_idx,
                "side": "right" if is_outer_edge else "interior",
            },
        )
        roadlines.extend(generated_roadlines)
        forward_boundary_line_ids.append(generated_ids)

    forward_lanes: list[Lane] = []
    for i in range(forward_lane_num):
        lane = build_lane_from_boundaries(
            id_=id_counter,
            left_points=forward_boundary_pts[i],
            right_points=forward_boundary_pts[i + 1],
            left_roadline_ids=forward_boundary_line_ids[i],
            right_roadline_ids=forward_boundary_line_ids[i + 1],
            speed_limit=speed_limit,
            custom_tags={
                "module": "ramp",
                "submodule": "main",
                "main_road_type": "urban",
                "direction": "forward",
                "lane_index": i,
            },
        )
        id_counter += 1
        forward_lanes.append(lane)
        lanes.append(lane)

    add_ordered_lane_neighbors(forward_lanes)

    backward_boundary_pts_raw = [
        offset_polyline(center_pts, i * lane_width) for i in range(backward_lane_num + 1)
    ]
    backward_boundary_line_ids: list[list[str]] = [[center_roadline.id_]]

    for boundary_idx in range(1, backward_lane_num + 1):
        pts = backward_boundary_pts_raw[boundary_idx]
        is_outer_edge = boundary_idx == backward_lane_num
        marking_kwargs = roadline_render_kwargs(
            two_way_backward(boundary_idx - 1, backward_lane_num, "left")
        )

        roadline = build_roadline_from_points(
            id_=id_counter,
            points=pts,
            marking_kwargs=marking_kwargs,
            custom_tags={
                "module": "ramp",
                "submodule": "main",
                "main_road_type": "urban",
                "direction": "backward",
                "boundary_index": boundary_idx,
                "side": "left" if is_outer_edge else "interior",
            },
        )
        id_counter += 1
        roadlines.append(roadline)
        backward_boundary_line_ids.append([roadline.id_])

    backward_lanes: list[Lane] = []
    for i in range(backward_lane_num):
        left_pts = backward_boundary_pts_raw[i][::-1]
        right_pts = backward_boundary_pts_raw[i + 1][::-1]

        lane = build_lane_from_boundaries(
            id_=id_counter,
            left_points=left_pts,
            right_points=right_pts,
            left_roadline_ids=backward_boundary_line_ids[i],
            right_roadline_ids=backward_boundary_line_ids[i + 1],
            speed_limit=speed_limit,
            custom_tags={
                "module": "ramp",
                "submodule": "main",
                "main_road_type": "urban",
                "direction": "backward",
                "lane_index": i,
            },
        )
        id_counter += 1
        backward_lanes.append(lane)
        lanes.append(lane)

    add_ordered_lane_neighbors(
        backward_lanes,
        left_relationship=LaneRelationship.RIGHT_NEIGHBOR,
        right_relationship=LaneRelationship.LEFT_NEIGHBOR,
    )

    return MainRoadSection(
        lanes=lanes,
        roadlines=roadlines,
        forward_lanes=forward_lanes,
        backward_lanes=backward_lanes,
        attach_boundary=forward_boundary_pts[-1],
        attach_lane=forward_lanes[-1],
        id_counter=id_counter,
    )


def _build_single_lane_from_center(
    center_pts: np.ndarray,
    lane_width: float,
    speed_limit: float,
    left_marking_kwargs: dict,
    right_marking_kwargs: dict,
    id_counter: int,
    tags: dict,
) -> tuple[Lane, list[RoadLine], int]:
    """Build a single lane symmetrically around a centreline.

    Args:
        center_pts: Lane centreline points. Left and right boundaries are offset
            by ``lane_width / 2`` on each side.
        lane_width: Total lane width in metres.
        speed_limit: Speed limit in km/h.
        left_marking_kwargs: Marking style dict for the left boundary RoadLine.
        right_marking_kwargs: Marking style dict for the right boundary RoadLine.
        id_counter: Starting id counter.
        tags: Extra custom_tags merged into all three generated elements.

    Returns:
        Tuple ``(lane, [left_roadline, right_roadline], updated_id_counter)``.
    """
    left_pts = offset_polyline(center_pts, lane_width / 2.0)
    right_pts = offset_polyline(center_pts, -lane_width / 2.0)

    left_roadline = build_roadline_from_points(
        id_=id_counter,
        points=left_pts,
        marking_kwargs=left_marking_kwargs,
        custom_tags={**tags, "side": "left"},
    )
    id_counter += 1

    right_roadline = build_roadline_from_points(
        id_=id_counter,
        points=right_pts,
        marking_kwargs=right_marking_kwargs,
        custom_tags={**tags, "side": "right"},
    )
    id_counter += 1

    lane = build_lane_from_boundaries(
        id_=id_counter,
        left_points=left_pts,
        right_points=right_pts,
        left_roadline_ids=left_roadline.id_,
        right_roadline_ids=right_roadline.id_,
        speed_limit=speed_limit,
        custom_tags=tags,
    )
    id_counter += 1

    return lane, [left_roadline, right_roadline], id_counter


def _compute_ramp_s_intervals(
    kind: RampKind,
    ramp_s: float,
    boundary_length: float,
    taper_length: float,
    parallel_length: float,
) -> tuple[float, float, float, float, tuple[float, float]]:
    """Compute longitudinal s-intervals for the ramp auxiliary lane on the main boundary.

    Args:
        kind: ``"exit"`` or ``"entrance"``.
        ramp_s: Arc-length of the ramp attachment point on the boundary polyline.
        boundary_length: Total arc-length of the boundary polyline in metres.
        taper_length: Taper zone length in metres.
        parallel_length: Parallel zone length in metres.

    Returns:
        Tuple ``(parallel_s_start, parallel_s_end, taper_s_start, taper_s_end,
        edge_gap)`` where ``edge_gap`` is the ``(start, end)`` gap interval
        passed to the segmented boundary builder.
    """
    if kind == "exit":
        parallel_s_end = float(np.clip(ramp_s, taper_length + parallel_length, boundary_length))
        parallel_s_start = parallel_s_end - parallel_length
        taper_s_start = parallel_s_start - taper_length
        taper_s_end = parallel_s_start
        edge_gap = (taper_s_start, parallel_s_end)
    else:
        parallel_s_start = float(
            np.clip(ramp_s, 0.0, boundary_length - parallel_length - taper_length)
        )
        parallel_s_end = parallel_s_start + parallel_length
        taper_s_start = parallel_s_end
        taper_s_end = taper_s_start + taper_length
        edge_gap = (parallel_s_start, taper_s_end)

    return parallel_s_start, parallel_s_end, taper_s_start, taper_s_end, edge_gap


@dataclass
class _AuxLaneGeometry:
    """Auxiliary lane and connector geometry for one ramp kind.

    Attributes:
        aux_inner: Inner (main-road-side) boundary polyline of the auxiliary lane.
        aux_outer: Outer (ramp-side) boundary polyline of the auxiliary lane.
        connector_start: Start point of the connector lane centreline.
        connector_start_heading: Heading at ``connector_start`` in radians.
        connector_end: End point of the connector lane centreline.
        connector_end_heading: Heading at ``connector_end`` in radians.
        aux_tag: Custom-tag segment label for auxiliary lane elements.
        connector_tag: Custom-tag segment label for connector lane elements.
        ramp_port_kind: Port kind string for the ramp output socket.
    """

    aux_inner: np.ndarray
    aux_outer: np.ndarray
    connector_start: np.ndarray
    connector_start_heading: float
    connector_end: np.ndarray
    connector_end_heading: float
    aux_tag: str
    connector_tag: str
    ramp_port_kind: str


def _build_aux_geometry(
    kind: RampKind,
    attach_boundary: np.ndarray,
    taper_s_start: float,
    taper_s_end: float,
    parallel_s_start: float,
    parallel_s_end: float,
    step_size: float,
    lane_w: float,
    ramp_side: RampSide,
    ramp_port: RoadPort,
) -> _AuxLaneGeometry:
    """Compute ramp auxiliary lane and connector geometry.

    For exit ramps the taper precedes the parallel zone (upstream); for
    entrance ramps the parallel zone precedes the taper (downstream).  The
    connector bridges the parallel-zone tip to the ramp socket.

    Args:
        kind: ``"exit"`` or ``"entrance"``.
        attach_boundary: Main-road edge boundary polyline on the ramp side.
        taper_s_start: Arc-length start of the taper zone.
        taper_s_end: Arc-length end of the taper zone.
        parallel_s_start: Arc-length start of the parallel zone.
        parallel_s_end: Arc-length end of the parallel zone.
        step_size: Bezier sampling interval in metres.
        lane_w: Lane width in metres.
        ramp_side: Side of the main road where the ramp attaches.
        ramp_port: Ramp output/input socket.

    Returns:
        :class:`_AuxLaneGeometry` for the configured ramp kind.
    """
    n_taper = max(8, int((taper_s_end - taper_s_start) / step_size) + 1)
    taper_pos, taper_hdgs, _ = sample_by_s(attach_boundary, taper_s_start, taper_s_end, n_taper)

    n_parallel = max(4, int((parallel_s_end - parallel_s_start) / step_size) + 1)
    parallel_pos, parallel_hdgs, parallel_right_normals = sample_by_s(
        attach_boundary, parallel_s_start, parallel_s_end, n_parallel
    )

    out_normals = parallel_right_normals if ramp_side == "right" else -parallel_right_normals
    parallel_outer = parallel_pos + lane_w * out_normals

    if kind == "exit":
        taper_outer = bezier_connection(
            taper_pos[0],
            float(taper_hdgs[0]),
            parallel_outer[0],
            float(parallel_hdgs[0]),
            step_size,
        )
        connector_start = parallel_pos[-1] + 0.5 * lane_w * out_normals[-1]
        connector_end = as_point(ramp_port.point)
        connector_end_heading = _repair_heading_towards_chord(
            connector_start, connector_end, float(ramp_port.heading)
        )
        return _AuxLaneGeometry(
            aux_inner=np.vstack([taper_pos, parallel_pos[1:]]),
            aux_outer=np.vstack([taper_outer, parallel_outer[1:]]),
            connector_start=connector_start,
            connector_start_heading=float(parallel_hdgs[-1]),
            connector_end=connector_end,
            connector_end_heading=connector_end_heading,
            aux_tag="exit_auxiliary_taper_parallel",
            connector_tag="exit_connector",
            ramp_port_kind="ramp_out",
        )

    taper_outer = bezier_connection(
        parallel_outer[-1],
        float(parallel_hdgs[-1]),
        taper_pos[-1],
        float(taper_hdgs[-1]),
        step_size,
    )
    connector_end = parallel_pos[0] + 0.5 * lane_w * out_normals[0]
    connector_start = as_point(ramp_port.point)
    connector_start_heading = _repair_heading_towards_chord(
        connector_start, connector_end, float(ramp_port.heading)
    )
    return _AuxLaneGeometry(
        aux_inner=np.vstack([parallel_pos, taper_pos[1:]]),
        aux_outer=np.vstack([parallel_outer, taper_outer[1:]]),
        connector_start=connector_start,
        connector_start_heading=connector_start_heading,
        connector_end=connector_end,
        connector_end_heading=float(parallel_hdgs[0]),
        aux_tag="entrance_parallel_taper",
        connector_tag="entrance_connector",
        ramp_port_kind="ramp_in",
    )


def _build_ramp(
    *,
    kind: RampKind,
    main_road_type: MainRoadType,
    ramp_side: RampSide,
    main_in: RoadPort,
    main_out: RoadPort,
    ramp_port: RoadPort,
    backward_lane_num: int,
    lane_width: float | None,
    main_speed_limit: float | None,
    ramp_speed_limit: float | None,
    taper_length: float,
    parallel_length: float,
    step_size: float,
    id_offset: int,
) -> RoadModuleResult:
    """Shared core builder for all six public ramp generators.

    This function is the single implementation backing ``exit_ramp()``,
    ``entrance_ramp()``, and their four ``freeway_*/urban_*`` aliases.

    Geometry overview
    -----------------
    A ramp consists of three longitudinal elements laid side-by-side on the
    main road:

    1. **Main road** (``_build_freeway_main_road`` or ``_build_urban_main_road``):
       Spans from ``main_in`` to ``main_out`` with a boundary gap on the ramp
       side marking the taper and parallel sections.
    2. **Auxiliary lane**: A full-width lane that runs parallel to the main road
       (``parallel_length``) and then tapers to/from zero width
       (``taper_length``).  For exit, taper is at the upstream end; for
       entrance, taper is at the downstream end.
    3. **Connector lane**: A bezier-curved single lane connecting the auxiliary
       lane tip to the ``ramp_port`` socket.

    Args:
        kind: ``"exit"`` (traffic leaves main road) or ``"entrance"``
            (traffic joins main road).
        main_road_type: ``"freeway"`` for a one-way carriageway;
            ``"urban"`` for a two-way road with a backward direction.
        ramp_side: Side of the main road where the ramp attaches.
            Only ``"right"`` is supported for urban type.
        main_in: Upstream main-road socket.
        main_out: Downstream main-road socket.
        ramp_port: Ramp socket — ``ramp_out`` for exit, ``ramp_in`` for
            entrance.
        backward_lane_num: Number of backward lanes; ignored for freeway.
        lane_width: Lane width in metres. Defaults to ``main_in.lane_width``.
        main_speed_limit: Main-road speed limit. Defaults to
            ``main_in.speed_limit``.
        ramp_speed_limit: Ramp speed limit. Defaults to
            ``ramp_port.speed_limit``.
        taper_length: Longitudinal length of the auxiliary-lane taper zone.
        parallel_length: Longitudinal length of the full-width auxiliary lane
            running parallel to the main road.
        step_size: Reference-line sampling interval in metres.
        id_offset: First id used for generated map elements.

    Returns:
        A ``RoadModuleResult`` with ports ``"main_in"``, ``"main_out"``,
        and ``"ramp"`` (plus ``"backward_in"``/``"backward_out"`` for urban).
    """
    if kind not in ("exit", "entrance"):
        raise ValueError("kind must be 'exit' or 'entrance'.")
    if main_in.lane_num != main_out.lane_num:
        raise ValueError("main_in and main_out must have the same lane_num.")

    lane_num = int(main_in.lane_num)
    lane_w = float(lane_width if lane_width is not None else main_in.lane_width)
    main_speed = float(main_speed_limit if main_speed_limit is not None else main_in.speed_limit)
    ramp_speed = float(ramp_speed_limit if ramp_speed_limit is not None else ramp_port.speed_limit)

    center_pts = fit_reference_line(
        main_in.point, main_in.heading, main_out.point, main_out.heading, step_size
    )

    if main_road_type == "freeway":
        total_half_width = lane_num * lane_w / 2.0
        ramp_edge_offset = -total_half_width if ramp_side == "right" else total_half_width
    else:
        ramp_edge_offset = -lane_num * lane_w

    boundary_preview = offset_polyline(center_pts, ramp_edge_offset)
    boundary_length = polyline_length(boundary_preview)
    ramp_s = nearest_s(boundary_preview, as_point(ramp_port.point))

    parallel_s_start, parallel_s_end, taper_s_start, taper_s_end, edge_gap = (
        _compute_ramp_s_intervals(kind, ramp_s, boundary_length, taper_length, parallel_length)
    )

    if main_road_type == "freeway":
        main_section = _build_freeway_main_road(
            center_pts, lane_num, lane_w, main_speed, id_offset, step_size, ramp_side, edge_gap
        )
    else:
        main_section = _build_urban_main_road(
            center_pts,
            lane_num,
            backward_lane_num,
            lane_w,
            main_speed,
            id_offset,
            step_size,
            ramp_side,
            edge_gap,
        )

    lanes = list(main_section.lanes)
    roadlines = list(main_section.roadlines)
    forward_lanes = main_section.forward_lanes
    backward_lanes = main_section.backward_lanes
    attach_boundary = main_section.attach_boundary
    attach_lane = main_section.attach_lane
    id_counter = main_section.id_counter

    aux_geom = _build_aux_geometry(
        kind,
        attach_boundary,
        taper_s_start,
        taper_s_end,
        parallel_s_start,
        parallel_s_end,
        step_size,
        lane_w,
        ramp_side,
        ramp_port,
    )

    connector_center = bezier_connection(
        aux_geom.connector_start,
        aux_geom.connector_start_heading,
        aux_geom.connector_end,
        aux_geom.connector_end_heading,
        step_size,
    )

    if ramp_side == "right":
        aux_left, aux_right = aux_geom.aux_inner, aux_geom.aux_outer
        aux_left_marking = roadline_render_kwargs(ramp_mark("aux_left"))
        aux_right_marking = roadline_render_kwargs(ramp_mark("aux_right"))
    else:
        aux_left, aux_right = aux_geom.aux_outer, aux_geom.aux_inner
        aux_left_marking = roadline_render_kwargs(ramp_mark("aux_right"))
        aux_right_marking = roadline_render_kwargs(ramp_mark("aux_left"))

    aux_left_roadline = build_roadline_from_points(
        id_=id_counter,
        points=aux_left,
        marking_kwargs=aux_left_marking,
        custom_tags={"module": "ramp", "segment": aux_geom.aux_tag, "side": "left"},
    )
    id_counter += 1

    aux_right_roadline = build_roadline_from_points(
        id_=id_counter,
        points=aux_right,
        marking_kwargs=aux_right_marking,
        custom_tags={"module": "ramp", "segment": aux_geom.aux_tag, "side": "right"},
    )
    id_counter += 1

    aux_lane = build_lane_from_boundaries(
        id_=id_counter,
        left_points=aux_left,
        right_points=aux_right,
        left_roadline_ids=aux_left_roadline.id_,
        right_roadline_ids=aux_right_roadline.id_,
        speed_limit=ramp_speed,
        custom_tags={"module": "ramp"},
    )
    id_counter += 1

    lanes.append(aux_lane)
    roadlines.extend([aux_left_roadline, aux_right_roadline])

    connector_lane, connector_roadlines, id_counter = _build_single_lane_from_center(
        connector_center,
        lane_w,
        ramp_speed,
        roadline_render_kwargs(ramp_mark("left_edge")),
        roadline_render_kwargs(ramp_mark("right_edge")),
        id_counter,
        tags={
            "module": "ramp",
            "kind": kind,
            "main_road_type": main_road_type,
            "ramp_side": ramp_side,
            "segment": aux_geom.connector_tag,
        },
    )
    lanes.append(connector_lane)
    roadlines.extend(connector_roadlines)

    areas: list[Area] = []
    gore_length = 25.0

    if kind == "exit":
        gore_s_start = parallel_s_end
        gore_s_end = min(parallel_s_end + gore_length, boundary_length)
    else:
        gore_s_end = parallel_s_start
        gore_s_start = max(parallel_s_start - gore_length, 0.0)

    if gore_s_end > gore_s_start:
        n_gore = max(5, int(gore_length / step_size))
        gore_main_pts, _, _ = sample_by_s(attach_boundary, gore_s_start, gore_s_end, n_gore)
        connector_side = (
            connector_lane.left_side if ramp_side == "right" else connector_lane.right_side
        )
        connector_inner_pts = np.array(connector_side.coords)

        n = min(len(gore_main_pts), len(connector_inner_pts))
        if n >= 3:
            gore_ramp_pts = connector_inner_pts[:n] if kind == "exit" else connector_inner_pts[-n:]
            gore_polygon_pts = np.vstack([gore_main_pts[:n], gore_ramp_pts[::-1]])
            areas.append(
                Area(
                    id_=str(id_counter),
                    geometry=Polygon(gore_polygon_pts),
                    type_="multipolygon",
                    subtype="gore",
                    color="none",
                    custom_tags={"module": "ramp", "role": "gore_island"},
                )
            )
            id_counter += 1

    if ramp_side == "right":
        main_to_aux_rel, aux_to_main_rel = (
            LaneRelationship.RIGHT_NEIGHBOR,
            LaneRelationship.LEFT_NEIGHBOR,
        )
    else:
        main_to_aux_rel, aux_to_main_rel = (
            LaneRelationship.LEFT_NEIGHBOR,
            LaneRelationship.RIGHT_NEIGHBOR,
        )
    attach_lane.add_related_lane(aux_lane.id_, main_to_aux_rel)
    aux_lane.add_related_lane(attach_lane.id_, aux_to_main_rel)

    if kind == "exit":
        link_lanes(attach_lane, aux_lane)
        link_lanes(aux_lane, connector_lane)
    else:
        link_lanes(connector_lane, aux_lane)
        link_lanes(aux_lane, attach_lane)

    forward_ids = tuple(lane.id_ for lane in forward_lanes)
    backward_ids = tuple(lane.id_ for lane in backward_lanes)
    ramp_ids = (connector_lane.id_,)

    ports = {
        "main_in": build_port(main_in, kind="main_in", name="main_in", lane_ids=forward_ids),
        "main_out": build_port(main_out, kind="main_out", name="main_out", lane_ids=forward_ids),
        "ramp": build_port(
            ramp_port, kind=aux_geom.ramp_port_kind, name=aux_geom.ramp_port_kind, lane_ids=ramp_ids
        ),
    }

    if main_road_type == "urban":
        reverse_main_in = RoadPort(
            point=np.asarray(main_out.point, dtype=float),
            heading=float(main_out.heading + np.pi),
            lane_num=backward_lane_num,
            lane_width=lane_w,
            speed_limit=main_speed,
        )
        reverse_main_out = RoadPort(
            point=np.asarray(main_in.point, dtype=float),
            heading=float(main_in.heading + np.pi),
            lane_num=backward_lane_num,
            lane_width=lane_w,
            speed_limit=main_speed,
        )
        ports["backward_in"] = build_port(
            reverse_main_in, kind="backward_in", name="backward_in", lane_ids=backward_ids
        )
        ports["backward_out"] = build_port(
            reverse_main_out, kind="backward_out", name="backward_out", lane_ids=backward_ids
        )

    return RoadModuleResult(
        lanes=lanes, roadlines=roadlines, ports=ports, id_counter=id_counter, areas=areas
    )


class _RampBase(RoadSegment):
    """Shared initialisation for :class:`ExitRamp` and :class:`EntranceRamp`.

    Attributes:
        main_road_type: ``"freeway"`` for a one-way carriageway; ``"urban"``
            for a two-way road with a backward direction.
        ramp_side: Side of the main road where the ramp attaches (``"left"``
            or ``"right"``). Urban ramps support ``"right"`` only.
        backward_lane_num: Number of backward lanes; ignored when
            ``main_road_type`` is ``"freeway"``.
        taper_length: Longitudinal length of the auxiliary lane taper in metres.
        parallel_length: Longitudinal length of the full-width auxiliary lane
            running parallel to the main road in metres.
        step_size: Reference-line sampling interval in metres.
    """

    def __init__(
        self,
        main_road_type: MainRoadType = "freeway",
        ramp_side: RampSide = "right",
        backward_lane_num: int = 1,
        taper_length: float = 50.0,
        parallel_length: float = 60.0,
        step_size: float = 0.5,
    ) -> None:
        """Initialise the generator.

        Args:
            main_road_type: ``"freeway"`` or ``"urban"``.
            ramp_side: Side where the ramp attaches. Only ``"right"`` is
                supported for ``"urban"`` type.
            backward_lane_num: Number of backward lanes; ignored for freeway.
            taper_length: Auxiliary lane taper length in metres.
            parallel_length: Auxiliary lane parallel length in metres.
            step_size: Reference-line sampling interval in metres.

        Raises:
            ValueError: If ``main_road_type`` is ``"urban"`` and ``ramp_side``
                is ``"left"``, or any length parameter is non-positive.
        """
        if main_road_type not in ("freeway", "urban"):
            raise ValueError("main_road_type must be 'freeway' or 'urban'.")
        if main_road_type == "urban" and ramp_side != "right":
            raise ValueError("urban ramp currently supports only ramp_side='right'.")
        if taper_length <= 0.0:
            raise ValueError("taper_length must be positive.")
        if parallel_length <= 0.0:
            raise ValueError("parallel_length must be positive.")
        if step_size <= 0.0:
            raise ValueError("step_size must be positive.")
        self.main_road_type = main_road_type
        self.ramp_side = ramp_side
        self.backward_lane_num = backward_lane_num
        self.taper_length = taper_length
        self.parallel_length = parallel_length
        self.step_size = step_size


class ExitRamp(_RampBase):
    """Exit ramp generator where traffic peels off from the main road.

    See :class:`_RampBase` for all shared attributes.
    """

    def build(
        self,
        main_in: RoadPort,
        main_out: RoadPort,
        ramp_out: RoadPort,
        *,
        lane_width: float | None = None,
        main_speed_limit: float | None = None,
        ramp_speed_limit: float | None = None,
        id_offset: int = 0,
    ) -> RoadModuleResult:
        """Build an exit ramp from the given port sockets.

        Args:
            main_in: Upstream main-road socket.
            main_out: Downstream main-road socket.
            ramp_out: Downstream ramp socket where exiting traffic departs.
            lane_width: Lane width in metres. Defaults to ``main_in.lane_width``.
            main_speed_limit: Main-road speed limit in km/h. Defaults to
                ``main_in.speed_limit``.
            ramp_speed_limit: Ramp speed limit in km/h. Defaults to
                ``ramp_out.speed_limit``.
            id_offset: First element id.

        Returns:
            :class:`RoadModuleResult` with ports ``"main_in"``, ``"main_out"``,
            and ``"ramp"`` (plus ``"backward_in"``/``"backward_out"`` for urban).
        """
        return _build_ramp(
            kind="exit",
            main_road_type=self.main_road_type,
            ramp_side=self.ramp_side,
            main_in=main_in,
            main_out=main_out,
            ramp_port=ramp_out,
            backward_lane_num=self.backward_lane_num,
            lane_width=lane_width,
            main_speed_limit=main_speed_limit,
            ramp_speed_limit=ramp_speed_limit,
            taper_length=self.taper_length,
            parallel_length=self.parallel_length,
            step_size=self.step_size,
            id_offset=id_offset,
        )


class EntranceRamp(_RampBase):
    """Entrance ramp generator where traffic merges onto the main road.

    See :class:`_RampBase` for all shared attributes.
    """

    def build(
        self,
        main_in: RoadPort,
        main_out: RoadPort,
        ramp_in: RoadPort,
        *,
        lane_width: float | None = None,
        main_speed_limit: float | None = None,
        ramp_speed_limit: float | None = None,
        id_offset: int = 0,
    ) -> RoadModuleResult:
        """Build an entrance ramp from the given port sockets.

        Args:
            main_in: Upstream main-road socket.
            main_out: Downstream main-road socket.
            ramp_in: Upstream ramp socket where entering traffic arrives.
            lane_width: Lane width in metres. Defaults to ``main_in.lane_width``.
            main_speed_limit: Main-road speed limit in km/h. Defaults to
                ``main_in.speed_limit``.
            ramp_speed_limit: Ramp speed limit in km/h. Defaults to
                ``ramp_in.speed_limit``.
            id_offset: First element id.

        Returns:
            :class:`RoadModuleResult` with ports ``"main_in"``, ``"main_out"``,
            and ``"ramp"`` (plus ``"backward_in"``/``"backward_out"`` for urban).
        """
        return _build_ramp(
            kind="entrance",
            main_road_type=self.main_road_type,
            ramp_side=self.ramp_side,
            main_in=main_in,
            main_out=main_out,
            ramp_port=ramp_in,
            backward_lane_num=self.backward_lane_num,
            lane_width=lane_width,
            main_speed_limit=main_speed_limit,
            ramp_speed_limit=ramp_speed_limit,
            taper_length=self.taper_length,
            parallel_length=self.parallel_length,
            step_size=self.step_size,
            id_offset=id_offset,
        )
