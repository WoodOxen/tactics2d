# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Socket-driven entrance and exit ramp generators."""

from __future__ import annotations

from typing import Literal

import numpy as np
from shapely.geometry import Polygon

from tactics2d.geometry import (
    as_point,
    curvature_stats,
    has_self_intersection,
    nearest_s,
    normalize_angle,
    offset_polyline,
    polyline_length,
    sample_by_s,
)
from tactics2d.map.element import Area, Lane, LaneRelationship, RoadLine
from tactics2d.map.generator.helpers.element_builder import (
    add_ordered_lane_neighbors,
    build_lane_from_boundaries,
    build_roadline_from_points,
    link_lanes,
)
from tactics2d.map.generator.helpers.reference_line import bezier_connection, fit_reference_line

from ..rules.lane_marking_rules import (
    one_way_mark_kwargs,
    ramp_mark_kwargs,
    two_way_backward_kwargs,
    two_way_centerline_kwargs,
    two_way_forward_kwargs,
)
from ..rules.module_types import RoadModuleResult, RoadPort, make_port, ports_to_interfaces

RampKind = Literal["exit", "entrance"]
MainRoadType = Literal["freeway", "urban"]
RampSide = Literal["right", "left"]


def _repair_heading_towards_chord(
    start_point: np.ndarray, end_point: np.ndarray, heading: float, max_error: float = np.pi / 3.0
) -> float:
    """Repair an endpoint heading when it is inconsistent with the chord."""
    start_point = as_point(start_point)
    end_point = as_point(end_point)
    chord = end_point - start_point

    if np.linalg.norm(chord) < 1e-6:
        return float(heading)

    chord_heading = float(np.arctan2(chord[1], chord[0]))
    if abs(normalize_angle(float(heading) - chord_heading)) > max_error:
        return chord_heading

    return float(heading)


def _build_segmented_roadline(
    pts: np.ndarray,
    gap_interval: tuple[float, float] | None,
    marking_kwargs: dict,
    id_counter: int,
    step_size: float,
    tags: dict,
    global_s_offset: float = 0.0,
) -> tuple[list[RoadLine], list[str], int]:
    """Build a RoadLine, optionally split around a single gap interval.

    Args:
        pts: Polyline points for the boundary.
        gap_interval: ``(start_s, end_s)`` arc-length range to leave empty, or
            ``None`` for a continuous line.
        marking_kwargs: Marking style dict passed to ``build_roadline_from_points``.
        id_counter: Current id counter; incremented for each created RoadLine.
        step_size: Sampling interval used when slicing segments.
        tags: Extra custom_tags dict merged into each created RoadLine.
        global_s_offset: Arc-length origin offset added to ``dash_offset`` tags.

    Returns:
        Tuple of ``(roadlines, roadline_ids, updated_id_counter)``.
    """
    total_length = polyline_length(pts)

    if gap_interval is None:
        t = dict(tags)
        t["dash_offset"] = float(global_s_offset)
        roadline = build_roadline_from_points(
            id_=id_counter, points=pts, marking_kwargs=marking_kwargs, custom_tags=t
        )
        return [roadline], [roadline.id_], id_counter + 1

    gap_start, gap_end = gap_interval
    gap_start = float(np.clip(gap_start, 0.0, total_length))
    gap_end = float(np.clip(gap_end, gap_start, total_length))

    roadlines: list[RoadLine] = []
    roadline_ids: list[str] = []

    for seg_start, seg_end in [(0.0, gap_start), (gap_end, total_length)]:
        if seg_end - seg_start <= max(step_size, 1e-3):
            continue

        _seg_len = max(0.0, float(seg_end - seg_start))
        _n = max(2, int(_seg_len / max(step_size, 1e-3)) + 1)
        seg_pts, _, _ = sample_by_s(pts, seg_start, seg_end, _n)
        t = dict(tags)
        t["dash_offset"] = float(global_s_offset + seg_start)
        roadline = build_roadline_from_points(
            id_=id_counter, points=seg_pts, marking_kwargs=marking_kwargs, custom_tags=t
        )
        id_counter += 1

        roadlines.append(roadline)
        roadline_ids.append(roadline.id_)

    return roadlines, roadline_ids, id_counter


def _build_freeway_main_road(
    center_pts: np.ndarray,
    lane_num: int,
    lane_width: float,
    speed_limit: float,
    id_counter: int,
    step_size: float,
    ramp_side: RampSide,
    edge_gap: tuple[float, float] | None = None,
) -> tuple[list[Lane], list[RoadLine], list[Lane], list[Lane], np.ndarray, Lane, int]:
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
        Tuple ``(lanes, roadlines, forward_lanes, backward_lanes,
        attach_boundary, attach_lane, id_counter)`` where ``backward_lanes``
        is always empty for freeway.
    """
    if lane_num < 1:
        raise ValueError("lane_num must be greater than or equal to 1.")

    lanes: list[Lane] = []
    roadlines: list[RoadLine] = []

    total_half_width = lane_num * lane_width / 2.0
    boundary_offsets = [total_half_width - i * lane_width for i in range(lane_num + 1)]
    boundary_pts = [offset_polyline(center_pts, offset) for offset in boundary_offsets]
    boundary_line_ids: list[list[str]] = []

    for boundary_idx, pts in enumerate(boundary_pts):
        is_left_edge = boundary_idx == 0
        is_right_edge = boundary_idx == lane_num

        if is_left_edge:
            marking_kwargs = one_way_mark_kwargs(0, lane_num, "left")
            side = "left"
            gap = edge_gap if ramp_side == "left" else None
        elif is_right_edge:
            marking_kwargs = one_way_mark_kwargs(lane_num - 1, lane_num, "right")
            side = "right"
            gap = edge_gap if ramp_side == "right" else None
        else:
            marking_kwargs = one_way_mark_kwargs(boundary_idx - 1, lane_num, "right")
            side = "interior"
            gap = None

        generated_roadlines, generated_ids, id_counter = _build_segmented_roadline(
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

    forward_lanes = lanes
    backward_lanes: list[Lane] = []
    return lanes, roadlines, forward_lanes, backward_lanes, attach_boundary, attach_lane, id_counter


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
) -> tuple[list[Lane], list[RoadLine], list[Lane], list[Lane], np.ndarray, Lane, int]:
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
        Tuple ``(lanes, roadlines, forward_lanes, backward_lanes,
        attach_boundary, attach_lane, id_counter)``.
    """
    if ramp_side != "right":
        raise ValueError("urban ramp currently supports only ramp_side='right'.")

    lanes: list[Lane] = []
    roadlines: list[RoadLine] = []

    center_roadline = build_roadline_from_points(
        id_=id_counter,
        points=center_pts,
        marking_kwargs=two_way_centerline_kwargs(
            forward_lane_num, backward_lane_num, no_passing=True
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
        marking_kwargs = two_way_forward_kwargs(boundary_idx - 1, forward_lane_num, "right")

        generated_roadlines, generated_ids, id_counter = _build_segmented_roadline(
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
        marking_kwargs = two_way_backward_kwargs(boundary_idx - 1, backward_lane_num, "left")

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

    attach_boundary = forward_boundary_pts[-1]
    attach_lane = forward_lanes[-1]

    return lanes, roadlines, forward_lanes, backward_lanes, attach_boundary, attach_lane, id_counter


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
        ``"ramp"`` (plus ``"backward_in"``/``"backward_out"`` for urban),
        and a ``quality`` dict containing ``module``, ``kind``,
        ``main_road_type``, ``ramp_side``, geometry statistics for main and
        connector roads, ``accepted_reasons``, and ``accepted``.
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
    main_length = polyline_length(center_pts)

    if main_road_type == "freeway":
        total_half_width = lane_num * lane_w / 2.0
        boundary_offset = -total_half_width if ramp_side == "right" else total_half_width
    else:
        boundary_offset = -lane_num * lane_w

    boundary_preview = offset_polyline(center_pts, boundary_offset)
    boundary_length = polyline_length(boundary_preview)
    ramp_s = nearest_s(boundary_preview, as_point(ramp_port.point))

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

    if main_road_type == "freeway":
        (
            lanes,
            roadlines,
            forward_lanes,
            backward_lanes,
            attach_boundary,
            attach_lane,
            id_counter,
        ) = _build_freeway_main_road(
            center_pts, lane_num, lane_w, main_speed, id_offset, step_size, ramp_side, edge_gap
        )
    else:
        (
            lanes,
            roadlines,
            forward_lanes,
            backward_lanes,
            attach_boundary,
            attach_lane,
            id_counter,
        ) = _build_urban_main_road(
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

    n_taper = max(8, int(taper_length / step_size) + 1)
    taper_pos, taper_hdgs, _ = sample_by_s(attach_boundary, taper_s_start, taper_s_end, n_taper)

    n_parallel = max(4, int(parallel_length / step_size) + 1)
    parallel_pos, parallel_headings, parallel_right_normals = sample_by_s(
        attach_boundary, parallel_s_start, parallel_s_end, n_parallel
    )

    parallel_out_normals = (
        parallel_right_normals if ramp_side == "right" else -parallel_right_normals
    )
    parallel_inner = parallel_pos
    parallel_outer = parallel_pos + lane_w * parallel_out_normals

    if kind == "exit":
        taper_outer = bezier_connection(
            taper_pos[0],
            float(taper_hdgs[0]),
            parallel_outer[0],
            float(parallel_headings[0]),
            step_size,
        )
        aux_inner = np.vstack([taper_pos, parallel_inner[1:]])
        aux_outer = np.vstack([taper_outer, parallel_outer[1:]])

        connector_start = parallel_pos[-1] + 0.5 * lane_w * parallel_out_normals[-1]
        connector_start_heading = float(parallel_headings[-1])
        connector_end = as_point(ramp_port.point)
        connector_end_heading = _repair_heading_towards_chord(
            connector_start, connector_end, float(ramp_port.heading)
        )
        aux_segment = "exit_auxiliary_taper_parallel"
        connector_segment = "exit_connector"
        ramp_port_kind = "ramp_out"
    else:
        taper_outer = bezier_connection(
            parallel_outer[-1],
            float(parallel_headings[-1]),
            taper_pos[-1],
            float(taper_hdgs[-1]),
            step_size,
        )
        aux_inner = np.vstack([parallel_inner, taper_pos[1:]])
        aux_outer = np.vstack([parallel_outer, taper_outer[1:]])

        connector_start = as_point(ramp_port.point)
        connector_end = parallel_pos[0] + 0.5 * lane_w * parallel_out_normals[0]
        connector_start_heading = _repair_heading_towards_chord(
            connector_start, connector_end, float(ramp_port.heading)
        )
        connector_end_heading = float(parallel_headings[0])
        aux_segment = "entrance_parallel_taper"
        connector_segment = "entrance_connector"
        ramp_port_kind = "ramp_in"

    connector_center = bezier_connection(
        connector_start, connector_start_heading, connector_end, connector_end_heading, step_size
    )

    if ramp_side == "right":
        aux_left, aux_right = aux_inner, aux_outer
        aux_left_marking = ramp_mark_kwargs("aux_left")
        aux_right_marking = ramp_mark_kwargs("aux_right")
    else:
        aux_left, aux_right = aux_outer, aux_inner
        aux_left_marking = ramp_mark_kwargs("aux_right")
        aux_right_marking = ramp_mark_kwargs("aux_left")

    aux_left_roadline = build_roadline_from_points(
        id_=id_counter,
        points=aux_left,
        marking_kwargs=aux_left_marking,
        custom_tags={"module": "ramp", "segment": aux_segment, "side": "left"},
    )
    id_counter += 1

    aux_right_roadline = build_roadline_from_points(
        id_=id_counter,
        points=aux_right,
        marking_kwargs=aux_right_marking,
        custom_tags={"module": "ramp", "segment": aux_segment, "side": "right"},
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
        ramp_mark_kwargs("left_edge"),
        ramp_mark_kwargs("right_edge"),
        id_counter,
        tags={
            "module": "ramp",
            "kind": kind,
            "main_road_type": main_road_type,
            "ramp_side": ramp_side,
            "segment": connector_segment,
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
        "main_in": make_port(main_in, kind="main_in", name="main_in", lane_ids=forward_ids),
        "main_out": make_port(main_out, kind="main_out", name="main_out", lane_ids=forward_ids),
        "ramp": make_port(ramp_port, kind=ramp_port_kind, name=ramp_port_kind, lane_ids=ramp_ids),
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
        ports["backward_in"] = make_port(
            reverse_main_in, kind="backward_in", name="backward_in", lane_ids=backward_ids
        )
        ports["backward_out"] = make_port(
            reverse_main_out, kind="backward_out", name="backward_out", lane_ids=backward_ids
        )

    main_stats = curvature_stats(center_pts)
    connector_stats = curvature_stats(connector_center)
    connector_max_curvature = connector_stats["max_abs_curvature"]
    connector_min_radius = (
        float("inf") if connector_max_curvature <= 1e-9 else 1.0 / connector_max_curvature
    )

    main_self_intersection = has_self_intersection(center_pts)
    connector_self_intersection = has_self_intersection(connector_center)

    accepted_reasons = []
    if main_self_intersection:
        accepted_reasons.append("main_self_intersection")
    if connector_self_intersection:
        accepted_reasons.append("connector_self_intersection")

    quality = {
        "module": "ramp",
        "kind": kind,
        "main_road_type": main_road_type,
        "ramp_side": ramp_side,
        "main_length": main_length,
        "boundary_length": boundary_length,
        "ramp_projection_s": ramp_s,
        "taper_s_start": taper_s_start,
        "taper_s_end": taper_s_end,
        "parallel_s_start": parallel_s_start,
        "parallel_s_end": parallel_s_end,
        "connector_length": polyline_length(connector_center),
        "main_self_intersection": main_self_intersection,
        "connector_self_intersection": connector_self_intersection,
        "main_max_abs_curvature": main_stats["max_abs_curvature"],
        "main_max_abs_curvature_rate": main_stats["max_abs_curvature_rate"],
        "connector_max_abs_curvature": connector_max_curvature,
        "connector_min_radius": connector_min_radius,
        "connector_max_abs_curvature_rate": connector_stats["max_abs_curvature_rate"],
        "attach_lane_id": attach_lane.id_,
        "aux_lane_id": aux_lane.id_,
        "connector_lane_id": connector_lane.id_,
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
        junctions=areas,
    )


def exit_ramp(
    main_in: RoadPort,
    main_out: RoadPort,
    ramp_out: RoadPort,
    *,
    main_road_type: MainRoadType = "freeway",
    ramp_side: RampSide = "right",
    backward_lane_num: int = 1,
    lane_width: float | None = None,
    main_speed_limit: float | None = None,
    ramp_speed_limit: float | None = None,
    taper_length: float = 50.0,
    parallel_length: float = 60.0,
    step_size: float = 0.5,
    id_offset: int = 0,
) -> RoadModuleResult:
    """Generate an exit ramp module where traffic peels off from the main road.

    Args:
        main_in: Upstream main-road socket.
        main_out: Downstream main-road socket (after the exit opening).
        ramp_out: Downstream ramp socket where exiting traffic departs.
        main_road_type: ``"freeway"`` for a one-way main road; ``"urban"`` for a
            two-way main road with a separate backward direction.
        ramp_side: Side of the main road where the ramp exits. Currently only
            ``"right"`` is supported for ``"urban"`` type.
        backward_lane_num: Number of backward lanes; ignored for ``"freeway"``.
        lane_width: Lane width in metres. Defaults to ``main_in.lane_width``.
        main_speed_limit: Speed limit on the main road in km/h. Defaults to
            ``main_in.speed_limit``.
        ramp_speed_limit: Speed limit on the ramp in km/h. Defaults to
            ``ramp_out.speed_limit``.
        taper_length: Longitudinal length of the auxiliary lane taper in metres.
        parallel_length: Longitudinal length of the full-width auxiliary lane
            running parallel to the main road before the taper.
        step_size: Reference-line sampling interval in metres.
        id_offset: First id used by generated map elements.

    Returns:
        A ``RoadModuleResult`` with named ports ``"main_in"``, ``"main_out"``,
        and ``"ramp_out"``, plus geometry quality statistics and ``accepted``.

    Raises:
        ValueError: If ``main_road_type`` or ``ramp_side`` is unsupported, or
            any length/width/step parameter is non-positive.
    """
    return _build_ramp(
        kind="exit",
        main_road_type=main_road_type,
        ramp_side=ramp_side,
        main_in=main_in,
        main_out=main_out,
        ramp_port=ramp_out,
        backward_lane_num=backward_lane_num,
        lane_width=lane_width,
        main_speed_limit=main_speed_limit,
        ramp_speed_limit=ramp_speed_limit,
        taper_length=taper_length,
        parallel_length=parallel_length,
        step_size=step_size,
        id_offset=id_offset,
    )


def entrance_ramp(
    main_in: RoadPort,
    main_out: RoadPort,
    ramp_in: RoadPort,
    *,
    main_road_type: MainRoadType = "freeway",
    ramp_side: RampSide = "right",
    backward_lane_num: int = 1,
    lane_width: float | None = None,
    main_speed_limit: float | None = None,
    ramp_speed_limit: float | None = None,
    taper_length: float = 50.0,
    parallel_length: float = 60.0,
    step_size: float = 0.5,
    id_offset: int = 0,
) -> RoadModuleResult:
    """Generate an entrance ramp module where traffic merges onto the main road.

    Args:
        main_in: Upstream main-road socket.
        main_out: Downstream main-road socket (after the merge opening).
        ramp_in: Upstream ramp socket where entering traffic arrives.
        main_road_type: ``"freeway"`` for a one-way main road; ``"urban"`` for a
            two-way main road with a separate backward direction.
        ramp_side: Side of the main road where the ramp merges. Currently only
            ``"right"`` is supported for ``"urban"`` type.
        backward_lane_num: Number of backward lanes; ignored for ``"freeway"``.
        lane_width: Lane width in metres. Defaults to ``main_in.lane_width``.
        main_speed_limit: Speed limit on the main road in km/h. Defaults to
            ``main_in.speed_limit``.
        ramp_speed_limit: Speed limit on the ramp in km/h. Defaults to
            ``ramp_in.speed_limit``.
        taper_length: Longitudinal length of the auxiliary lane taper in metres.
        parallel_length: Longitudinal length of the full-width auxiliary lane
            running parallel to the main road before the taper.
        step_size: Reference-line sampling interval in metres.
        id_offset: First id used by generated map elements.

    Returns:
        A ``RoadModuleResult`` with named ports ``"main_in"``, ``"main_out"``,
        and ``"ramp_in"``, plus geometry quality statistics and ``accepted``.

    Raises:
        ValueError: If ``main_road_type`` or ``ramp_side`` is unsupported, or
            any length/width/step parameter is non-positive.
    """
    return _build_ramp(
        kind="entrance",
        main_road_type=main_road_type,
        ramp_side=ramp_side,
        main_in=main_in,
        main_out=main_out,
        ramp_port=ramp_in,
        backward_lane_num=backward_lane_num,
        lane_width=lane_width,
        main_speed_limit=main_speed_limit,
        ramp_speed_limit=ramp_speed_limit,
        taper_length=taper_length,
        parallel_length=parallel_length,
        step_size=step_size,
        id_offset=id_offset,
    )


def freeway_exit_ramp(
    main_in: RoadPort,
    main_out: RoadPort,
    ramp_out: RoadPort,
    *,
    ramp_side: RampSide = "right",
    **kwargs,
) -> RoadModuleResult:
    """Generate a one-way freeway exit ramp.

    Convenience alias for :func:`exit_ramp` with ``main_road_type="freeway"``.

    Args:
        main_in: Upstream main-road socket.
        main_out: Downstream main-road socket.
        ramp_out: Downstream ramp socket where exiting traffic departs.
        ramp_side: Side of the main road where the ramp exits
            (``"right"`` or ``"left"``).
        **kwargs: Additional keyword arguments forwarded to :func:`exit_ramp`:
            ``lane_width``, ``main_speed_limit``, ``ramp_speed_limit``,
            ``taper_length``, ``parallel_length``, ``step_size``,
            ``id_offset``.

    Returns:
        See :func:`exit_ramp`.
    """
    return exit_ramp(
        main_in, main_out, ramp_out, main_road_type="freeway", ramp_side=ramp_side, **kwargs
    )


def freeway_entrance_ramp(
    main_in: RoadPort,
    main_out: RoadPort,
    ramp_in: RoadPort,
    *,
    ramp_side: RampSide = "right",
    **kwargs,
) -> RoadModuleResult:
    """Generate a one-way freeway entrance ramp.

    Convenience alias for :func:`entrance_ramp` with ``main_road_type="freeway"``.

    Args:
        main_in: Upstream main-road socket.
        main_out: Downstream main-road socket.
        ramp_in: Upstream ramp socket where entering traffic arrives.
        ramp_side: Side of the main road where the ramp merges
            (``"right"`` or ``"left"``).
        **kwargs: Additional keyword arguments forwarded to
            :func:`entrance_ramp`: ``lane_width``, ``main_speed_limit``,
            ``ramp_speed_limit``, ``taper_length``, ``parallel_length``,
            ``step_size``, ``id_offset``.

    Returns:
        See :func:`entrance_ramp`.
    """
    return entrance_ramp(
        main_in, main_out, ramp_in, main_road_type="freeway", ramp_side=ramp_side, **kwargs
    )


def urban_exit_ramp(
    main_in: RoadPort,
    main_out: RoadPort,
    ramp_out: RoadPort,
    *,
    backward_lane_num: int = 1,
    **kwargs,
) -> RoadModuleResult:
    """Generate a two-way urban exit ramp (``ramp_side`` fixed to ``"right"``).

    Convenience alias for :func:`exit_ramp` with ``main_road_type="urban"``
    and ``ramp_side="right"``.

    Args:
        main_in: Upstream main-road socket.
        main_out: Downstream main-road socket.
        ramp_out: Downstream ramp socket where exiting traffic departs.
        backward_lane_num: Number of backward lanes on the main road.
        **kwargs: Additional keyword arguments forwarded to :func:`exit_ramp`:
            ``lane_width``, ``main_speed_limit``, ``ramp_speed_limit``,
            ``taper_length``, ``parallel_length``, ``step_size``,
            ``id_offset``.

    Returns:
        See :func:`exit_ramp`.
    """
    return exit_ramp(
        main_in,
        main_out,
        ramp_out,
        main_road_type="urban",
        ramp_side="right",
        backward_lane_num=backward_lane_num,
        **kwargs,
    )


def urban_entrance_ramp(
    main_in: RoadPort,
    main_out: RoadPort,
    ramp_in: RoadPort,
    *,
    backward_lane_num: int = 1,
    **kwargs,
) -> RoadModuleResult:
    """Generate a two-way urban entrance ramp (``ramp_side`` fixed to ``"right"``).

    Convenience alias for :func:`entrance_ramp` with ``main_road_type="urban"``
    and ``ramp_side="right"``.

    Args:
        main_in: Upstream main-road socket.
        main_out: Downstream main-road socket.
        ramp_in: Upstream ramp socket where entering traffic arrives.
        backward_lane_num: Number of backward lanes on the main road.
        **kwargs: Additional keyword arguments forwarded to
            :func:`entrance_ramp`: ``lane_width``, ``main_speed_limit``,
            ``ramp_speed_limit``, ``taper_length``, ``parallel_length``,
            ``step_size``, ``id_offset``.

    Returns:
        See :func:`entrance_ramp`.
    """
    return entrance_ramp(
        main_in,
        main_out,
        ramp_in,
        main_road_type="urban",
        ramp_side="right",
        backward_lane_num=backward_lane_num,
        **kwargs,
    )
