# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Socket-driven entrance and exit ramp generators."""

from __future__ import annotations

from typing import Literal

import numpy as np
from shapely.geometry import LineString, Polygon

from tactics2d.map.element import Area, Lane, LaneRelationship, RoadLine

from ..geometry.geometry_utils import offset_polyline
from ..geometry.module_geometry import (
    as_point,
    bezier_connection,
    curvature_stats,
    fit_reference_line,
    has_self_intersection,
    nearest_s,
    polyline_length,
    sample_by_s,
)
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


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


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
    if abs(_wrap_angle(float(heading) - chord_heading)) > max_error:
        return chord_heading

    return float(heading)


def _merge_tags(marking_kwargs: dict, tags: dict | None = None) -> dict:
    """Merge marking metadata with module-specific metadata."""
    merged = dict(marking_kwargs)
    custom_tags = dict(merged.get("custom_tags", {}))
    if tags is not None:
        custom_tags.update(tags)
    merged["custom_tags"] = custom_tags
    return merged


def _rl(id_: str, pts: np.ndarray, marking_kwargs: dict, tags=None) -> RoadLine:
    """Create a RoadLine from renderer-compatible marking kwargs."""
    return RoadLine(id_=id_, geometry=LineString(pts), **_merge_tags(marking_kwargs, tags))


def _line_ids(ids: str | list[str]) -> list[str]:
    """Return RoadLine ids as a list."""
    if isinstance(ids, str):
        return [ids]
    return list(ids)


def _lane(
    id_: str,
    left_pts: np.ndarray,
    right_pts: np.ndarray,
    left_rl_ids: str | list[str],
    right_rl_ids: str | list[str],
    speed_limit: float,
    tags: dict,
) -> Lane:
    """Create a lane with one or more RoadLine ids on each side."""
    return Lane(
        id_=id_,
        left_side=LineString(left_pts),
        right_side=LineString(right_pts),
        subtype="road",
        speed_limit=speed_limit,
        speed_limit_unit="km/h",
        line_ids={"left": _line_ids(left_rl_ids), "right": _line_ids(right_rl_ids)},
        custom_tags=tags,
    )


def _link(predecessor: Lane, successor: Lane) -> None:
    """Link two lanes with predecessor-successor relationship."""
    predecessor.add_related_lane(successor.id_, LaneRelationship.SUCCESSOR)
    successor.add_related_lane(predecessor.id_, LaneRelationship.PREDECESSOR)


def _side_neighbor_relationship(side: RampSide) -> tuple[LaneRelationship, LaneRelationship]:
    """Return neighbor relationships between main lane and ramp auxiliary lane."""
    if side == "right":
        return LaneRelationship.RIGHT_NEIGHBOR, LaneRelationship.LEFT_NEIGHBOR
    return LaneRelationship.LEFT_NEIGHBOR, LaneRelationship.RIGHT_NEIGHBOR


def _subline_by_s(pts: np.ndarray, s_start: float, s_end: float, step_size: float) -> np.ndarray:
    """Sample a polyline segment by arc-length interval."""
    length = max(0.0, float(s_end - s_start))
    n_samples = max(2, int(length / max(step_size, 1e-3)) + 1)
    positions, _, _ = sample_by_s(pts, s_start, s_end, n_samples)
    return positions


def _build_segmented_roadline(
    pts: np.ndarray,
    gap_interval: tuple[float, float] | None,
    marking_kwargs: dict,
    id_counter: int,
    step_size: float,
    tags: dict,
    global_s_offset: float = 0.0,
) -> tuple[list[RoadLine], list[str], int]:
    """Build a roadline while optionally leaving one gap interval."""
    total_length = polyline_length(pts)

    if gap_interval is None:
        t = dict(tags)
        t["dash_offset"] = float(global_s_offset)
        roadline = _rl(str(id_counter), pts, marking_kwargs, t)
        return [roadline], [roadline.id_], id_counter + 1

    gap_start, gap_end = gap_interval
    gap_start = float(np.clip(gap_start, 0.0, total_length))
    gap_end = float(np.clip(gap_end, gap_start, total_length))

    roadlines: list[RoadLine] = []
    roadline_ids: list[str] = []

    for seg_start, seg_end in [(0.0, gap_start), (gap_end, total_length)]:
        if seg_end - seg_start <= max(step_size, 1e-3):
            continue

        seg_pts = _subline_by_s(pts, seg_start, seg_end, step_size)
        t = dict(tags)
        t["dash_offset"] = float(global_s_offset + seg_start)
        roadline = _rl(str(id_counter), seg_pts, marking_kwargs, t)
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
    """Build a one-way freeway mainline from a carriageway reference line."""
    if lane_num < 1:
        raise ValueError("lane_num must be greater than or equal to 1.")

    lanes: list[Lane] = []
    roadlines: list[RoadLine] = []
    c = id_counter

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

        generated_roadlines, generated_ids, c = _build_segmented_roadline(
            pts,
            gap,
            marking_kwargs,
            c,
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
        lane = _lane(
            str(c),
            boundary_pts[i],
            boundary_pts[i + 1],
            boundary_line_ids[i],
            boundary_line_ids[i + 1],
            speed_limit,
            {
                "module": "ramp",
                "submodule": "main",
                "main_road_type": "freeway",
                "direction": "forward",
                "lane_index": i,
            },
        )
        c += 1
        lanes.append(lane)

    for i, lane in enumerate(lanes):
        if i > 0:
            lane.add_related_lane(lanes[i - 1].id_, LaneRelationship.LEFT_NEIGHBOR)
        if i < len(lanes) - 1:
            lane.add_related_lane(lanes[i + 1].id_, LaneRelationship.RIGHT_NEIGHBOR)

    if ramp_side == "right":
        attach_boundary = boundary_pts[-1]
        attach_lane = lanes[-1]
    else:
        attach_boundary = boundary_pts[0]
        attach_lane = lanes[0]

    return lanes, roadlines, lanes, [], attach_boundary, attach_lane, c


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
    """Build a two-way urban main road from a centerline."""
    if ramp_side != "right":
        raise ValueError("urban ramp currently supports only ramp_side='right'.")

    lanes: list[Lane] = []
    roadlines: list[RoadLine] = []
    c = id_counter

    center_roadline = _rl(
        str(c),
        center_pts,
        two_way_centerline_kwargs(forward_lane_num, backward_lane_num, no_passing=True),
        {
            "module": "ramp",
            "submodule": "main",
            "main_road_type": "urban",
            "role": "centerline",
            "boundary_index": 0,
        },
    )
    c += 1
    roadlines.append(center_roadline)

    forward_boundary_pts = [
        offset_polyline(center_pts, -i * lane_width) for i in range(forward_lane_num + 1)
    ]
    forward_boundary_line_ids: list[list[str]] = [[center_roadline.id_]]

    for boundary_idx in range(1, forward_lane_num + 1):
        pts = forward_boundary_pts[boundary_idx]
        is_outer_edge = boundary_idx == forward_lane_num
        marking_kwargs = two_way_forward_kwargs(boundary_idx - 1, forward_lane_num, "right")

        generated_roadlines, generated_ids, c = _build_segmented_roadline(
            pts,
            edge_gap if is_outer_edge else None,
            marking_kwargs,
            c,
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
        lane = _lane(
            str(c),
            forward_boundary_pts[i],
            forward_boundary_pts[i + 1],
            forward_boundary_line_ids[i],
            forward_boundary_line_ids[i + 1],
            speed_limit,
            {
                "module": "ramp",
                "submodule": "main",
                "main_road_type": "urban",
                "direction": "forward",
                "lane_index": i,
            },
        )
        c += 1
        forward_lanes.append(lane)
        lanes.append(lane)

    for i, lane in enumerate(forward_lanes):
        if i > 0:
            lane.add_related_lane(forward_lanes[i - 1].id_, LaneRelationship.LEFT_NEIGHBOR)
        if i < len(forward_lanes) - 1:
            lane.add_related_lane(forward_lanes[i + 1].id_, LaneRelationship.RIGHT_NEIGHBOR)

    backward_boundary_pts_raw = [
        offset_polyline(center_pts, i * lane_width) for i in range(backward_lane_num + 1)
    ]
    backward_boundary_line_ids: list[list[str]] = [[center_roadline.id_]]

    for boundary_idx in range(1, backward_lane_num + 1):
        pts = backward_boundary_pts_raw[boundary_idx]
        is_outer_edge = boundary_idx == backward_lane_num
        marking_kwargs = two_way_backward_kwargs(boundary_idx - 1, backward_lane_num, "left")

        roadline = _rl(
            str(c),
            pts,
            marking_kwargs,
            {
                "module": "ramp",
                "submodule": "main",
                "main_road_type": "urban",
                "direction": "backward",
                "boundary_index": boundary_idx,
                "side": "left" if is_outer_edge else "interior",
            },
        )
        c += 1
        roadlines.append(roadline)
        backward_boundary_line_ids.append([roadline.id_])

    backward_lanes: list[Lane] = []
    for i in range(backward_lane_num):
        left_pts = backward_boundary_pts_raw[i][::-1]
        right_pts = backward_boundary_pts_raw[i + 1][::-1]

        lane = _lane(
            str(c),
            left_pts,
            right_pts,
            backward_boundary_line_ids[i],
            backward_boundary_line_ids[i + 1],
            speed_limit,
            {
                "module": "ramp",
                "submodule": "main",
                "main_road_type": "urban",
                "direction": "backward",
                "lane_index": i,
            },
        )
        c += 1
        backward_lanes.append(lane)
        lanes.append(lane)

    for i, lane in enumerate(backward_lanes):
        if i > 0:
            lane.add_related_lane(backward_lanes[i - 1].id_, LaneRelationship.RIGHT_NEIGHBOR)
        if i < len(backward_lanes) - 1:
            lane.add_related_lane(backward_lanes[i + 1].id_, LaneRelationship.LEFT_NEIGHBOR)

    attach_boundary = forward_boundary_pts[-1]
    attach_lane = forward_lanes[-1]

    return lanes, roadlines, forward_lanes, backward_lanes, attach_boundary, attach_lane, c


def _build_single_lane_from_center(
    center_pts: np.ndarray,
    lane_width: float,
    speed_limit: float,
    left_marking_kwargs: dict,
    right_marking_kwargs: dict,
    id_counter: int,
    tags: dict,
) -> tuple[Lane, list[RoadLine], int]:
    """Build a single lane around a centerline.

    left/right are defined in the lane driving direction.
    """
    left_pts = offset_polyline(center_pts, lane_width / 2.0)
    right_pts = offset_polyline(center_pts, -lane_width / 2.0)

    left_roadline = _rl(str(id_counter), left_pts, left_marking_kwargs, {**tags, "side": "left"})
    id_counter += 1

    right_roadline = _rl(
        str(id_counter), right_pts, right_marking_kwargs, {**tags, "side": "right"}
    )
    id_counter += 1

    lane = _lane(
        str(id_counter),
        left_pts,
        right_pts,
        left_roadline.id_,
        right_roadline.id_,
        speed_limit,
        tags,
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
    """Build an entrance or exit ramp module."""
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

    aux_left_roadline = _rl(
        str(id_counter),
        aux_left,
        aux_left_marking,
        {"module": "ramp", "segment": aux_segment, "side": "left"},
    )
    id_counter += 1

    aux_right_roadline = _rl(
        str(id_counter),
        aux_right,
        aux_right_marking,
        {"module": "ramp", "segment": aux_segment, "side": "right"},
    )
    id_counter += 1

    aux_lane = _lane(
        str(id_counter),
        aux_left,
        aux_right,
        aux_left_roadline.id_,
        aux_right_roadline.id_,
        ramp_speed,
        {"module": "ramp"},
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

    main_to_aux_rel, aux_to_main_rel = _side_neighbor_relationship(ramp_side)
    attach_lane.add_related_lane(aux_lane.id_, main_to_aux_rel)
    aux_lane.add_related_lane(attach_lane.id_, aux_to_main_rel)

    if kind == "exit":
        _link(attach_lane, aux_lane)
        _link(aux_lane, connector_lane)
    else:
        _link(connector_lane, aux_lane)
        _link(aux_lane, attach_lane)

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
    """Generate an exit ramp."""
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
    """Generate an entrance ramp."""
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
    """Generate a one-way freeway exit ramp."""
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
    """Generate a one-way freeway entrance ramp."""
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
    """Generate a two-way urban exit ramp."""
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
    """Generate a two-way urban entrance ramp."""
    return entrance_ramp(
        main_in,
        main_out,
        ramp_in,
        main_road_type="urban",
        ramp_side="right",
        backward_lane_num=backward_lane_num,
        **kwargs,
    )
