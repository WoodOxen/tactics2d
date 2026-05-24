# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Socket-driven one-way road generator."""

from __future__ import annotations

from tactics2d.geometry import offset_polyline
from tactics2d.map.element import Lane, RoadLine
from tactics2d.map.generator.helpers.element_builder import (
    add_ordered_lane_neighbors,
    build_lane_from_boundaries,
    build_module_quality,
    build_roadline_from_points,
    lane_ids,
)
from tactics2d.map.generator.helpers.reference_line import fit_reference_line

from ..rules.lane_marking_rules import one_way_mark_kwargs
from ..rules.module_types import RoadModuleResult, RoadPort, make_port, ports_to_interfaces


def one_way(
    start_port: RoadPort,
    end_port: RoadPort,
    *,
    lane_num: int | None = None,
    lane_width: float | None = None,
    speed_limit: float | None = None,
    step_size: float = 0.1,
    id_offset: int = 0,
) -> RoadModuleResult:
    """Generate a one-way road between two explicit ports.

    Args:
        start_port: Entry socket.
        end_port: Exit socket.
        lane_num: Lane count. Defaults to start_port.lane_num.
        lane_width: Lane width in metres. Defaults to start_port.lane_width.
        speed_limit: Speed limit in km/h. Defaults to start_port.speed_limit.
        step_size: Polyline sampling interval.
        id_offset: Starting id.

    Returns:
        RoadModuleResult with ports ``entry`` and ``exit``.
    """
    lane_n = int(lane_num if lane_num is not None else start_port.lane_num)
    lane_w = float(lane_width if lane_width is not None else start_port.lane_width)
    speed = float(speed_limit if speed_limit is not None else start_port.speed_limit)

    if lane_n < 1:
        raise ValueError("lane_num must be >= 1.")
    if lane_w <= 0.0:
        raise ValueError("lane_width must be positive.")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive.")

    center_pts = fit_reference_line(
        start_port.point, start_port.heading, end_port.point, end_port.heading, step_size
    )

    total_half_width = lane_n * lane_w / 2.0

    lanes: list[Lane] = []
    roadlines: list[RoadLine] = []
    id_counter = id_offset

    for lane_idx in range(lane_n):
        left_offset = total_half_width - lane_idx * lane_w
        right_offset = left_offset - lane_w

        left_pts = offset_polyline(center_pts, left_offset)
        right_pts = offset_polyline(center_pts, right_offset)

        left_roadline = build_roadline_from_points(
            id_=id_counter,
            points=left_pts,
            marking_kwargs=one_way_mark_kwargs(lane_idx, lane_n, "left"),
        )
        id_counter += 1

        right_roadline = build_roadline_from_points(
            id_=id_counter,
            points=right_pts,
            marking_kwargs=one_way_mark_kwargs(lane_idx, lane_n, "right"),
        )
        id_counter += 1

        lane = build_lane_from_boundaries(
            id_=id_counter,
            left_points=left_pts,
            right_points=right_pts,
            left_roadline_ids=left_roadline.id_,
            right_roadline_ids=right_roadline.id_,
            speed_limit=speed,
            custom_tags={"module": "one_way", "lane_index": lane_idx},
        )
        id_counter += 1

        lanes.append(lane)
        roadlines.extend([left_roadline, right_roadline])

    add_ordered_lane_neighbors(lanes)

    ids = lane_ids(lanes)
    ports = {
        "entry": make_port(start_port, kind="entry", name="entry", lane_ids=ids),
        "exit": make_port(end_port, kind="exit", name="exit", lane_ids=ids),
    }

    quality = build_module_quality("one_way", center_pts)

    return RoadModuleResult(
        lanes=lanes,
        roadlines=roadlines,
        ports=ports,
        interfaces=ports_to_interfaces(ports),
        quality=quality,
        id_counter=id_counter,
    )
