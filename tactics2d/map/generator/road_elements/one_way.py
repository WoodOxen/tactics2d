# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Socket-driven one-way road generator."""

from __future__ import annotations

from shapely.geometry import LineString

from tactics2d.map.element import Lane, LaneRelationship, RoadLine

from ..geometry.geometry_utils import offset_polyline
from ..geometry.module_geometry import (
    curvature_stats,
    fit_reference_line,
    has_self_intersection,
    polyline_length,
)
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
    if step_size <= 0.0:
        raise ValueError("step_size must be positive.")

    center_pts = fit_reference_line(
        start_port.point, start_port.heading, end_port.point, end_port.heading, step_size
    )

    total_half_width = lane_n * lane_w / 2.0
    lanes: list[Lane] = []
    roadlines: list[RoadLine] = []
    id_counter = id_offset

    for i in range(lane_n):
        left_offset = total_half_width - i * lane_w
        right_offset = left_offset - lane_w

        left_pts = offset_polyline(center_pts, left_offset)
        right_pts = offset_polyline(center_pts, right_offset)

        left_rl = RoadLine(
            id_=str(id_counter),
            geometry=LineString(left_pts),
            **one_way_mark_kwargs(i, lane_n, "left"),
        )
        id_counter += 1

        right_rl = RoadLine(
            id_=str(id_counter),
            geometry=LineString(right_pts),
            **one_way_mark_kwargs(i, lane_n, "right"),
        )
        id_counter += 1

        lane = Lane(
            id_=str(id_counter),
            left_side=LineString(left_pts),
            right_side=LineString(right_pts),
            subtype="road",
            speed_limit=speed,
            speed_limit_unit="km/h",
            line_ids={"left": [left_rl.id_], "right": [right_rl.id_]},
            custom_tags={"module": "one_way", "lane_index": i},
        )
        id_counter += 1

        lanes.append(lane)
        roadlines.extend([left_rl, right_rl])

    for i, lane in enumerate(lanes):
        if i > 0:
            lane.add_related_lane(lanes[i - 1].id_, LaneRelationship.LEFT_NEIGHBOR)
        if i < len(lanes) - 1:
            lane.add_related_lane(lanes[i + 1].id_, LaneRelationship.RIGHT_NEIGHBOR)

    lane_ids = tuple(lane.id_ for lane in lanes)
    ports = {
        "entry": make_port(start_port, kind="entry", name="entry", lane_ids=lane_ids),
        "exit": make_port(end_port, kind="exit", name="exit", lane_ids=lane_ids),
    }

    stats = curvature_stats(center_pts)
    quality = {
        "module": "one_way",
        "length": polyline_length(center_pts),
        "self_intersection": has_self_intersection(center_pts),
        **stats,
    }

    return RoadModuleResult(
        lanes=lanes,
        roadlines=roadlines,
        ports=ports,
        interfaces=ports_to_interfaces(ports),
        quality=quality,
        id_counter=id_counter,
    )
