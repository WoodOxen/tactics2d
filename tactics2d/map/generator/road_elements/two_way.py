# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Socket-driven two-way road generator."""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString

from tactics2d.map.element import Lane, LaneRelationship, RoadLine

from ..geometry.geometry_utils import offset_polyline
from ..geometry.module_geometry import (
    curvature_stats,
    fit_reference_line,
    has_self_intersection,
    polyline_length,
)
from ..rules.lane_marking_rules import (
    two_way_backward_kwargs,
    two_way_centerline_kwargs,
    two_way_forward_kwargs,
)
from ..rules.module_types import RoadModuleResult, RoadPort, make_port, ports_to_interfaces


def two_way(
    start_port: RoadPort,
    end_port: RoadPort,
    *,
    forward_lane_num: int | None = None,
    backward_lane_num: int | None = None,
    lane_width: float | None = None,
    speed_limit: float | None = None,
    step_size: float = 0.1,
    id_offset: int = 0,
) -> RoadModuleResult:
    """Generate a two-way road between two explicit ports.

    Args:
        start_port: Forward-direction start socket.
        end_port: Forward-direction end socket.
        forward_lane_num: Forward lane count. Defaults to start_port.lane_num.
        backward_lane_num: Backward lane count. Defaults to start_port.lane_num.
        lane_width: Lane width in metres. Defaults to start_port.lane_width.
        speed_limit: Speed limit in km/h. Defaults to start_port.speed_limit.
        step_size: Polyline sampling interval.
        id_offset: Starting id.

    Returns:
        RoadModuleResult exposing directional road ports.
    """
    forward_n = int(forward_lane_num if forward_lane_num is not None else start_port.lane_num)
    backward_n = int(backward_lane_num if backward_lane_num is not None else start_port.lane_num)
    lane_w = float(lane_width if lane_width is not None else start_port.lane_width)
    speed = float(speed_limit if speed_limit is not None else start_port.speed_limit)

    if forward_n < 1:
        raise ValueError("forward_lane_num must be >= 1.")
    if backward_n < 1:
        raise ValueError("backward_lane_num must be >= 1 for two_way roads.")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive.")

    center_pts = fit_reference_line(
        start_port.point, start_port.heading, end_port.point, end_port.heading, step_size
    )

    lanes: list[Lane] = []
    roadlines: list[RoadLine] = []
    id_counter = id_offset

    center_rl = RoadLine(
        id_=str(id_counter),
        geometry=LineString(center_pts),
        **two_way_centerline_kwargs(forward_n, backward_n, custom_tags={"submodule": "centerline"}),
    )
    id_counter += 1
    roadlines.append(center_rl)

    forward_lanes: list[Lane] = []

    for i in range(forward_n):
        left_pts = offset_polyline(center_pts, -i * lane_w)
        right_pts = offset_polyline(center_pts, -(i + 1) * lane_w)

        if i == 0:
            left_rl = center_rl
        else:
            left_rl = RoadLine(
                id_=str(id_counter),
                geometry=LineString(left_pts),
                **two_way_forward_kwargs(
                    i, forward_n, "left", custom_tags={"submodule": "forward"}
                ),
            )
            id_counter += 1
            roadlines.append(left_rl)

        right_rl = RoadLine(
            id_=str(id_counter),
            geometry=LineString(right_pts),
            **two_way_forward_kwargs(i, forward_n, "right", custom_tags={"submodule": "forward"}),
        )
        id_counter += 1
        roadlines.append(right_rl)

        lane = Lane(
            id_=str(id_counter),
            left_side=LineString(left_pts),
            right_side=LineString(right_pts),
            subtype="road",
            speed_limit=speed,
            speed_limit_unit="km/h",
            line_ids={"left": [left_rl.id_], "right": [right_rl.id_]},
            custom_tags={"module": "two_way", "direction": "forward", "lane_index": i},
        )
        id_counter += 1

        forward_lanes.append(lane)
        lanes.append(lane)

    for i, lane in enumerate(forward_lanes):
        if i > 0:
            lane.add_related_lane(forward_lanes[i - 1].id_, LaneRelationship.LEFT_NEIGHBOR)
        if i < len(forward_lanes) - 1:
            lane.add_related_lane(forward_lanes[i + 1].id_, LaneRelationship.RIGHT_NEIGHBOR)

    backward_lanes: list[Lane] = []

    for i in range(backward_n):
        left_pts_raw = offset_polyline(center_pts, (i + 1) * lane_w)[::-1]
        right_pts_raw = offset_polyline(center_pts, i * lane_w)[::-1]

        left_rl = RoadLine(
            id_=str(id_counter),
            geometry=LineString(left_pts_raw),
            **two_way_backward_kwargs(i, backward_n, "left", custom_tags={"submodule": "backward"}),
        )
        id_counter += 1
        roadlines.append(left_rl)

        if i == 0:
            right_rl = center_rl
        else:
            right_rl = RoadLine(
                id_=str(id_counter),
                geometry=LineString(right_pts_raw),
                **two_way_backward_kwargs(
                    i, backward_n, "right", custom_tags={"submodule": "backward"}
                ),
            )
            id_counter += 1
            roadlines.append(right_rl)

        lane = Lane(
            id_=str(id_counter),
            left_side=LineString(right_pts_raw),
            right_side=LineString(left_pts_raw),
            subtype="road",
            speed_limit=speed,
            speed_limit_unit="km/h",
            line_ids={"left": [right_rl.id_], "right": [left_rl.id_]},
            custom_tags={"module": "two_way", "direction": "backward", "lane_index": i},
        )
        id_counter += 1

        backward_lanes.append(lane)
        lanes.append(lane)

    for i, lane in enumerate(backward_lanes):
        if i > 0:
            lane.add_related_lane(backward_lanes[i - 1].id_, LaneRelationship.RIGHT_NEIGHBOR)
        if i < len(backward_lanes) - 1:
            lane.add_related_lane(backward_lanes[i + 1].id_, LaneRelationship.LEFT_NEIGHBOR)

    forward_ids = tuple(lane.id_ for lane in forward_lanes)
    backward_ids = tuple(lane.id_ for lane in backward_lanes)

    reverse_start = RoadPort(
        point=np.asarray(end_port.point, dtype=float),
        heading=float(end_port.heading + np.pi),
        lane_num=backward_n,
        lane_width=lane_w,
        speed_limit=speed,
    )
    reverse_end = RoadPort(
        point=np.asarray(start_port.point, dtype=float),
        heading=float(start_port.heading + np.pi),
        lane_num=backward_n,
        lane_width=lane_w,
        speed_limit=speed,
    )

    ports = {
        "forward_in": make_port(
            start_port, kind="forward_in", name="forward_in", lane_ids=forward_ids
        ),
        "forward_out": make_port(
            end_port, kind="forward_out", name="forward_out", lane_ids=forward_ids
        ),
        "backward_in": make_port(
            reverse_start, kind="backward_in", name="backward_in", lane_ids=backward_ids
        ),
        "backward_out": make_port(
            reverse_end, kind="backward_out", name="backward_out", lane_ids=backward_ids
        ),
    }

    stats = curvature_stats(center_pts)
    quality = {
        "module": "two_way",
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
