# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Two-way road segment generator implementation."""

from __future__ import annotations

import numpy as np

from tactics2d.geometry import polyline
from tactics2d.map.element import Lane, LaneRelationship, RoadLine
from tactics2d.map.generator.rules.lane_marking_rules import (
    roadline_render_kwargs,
    two_way_backward,
    two_way_centerline,
    two_way_forward,
)
from tactics2d.map.generator.rules.module_types import RoadModuleResult, RoadPort, build_port

from .element_builder import (
    add_ordered_lane_neighbors,
    build_lane_from_boundaries,
    build_roadline_from_points,
    lane_ids,
)
from .reference_line import fit_reference_line
from .road_segment import RoadSegment


class TwoWay(RoadSegment):
    """Bidirectional road segment generator with separate forward and backward lanes.

    Forward and backward lanes share a common centreline reference but are
    offset symmetrically.  A double solid or dashed yellow centre line
    separates the two directions; outer edges and interior dividers follow
    the active standard (MUTCD or GB).

    Attributes:
        step_size: Reference-line sampling interval in metres.
    """

    def __init__(self, step_size: float = 0.1) -> None:
        """Initialise the generator.

        Args:
            step_size: Reference-line sampling interval in metres.
        """
        if step_size <= 0.0:
            raise ValueError("step_size must be positive.")
        self.step_size = step_size

    def build(
        self,
        start_port: RoadPort,
        end_port: RoadPort,
        *,
        forward_lane_num: int | None = None,
        backward_lane_num: int | None = None,
        lane_width: float | None = None,
        speed_limit: float | None = None,
        id_offset: int = 0,
    ) -> RoadModuleResult:
        """Build a two-way road between two ports.

        Args:
            start_port: Forward-direction start socket.
            end_port: Forward-direction end socket.
            forward_lane_num: Forward lane count. Defaults to ``start_port.lane_num``.
            backward_lane_num: Backward lane count. Defaults to ``start_port.lane_num``.
            lane_width: Lane width in metres. Defaults to ``start_port.lane_width``.
            speed_limit: Speed limit in km/h. Defaults to ``start_port.speed_limit``.
            id_offset: First element id.

        Returns:
            :class:`RoadModuleResult` with ports ``"forward_in"``, ``"forward_out"``,
            ``"backward_in"``, and ``"backward_out"``.

        Raises:
            ValueError: If any lane count is less than 1 or ``lane_width <= 0``.
        """
        forward_n = int(forward_lane_num if forward_lane_num is not None else start_port.lane_num)
        backward_n = int(
            backward_lane_num if backward_lane_num is not None else start_port.lane_num
        )
        lane_w = float(lane_width if lane_width is not None else start_port.lane_width)
        speed = float(speed_limit if speed_limit is not None else start_port.speed_limit)

        if forward_n < 1:
            raise ValueError("forward_lane_num must be >= 1.")
        if backward_n < 1:
            raise ValueError("backward_lane_num must be >= 1 for two_way roads.")
        if lane_w <= 0.0:
            raise ValueError("lane_width must be positive.")

        center_pts = fit_reference_line(
            start_port.point, start_port.heading, end_port.point, end_port.heading, self.step_size
        )

        lanes: list[Lane] = []
        roadlines: list[RoadLine] = []
        id_counter = id_offset

        center_roadline = build_roadline_from_points(
            id_=id_counter,
            points=center_pts,
            marking_kwargs=roadline_render_kwargs(
                two_way_centerline(forward_n, backward_n),
                {"module": "two_way", "role": "centerline", "submodule": "centerline"},
            ),
        )
        id_counter += 1
        roadlines.append(center_roadline)

        forward_boundary_pts: list[np.ndarray] = [center_pts]
        forward_boundary_rls: list[RoadLine] = [center_roadline]

        for i in range(1, forward_n + 1):
            pts = polyline.offset(center_pts, -i * lane_w)
            rl = build_roadline_from_points(
                id_=id_counter,
                points=pts,
                marking_kwargs=roadline_render_kwargs(
                    two_way_forward(i - 1, forward_n, "right"),
                    {"module": "two_way", "direction": "forward", "boundary_index": i},
                ),
            )
            id_counter += 1
            roadlines.append(rl)
            forward_boundary_pts.append(pts)
            forward_boundary_rls.append(rl)

        forward_lanes: list[Lane] = []
        for i in range(forward_n):
            lane = build_lane_from_boundaries(
                id_=id_counter,
                left_points=forward_boundary_pts[i],
                right_points=forward_boundary_pts[i + 1],
                left_roadline_ids=forward_boundary_rls[i].id_,
                right_roadline_ids=forward_boundary_rls[i + 1].id_,
                speed_limit=speed,
                custom_tags={"module": "two_way", "direction": "forward", "lane_index": i},
            )
            id_counter += 1
            forward_lanes.append(lane)
            lanes.append(lane)

        add_ordered_lane_neighbors(forward_lanes)

        backward_boundary_pts: list[np.ndarray] = [center_pts[::-1]]
        backward_boundary_rls: list[RoadLine] = [center_roadline]

        for i in range(1, backward_n + 1):
            pts = polyline.offset(center_pts, i * lane_w)[::-1]
            rl = build_roadline_from_points(
                id_=id_counter,
                points=pts,
                marking_kwargs=roadline_render_kwargs(
                    two_way_backward(i - 1, backward_n, "left"),
                    {"module": "two_way", "direction": "backward", "boundary_index": i},
                ),
            )
            id_counter += 1
            roadlines.append(rl)
            backward_boundary_pts.append(pts)
            backward_boundary_rls.append(rl)

        backward_lanes: list[Lane] = []
        for i in range(backward_n):
            lane = build_lane_from_boundaries(
                id_=id_counter,
                left_points=backward_boundary_pts[i],
                right_points=backward_boundary_pts[i + 1],
                left_roadline_ids=backward_boundary_rls[i].id_,
                right_roadline_ids=backward_boundary_rls[i + 1].id_,
                speed_limit=speed,
                custom_tags={"module": "two_way", "direction": "backward", "lane_index": i},
            )
            id_counter += 1
            backward_lanes.append(lane)
            lanes.append(lane)

        add_ordered_lane_neighbors(
            backward_lanes,
            left_relationship=LaneRelationship.RIGHT_NEIGHBOR,
            right_relationship=LaneRelationship.LEFT_NEIGHBOR,
        )

        forward_ids = lane_ids(forward_lanes)
        backward_ids = lane_ids(backward_lanes)

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
            "forward_in": build_port(
                start_port, kind="forward_in", name="forward_in", lane_ids=forward_ids
            ),
            "forward_out": build_port(
                end_port, kind="forward_out", name="forward_out", lane_ids=forward_ids
            ),
            "backward_in": build_port(
                reverse_start, kind="backward_in", name="backward_in", lane_ids=backward_ids
            ),
            "backward_out": build_port(
                reverse_end, kind="backward_out", name="backward_out", lane_ids=backward_ids
            ),
        }

        return RoadModuleResult(
            lanes=lanes, roadlines=roadlines, ports=ports, id_counter=id_counter
        )
