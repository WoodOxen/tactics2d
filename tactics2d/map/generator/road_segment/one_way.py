# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""One-way road segment generator implementation."""

from __future__ import annotations

from tactics2d.geometry import polyline
from tactics2d.map.element import Lane, RoadLine
from tactics2d.map.generator.rules.lane_marking_rules import (
    one_way_boundary_token,
    roadline_render_kwargs,
)
from tactics2d.map.generator.rules.module_types import RoadModuleResult, RoadPort, build_port

from .element_builder import (
    add_ordered_lane_neighbors,
    boundary_offset,
    build_lane_from_boundaries,
    build_roadline_from_points,
    lane_ids,
)
from .reference_line import fit_reference_line
from .road_segment import RoadSegment


class OneWay(RoadSegment):
    """Single-direction road segment generator that builds N parallel lanes.

    All lanes share the same travel direction and are generated from a Hermite
    cubic reference line between the entry and exit sockets.  Lane markings
    follow the active standard (MUTCD or GB) configured via
    :func:`~tactics2d.map.generator.rules.lane_marking_rules.set_standard`.

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
        lane_num: int | None = None,
        lane_width: float | None = None,
        speed_limit: float | None = None,
        id_offset: int = 0,
    ) -> RoadModuleResult:
        """Build a one-way road between two ports.

        Args:
            start_port: Entry socket.
            end_port: Exit socket.
            lane_num: Lane count. Defaults to ``start_port.lane_num``.
            lane_width: Lane width in metres. Defaults to ``start_port.lane_width``.
            speed_limit: Speed limit in km/h. Defaults to ``start_port.speed_limit``.
            id_offset: First element id.

        Returns:
            :class:`RoadModuleResult` with ports ``"entry"`` and ``"exit"``.

        Raises:
            ValueError: If ``lane_num < 1`` or ``lane_width <= 0``.
        """
        lane_n = int(lane_num if lane_num is not None else start_port.lane_num)
        lane_w = float(lane_width if lane_width is not None else start_port.lane_width)
        speed = float(speed_limit if speed_limit is not None else start_port.speed_limit)

        if lane_n < 1:
            raise ValueError("lane_num must be >= 1.")
        if lane_w <= 0.0:
            raise ValueError("lane_width must be positive.")

        center_pts = fit_reference_line(
            start_port.point, start_port.heading, end_port.point, end_port.heading, self.step_size
        )

        boundary_num = lane_n + 1
        boundary_pts = [
            polyline.offset(center_pts, boundary_offset(i, lane_n, lane_w))
            for i in range(boundary_num)
        ]

        roadlines: list[RoadLine] = []
        boundary_roadlines: list[RoadLine] = []
        id_counter = id_offset

        for i, pts in enumerate(boundary_pts):
            rl = build_roadline_from_points(
                id_=id_counter,
                points=pts,
                marking_kwargs=roadline_render_kwargs(
                    one_way_boundary_token(i, boundary_num),
                    {"module": "one_way", "boundary_index": i},
                ),
            )
            id_counter += 1
            roadlines.append(rl)
            boundary_roadlines.append(rl)

        lanes: list[Lane] = []
        for i in range(lane_n):
            lane = build_lane_from_boundaries(
                id_=id_counter,
                left_points=boundary_pts[i],
                right_points=boundary_pts[i + 1],
                left_roadline_ids=boundary_roadlines[i].id_,
                right_roadline_ids=boundary_roadlines[i + 1].id_,
                speed_limit=speed,
                custom_tags={"module": "one_way", "lane_index": i},
            )
            id_counter += 1
            lanes.append(lane)

        add_ordered_lane_neighbors(lanes)

        ids = lane_ids(lanes)
        ports = {
            "entry": build_port(start_port, kind="entry", name="entry", lane_ids=ids),
            "exit": build_port(end_port, kind="exit", name="exit", lane_ids=ids),
        }

        return RoadModuleResult(
            lanes=lanes, roadlines=roadlines, ports=ports, id_counter=id_counter
        )
