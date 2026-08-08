# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Lane adapter road segment generator implementation."""

from __future__ import annotations

import numpy as np

from tactics2d.map.element import Lane, RoadLine
from tactics2d.map.generator.rules.lane_marking_rules import (
    one_way_boundary_token,
    roadline_render_kwargs,
)
from tactics2d.map.generator.rules.module_types import RoadModuleResult, RoadPort, build_port

from .element_builder import (
    add_ordered_lane_neighbors,
    apply_lane_side_shift,
    build_lane_from_boundaries,
    build_roadline_from_points,
    lane_role,
    variable_offset_polyline,
)
from .reference_line import fit_reference_line
from .road_segment import RoadSegment


def _boundary_offsets_for_adapter(
    start_lane_num: int, end_lane_num: int, lane_width: float, change_side: str
) -> tuple[list[float], list[float]]:
    """Return start and end lateral boundary offsets for the lane-count adapter.

    The lane that is added or dropped is represented by a zero-width slot padded
    on the appropriate side at the narrower end.  All offsets are measured from
    the road centreline; positive = left of travel direction.

    Args:
        start_lane_num: Number of lanes at the adapter entry.
        end_lane_num: Number of lanes at the adapter exit.
        lane_width: Uniform lane width in metres.
        change_side: Side on which the lane count changes (``"left"`` or
            ``"right"``).

    Returns:
        Tuple ``(start_offsets, end_offsets)`` each of length
        ``max(start_lane_num, end_lane_num) + 1``.
    """
    lane_delta = end_lane_num - start_lane_num

    def _uniform(n: int) -> list[float]:
        half = n * lane_width / 2.0
        return [half - i * lane_width for i in range(n + 1)]

    def _padded(n: int, pad_side: str) -> list[float]:
        offsets = _uniform(n)
        return offsets + [offsets[-1]] if pad_side == "right" else [offsets[0]] + offsets

    max_lane_num = max(start_lane_num, end_lane_num)
    if lane_delta == 0:
        offsets = _uniform(start_lane_num)
        return offsets, offsets
    if lane_delta > 0:
        return _padded(start_lane_num, change_side), _uniform(max_lane_num)
    return _uniform(max_lane_num), _padded(end_lane_num, change_side)


class LaneAdapter(RoadSegment):
    """Same-direction lane-count adapter generator.

    Smoothly transitions between two road sections with different lane counts
    (difference of at most one). The lane that is added or dropped tapers from
    zero width at one end to full width at the other via smoothstep interpolation.

    Attributes:
        change_side: Side where the lane is added or dropped. ``"right"`` means
            the rightmost lane changes; ``"left"`` means the leftmost lane changes.
        lane_side: Which side of the centreline the lane group occupies.
            ``"center"`` (default) places lanes symmetrically around the
            centreline, matching :class:`OneWay` convention.  ``"right"`` pins
            the leftmost boundary to the centreline so all lanes sit on the
            right, matching :class:`TwoWay` forward convention.  ``"left"``
            pins the rightmost boundary to the centreline (TwoWay backward
            convention).
        step_size: Reference-line sampling interval in metres.
    """

    def __init__(
        self,
        change_side: str = "right",
        step_size: float = 0.1,
        lane_side: str = "center",
        centerline_marking_token: str | None = None,
    ) -> None:
        """Initialise the generator.

        Args:
            change_side: Side where the lane count changes. Must be ``"left"``
                or ``"right"``.
            step_size: Reference-line sampling interval in metres.
            lane_side: Which side of the centreline all lanes sit on.
                ``"center"`` (default), ``"left"``, or ``"right"``.
            centerline_marking_token: When ``lane_side`` is ``"right"`` or
                ``"left"``, the boundary pinned to the centreline continues
                the TwoWay centre divider.  Pass a marking token string
                (e.g. ``"solid_double_yellow"`` or the result of
                :func:`~tactics2d.map.generator.rules.lane_marking_rules.two_way_centerline`)
                to use that marking instead of the one-way edge rule.
                Ignored when ``lane_side`` is ``"center"``.

        Raises:
            ValueError: If ``change_side`` is not ``"left"`` or ``"right"``,
                ``lane_side`` is invalid, or ``step_size <= 0``.
        """
        if change_side not in ("left", "right"):
            raise ValueError("change_side must be 'left' or 'right'.")
        if lane_side not in ("center", "left", "right"):
            raise ValueError("lane_side must be 'center', 'left', or 'right'.")
        if step_size <= 0.0:
            raise ValueError("step_size must be positive.")
        self.change_side = change_side
        self.lane_side = lane_side
        self.step_size = step_size
        self.centerline_marking_token = centerline_marking_token

    def build(
        self,
        start_port: RoadPort,
        end_port: RoadPort,
        *,
        start_lane_num: int | None = None,
        end_lane_num: int | None = None,
        lane_width: float | None = None,
        speed_limit: float | None = None,
        start_lane_side: str | None = None,
        end_lane_side: str | None = None,
        centerline_marking_token: str | None = None,
        id_offset: int = 0,
    ) -> RoadModuleResult:
        """Build a lane-count adapter between two ports.

        Args:
            start_port: Upstream socket.
            end_port: Downstream socket.
            start_lane_num: Number of lanes at the entry. Defaults to
                ``start_port.lane_num``.
            end_lane_num: Number of lanes at the exit. Defaults to
                ``end_port.lane_num``.
            lane_width: Lane width in metres. Defaults to ``start_port.lane_width``.
            speed_limit: Speed limit in km/h. Defaults to ``start_port.speed_limit``.
            start_lane_side: Override ``self.lane_side`` for the start end.
                ``None`` uses the generator default.
            end_lane_side: Override ``self.lane_side`` for the end end.
                ``None`` uses the generator default.
            centerline_marking_token: Per-call override for
                ``self.centerline_marking_token``.  ``None`` uses the generator
                default.
            id_offset: First element id.

        Returns:
            :class:`RoadModuleResult` with ports ``"entry"`` (``kind="adapter_in"``)
            and ``"exit"`` (``kind="adapter_out"``).

        Raises:
            ValueError: If either lane count is less than 1, ``lane_width <= 0``,
                or the lane count difference exceeds 1.
        """
        start_n = int(start_lane_num if start_lane_num is not None else start_port.lane_num)
        end_n = int(end_lane_num if end_lane_num is not None else end_port.lane_num)
        lane_w = float(lane_width if lane_width is not None else start_port.lane_width)
        speed = float(speed_limit if speed_limit is not None else start_port.speed_limit)
        start_side = start_lane_side if start_lane_side is not None else self.lane_side
        end_side = end_lane_side if end_lane_side is not None else self.lane_side
        cl_token = (
            centerline_marking_token
            if centerline_marking_token is not None
            else self.centerline_marking_token
        )

        if start_n < 1:
            raise ValueError("start_lane_num must be >= 1.")
        if end_n < 1:
            raise ValueError("end_lane_num must be >= 1.")
        if lane_w <= 0.0:
            raise ValueError("lane_width must be positive.")
        if abs(end_n - start_n) > 1:
            raise ValueError(
                "lane_adapter v1 only supports lane count difference of 1. "
                "Use chained adapters, e.g. 2 -> 3 -> 4."
            )
        for side in (start_side, end_side):
            if side not in ("center", "left", "right"):
                raise ValueError(f"lane_side must be 'center', 'left', or 'right', got {side!r}")

        center_pts = fit_reference_line(
            start_port.point, start_port.heading, end_port.point, end_port.heading, self.step_size
        )

        raw_start_offsets, raw_end_offsets = _boundary_offsets_for_adapter(
            start_lane_num=start_n,
            end_lane_num=end_n,
            lane_width=lane_w,
            change_side=self.change_side,
        )

        start_offsets = apply_lane_side_shift(raw_start_offsets, start_side)
        end_offsets = apply_lane_side_shift(raw_end_offsets, end_side)

        max_lane_num = max(start_n, end_n)
        boundary_num = max_lane_num + 1

        # ---- identify centreline boundary (if any) -----------------------
        cl_boundary_idx: int | None = None
        if start_side == end_side == "right":
            cl_boundary_idx = 0
        elif start_side == end_side == "left":
            cl_boundary_idx = boundary_num - 1

        boundary_points: list[np.ndarray] = []
        roadlines: list[RoadLine] = []
        lanes: list[Lane] = []
        id_counter = id_offset

        for boundary_idx in range(boundary_num):
            pts = variable_offset_polyline(
                centerline=center_pts,
                start_offset=start_offsets[boundary_idx],
                end_offset=end_offsets[boundary_idx],
            )
            boundary_points.append(pts)

            if boundary_idx == cl_boundary_idx and cl_token is not None:
                marking_token = cl_token
            else:
                marking_token = one_way_boundary_token(boundary_idx, boundary_num)

            roadline = build_roadline_from_points(
                id_=id_counter,
                points=pts,
                marking_kwargs=roadline_render_kwargs(
                    marking_token,
                    {
                        "module": "lane_adapter",
                        "boundary_index": boundary_idx,
                        "change_side": self.change_side,
                        "start_offset": float(start_offsets[boundary_idx]),
                        "end_offset": float(end_offsets[boundary_idx]),
                    },
                ),
            )
            id_counter += 1

            roadlines.append(roadline)

        for lane_idx in range(max_lane_num):
            left_pts = boundary_points[lane_idx]
            right_pts = boundary_points[lane_idx + 1]

            start_width = abs(start_offsets[lane_idx] - start_offsets[lane_idx + 1])
            end_width = abs(end_offsets[lane_idx] - end_offsets[lane_idx + 1])
            role = lane_role(start_width, end_width)

            lane = build_lane_from_boundaries(
                id_=id_counter,
                left_points=left_pts,
                right_points=right_pts,
                left_roadline_ids=roadlines[lane_idx].id_,
                right_roadline_ids=roadlines[lane_idx + 1].id_,
                speed_limit=speed,
                custom_tags={
                    "module": "lane_adapter",
                    "lane_index": lane_idx,
                    "lane_role": role,
                    "change_side": self.change_side,
                    "start_width": float(start_width),
                    "end_width": float(end_width),
                },
            )
            id_counter += 1

            lanes.append(lane)

        add_ordered_lane_neighbors(lanes)

        active_start_indices = np.where(np.abs(np.diff(start_offsets)) > 1e-3)[0].tolist()
        active_end_indices = np.where(np.abs(np.diff(end_offsets)) > 1e-3)[0].tolist()

        entry_lane_ids = tuple(lanes[i].id_ for i in active_start_indices)
        exit_lane_ids = tuple(lanes[i].id_ for i in active_end_indices)

        entry_base = RoadPort(
            point=np.asarray(start_port.point, dtype=float),
            heading=float(start_port.heading),
            lane_num=start_n,
            lane_width=lane_w,
            speed_limit=speed,
            metadata=dict(start_port.metadata),
        )
        exit_base = RoadPort(
            point=np.asarray(end_port.point, dtype=float),
            heading=float(end_port.heading),
            lane_num=end_n,
            lane_width=lane_w,
            speed_limit=speed,
            metadata=dict(end_port.metadata),
        )

        ports = {
            "entry": build_port(
                entry_base,
                kind="adapter_in",
                name="entry",
                lane_ids=entry_lane_ids,
                metadata={
                    "module": "lane_adapter",
                    "change_side": self.change_side,
                    "lane_num": start_n,
                },
            ),
            "exit": build_port(
                exit_base,
                kind="adapter_out",
                name="exit",
                lane_ids=exit_lane_ids,
                metadata={
                    "module": "lane_adapter",
                    "change_side": self.change_side,
                    "lane_num": end_n,
                },
            ),
        }

        return RoadModuleResult(
            lanes=lanes, roadlines=roadlines, ports=ports, id_counter=id_counter
        )
