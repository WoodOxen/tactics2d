# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Lane adapter road segment generator implementation."""

from __future__ import annotations

import numpy as np

from tactics2d.geometry import polyline
from tactics2d.map.element import Lane, RoadLine
from tactics2d.map.generator.rules.lane_marking_rules import (
    one_way_boundary_token,
    roadline_render_kwargs,
)
from tactics2d.map.generator.rules.module_types import RoadModuleResult, RoadPort, build_port

from .element_builder import (
    add_ordered_lane_neighbors,
    build_lane_from_boundaries,
    build_roadline_from_points,
)
from .reference_line import fit_reference_line
from .road_segment import RoadSegment


def _variable_offset_polyline(
    centerline: np.ndarray, start_offset: float, end_offset: float
) -> np.ndarray:
    """Offset a polyline with a smoothly varying lateral offset.

    The offset transitions from ``start_offset`` to ``end_offset`` via a
    smoothstep (zero-slope Hermite cubic) function of arc length, producing
    zero first derivative at both ends.

    Args:
        centerline: Centreline points with shape ``(N, 2)``.
        start_offset: Lateral offset at the entry end in metres.
            Positive = left of travel direction.
        end_offset: Lateral offset at the exit end in metres.
            Positive = left of travel direction.

    Returns:
        Offset polyline with shape ``(N, 2)``.
    """
    pts = np.asarray(centerline, dtype=float)

    if len(pts) < 2:
        return pts.copy()

    cumulative_lengths = polyline.arc_lengths(pts)
    total = cumulative_lengths[-1]
    t = np.zeros(len(pts)) if total < 1e-9 else cumulative_lengths / total
    t = t * t * (3.0 - 2.0 * t)

    tangents = np.empty_like(pts)
    tangents[0] = pts[1] - pts[0]
    tangents[-1] = pts[-1] - pts[-2]
    if len(pts) > 2:
        tangents[1:-1] = pts[2:] - pts[:-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents /= np.where(norms < 1e-9, 1.0, norms)
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    offsets = float(start_offset) + (float(end_offset) - float(start_offset)) * t
    return pts + offsets[:, None] * normals


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


def _lane_role(start_width: float, end_width: float) -> str:
    """Classify a lane's role in the adapter transition.

    Args:
        start_width: Lane width at the entry end in metres.
        end_width: Lane width at the exit end in metres.

    Returns:
        ``"added"`` if the lane opens from zero width, ``"dropped"`` if it
        closes to zero, or ``"through"`` otherwise.
    """
    eps = 1e-3

    if start_width <= eps and end_width > eps:
        return "added"

    if start_width > eps and end_width <= eps:
        return "dropped"

    return "through"


class LaneAdapter(RoadSegment):
    """Same-direction lane-count adapter generator.

    Smoothly transitions between two road sections with different lane counts
    (difference of at most one). The lane that is added or dropped tapers from
    zero width at one end to full width at the other via smoothstep interpolation.

    Attributes:
        change_side: Side where the lane is added or dropped. ``"right"`` means
            the rightmost lane changes; ``"left"`` means the leftmost lane changes.
        step_size: Reference-line sampling interval in metres.
    """

    def __init__(self, change_side: str = "right", step_size: float = 0.1) -> None:
        """Initialise the generator.

        Args:
            change_side: Side where the lane count changes. Must be ``"left"``
                or ``"right"``.
            step_size: Reference-line sampling interval in metres.

        Raises:
            ValueError: If ``change_side`` is not ``"left"`` or ``"right"``,
                or ``step_size <= 0``.
        """
        if change_side not in ("left", "right"):
            raise ValueError("change_side must be 'left' or 'right'.")
        if step_size <= 0.0:
            raise ValueError("step_size must be positive.")
        self.change_side = change_side
        self.step_size = step_size

    def build(
        self,
        start_port: RoadPort,
        end_port: RoadPort,
        *,
        start_lane_num: int | None = None,
        end_lane_num: int | None = None,
        lane_width: float | None = None,
        speed_limit: float | None = None,
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

        center_pts = fit_reference_line(
            start_port.point, start_port.heading, end_port.point, end_port.heading, self.step_size
        )

        start_offsets, end_offsets = _boundary_offsets_for_adapter(
            start_lane_num=start_n,
            end_lane_num=end_n,
            lane_width=lane_w,
            change_side=self.change_side,
        )

        max_lane_num = max(start_n, end_n)
        boundary_num = max_lane_num + 1

        boundary_points: list[np.ndarray] = []
        roadlines: list[RoadLine] = []
        lanes: list[Lane] = []
        id_counter = id_offset

        for boundary_idx in range(boundary_num):
            pts = _variable_offset_polyline(
                centerline=center_pts,
                start_offset=start_offsets[boundary_idx],
                end_offset=end_offsets[boundary_idx],
            )
            boundary_points.append(pts)

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
            role = _lane_role(start_width, end_width)

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
