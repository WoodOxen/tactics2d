# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Junction approach road segment generator implementation."""

from __future__ import annotations

import numpy as np

from tactics2d.geometry import cumulative_s
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

    cum = cumulative_s(pts)
    total = cum[-1]
    t = np.zeros(len(pts)) if total < 1e-9 else cum / total
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


def _heading_at(pts: np.ndarray, at_end: bool = False) -> float:
    """Heading angle of a polyline at its start or end."""
    if len(pts) < 2:
        return 0.0
    d = pts[-1] - pts[-2] if at_end else pts[1] - pts[0]
    return float(np.arctan2(d[1], d[0]))


def _approach_boundary_offsets(
    start_lane_num: int,
    end_lane_num: int,
    lane_width: float,
    end_boundary_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute start and end lateral boundary offsets for the junction approach.

    Start offsets are uniformly spaced for N lanes.  End offsets match the
    junction port's boundary positions, with outer slots collapsed (zero-width)
    for lanes that are dropped or added.

    Args:
        start_lane_num: Number of road lanes at the approach entry (N).
        end_lane_num: Number of junction port connector lanes (M).
        lane_width: Uniform lane width at the road side in metres.
        end_boundary_offsets: Lateral offsets of the junction port's M lanes'
            boundaries, sorted from leftmost to rightmost in travel direction.
            Shape ``(M + 1,)``.

    Returns:
        Tuple ``(start_offsets, end_offsets)`` each of shape ``(max(N, M) + 1,)``.
    """
    max_n = max(start_lane_num, end_lane_num)
    end_offsets_arr = np.asarray(end_boundary_offsets, dtype=float)

    # ---- start offsets: uniform -----------------------
    half = start_lane_num * lane_width / 2.0
    start_offsets = np.array(
        [half - i * lane_width for i in range(start_lane_num + 1)], dtype=float
    )

    # ---- end offsets: pad to max_n + 1 boundaries -----
    if len(end_offsets_arr) < max_n + 1:
        n_extra = max_n - end_lane_num
        n_left = n_extra // 2
        n_right = n_extra - n_left

        padded = np.empty(max_n + 1, dtype=float)
        padded[n_left : n_left + len(end_offsets_arr)] = end_offsets_arr
        padded[:n_left] = end_offsets_arr[0]  # collapse outermost left
        padded[n_left + len(end_offsets_arr) :] = end_offsets_arr[-1]  # collapse outermost right
        end_offsets_arr = padded

    # ---- pad start offsets if N < M -------------------
    if len(start_offsets) < max_n + 1:
        n_extra = max_n - start_lane_num
        n_left = n_extra // 2
        n_right = n_extra - n_left

        padded = np.empty(max_n + 1, dtype=float)
        padded[n_left : n_left + len(start_offsets)] = start_offsets
        padded[:n_left] = start_offsets[0]
        padded[n_left + len(start_offsets) :] = start_offsets[-1]
        start_offsets = padded

    return start_offsets, end_offsets_arr


class JunctionApproach(RoadSegment):
    """Generate tapered approach lanes connecting a road to a junction port.

    Smoothly transitions from *N* uniform road lanes to *M* junction port
    connector lanes.  The outermost ``|N-M|`` lanes taper from full width to
    zero (or open from zero) via smoothstep interpolation.  Unlike
    :class:`LaneAdapter`, this generator accepts a pre-computed centreline
    and non-uniform port boundary offsets so the taper follows the actual road
    geometry.

    Attributes:
        step_size: Reference-line sampling interval in metres.
    """

    def __init__(self, step_size: float = 0.1) -> None:
        """Initialise the generator.

        Args:
            step_size: Reference-line sampling interval in metres.

        Raises:
            ValueError: If ``step_size <= 0``.
        """
        if step_size <= 0.0:
            raise ValueError("step_size must be positive.")
        self.step_size = step_size

    def build(  # type: ignore[override]
        self,
        centerline: np.ndarray,
        start_lane_num: int,
        end_lane_num: int,
        end_boundary_offsets: np.ndarray,
        lane_width: float = 3.5,
        speed_limit: float = 50.0,
        *,
        id_offset: int = 0,
    ) -> RoadModuleResult:
        """Build a junction approach segment.

        Args:
            centerline: Centreline points from the taper start to the junction
                port point, shape ``(K, 2)``.  Travel direction is from the
                first point (road side) toward the last point (junction side).
            start_lane_num: Number of road lanes at the approach entry.
            end_lane_num: Number of junction port connector lanes.
            end_boundary_offsets: Lateral offsets of the port's lane boundaries
                (leftmost to rightmost in travel direction), shape ``(M + 1,)``.
            lane_width: Uniform lane width in metres at the road side.
            speed_limit: Speed limit in km/h for the approach lanes.
            id_offset: First element id.

        Returns:
            :class:`RoadModuleResult` with ports ``"entry"`` (road side) and
            ``"exit"`` (junction side).

        Raises:
            ValueError: If either lane count is less than 1, ``lane_width <= 0``,
                ``end_boundary_offsets`` does not have ``end_lane_num + 1``
                elements, or the centreline has fewer than 2 points.
        """
        start_n = int(start_lane_num)
        end_n = int(end_lane_num)
        lane_w = float(lane_width)
        speed = float(speed_limit)

        if start_n < 1:
            raise ValueError("start_lane_num must be >= 1.")
        if end_n < 1:
            raise ValueError("end_lane_num must be >= 1.")
        if lane_w <= 0.0:
            raise ValueError("lane_width must be positive.")

        end_offs_arr = np.asarray(end_boundary_offsets, dtype=float)
        if len(end_offs_arr) != end_n + 1:
            raise ValueError(
                f"end_boundary_offsets must have end_lane_num + 1 = {end_n + 1} "
                f"elements, got {len(end_offs_arr)}."
            )

        center_pts = np.asarray(centerline, dtype=float)
        if len(center_pts) < 2:
            raise ValueError("centreline must have at least 2 points.")

        # ---- boundary offsets ------------------------------------------
        start_offsets, end_offsets = _approach_boundary_offsets(
            start_lane_num=start_n,
            end_lane_num=end_n,
            lane_width=lane_w,
            end_boundary_offsets=end_offs_arr,
        )

        max_n = max(start_n, end_n)
        boundary_num = max_n + 1

        # ---- resample centreline to uniform step size ------------------
        cum = cumulative_s(center_pts)
        total_len = float(cum[-1])
        if total_len > self.step_size:
            n_steps = max(2, int(total_len / self.step_size) + 1)
            s_vals = np.linspace(0.0, total_len, n_steps)
            resampled = np.empty((n_steps, 2), dtype=float)
            seg_idx = 0
            for i, s in enumerate(s_vals):
                while seg_idx < len(cum) - 1 and cum[seg_idx + 1] < s:
                    seg_idx += 1
                if seg_idx >= len(cum) - 1:
                    resampled[i] = center_pts[-1]
                else:
                    seg_len = cum[seg_idx + 1] - cum[seg_idx]
                    t_val = 0.0 if seg_len < 1e-12 else (s - cum[seg_idx]) / seg_len
                    resampled[i] = center_pts[seg_idx] + t_val * (
                        center_pts[seg_idx + 1] - center_pts[seg_idx]
                    )
            center_pts = resampled

        # ---- generate boundaries and roadlines -------------------------
        boundary_points: list[np.ndarray] = []
        roadlines: list[RoadLine] = []
        lanes: list[Lane] = []
        id_counter = id_offset

        for b_idx in range(boundary_num):
            pts = _variable_offset_polyline(
                centerline=center_pts,
                start_offset=float(start_offsets[b_idx]),
                end_offset=float(end_offsets[b_idx]),
            )
            boundary_points.append(pts)

            marking_token = one_way_boundary_token(b_idx, boundary_num)

            roadline = build_roadline_from_points(
                id_=id_counter,
                points=pts,
                marking_kwargs=roadline_render_kwargs(
                    marking_token,
                    {
                        "module": "junction_approach",
                        "boundary_index": b_idx,
                        "start_offset": float(start_offsets[b_idx]),
                        "end_offset": float(end_offsets[b_idx]),
                    },
                ),
            )
            id_counter += 1
            roadlines.append(roadline)

        # ---- build lanes from adjacent boundaries ----------------------
        for lane_idx in range(max_n):
            left_pts = boundary_points[lane_idx]
            right_pts = boundary_points[lane_idx + 1]
            role = _lane_role(
                abs(start_offsets[lane_idx] - start_offsets[lane_idx + 1]),
                abs(end_offsets[lane_idx] - end_offsets[lane_idx + 1]),
            )

            lane = build_lane_from_boundaries(
                id_=id_counter,
                left_points=left_pts,
                right_points=right_pts,
                left_roadline_ids=roadlines[lane_idx].id_,
                right_roadline_ids=roadlines[lane_idx + 1].id_,
                speed_limit=speed,
                custom_tags={
                    "module": "junction_approach",
                    "lane_index": lane_idx,
                    "lane_role": role,
                },
            )
            id_counter += 1
            lanes.append(lane)

        add_ordered_lane_neighbors(lanes)

        # ---- determine which lane slots are active at each end ---------
        start_widths = np.abs(np.diff(start_offsets))
        end_widths = np.abs(np.diff(end_offsets))

        active_start = np.where(start_widths > 1e-3)[0].tolist()
        active_end = np.where(end_widths > 1e-3)[0].tolist()

        entry_lane_ids: tuple[str, ...] = tuple(lanes[i].id_ for i in active_start)
        exit_lane_ids: tuple[str, ...] = tuple(lanes[i].id_ for i in active_end)

        # ---- build entry / exit ports ---------------------------------
        start_pt = center_pts[0]
        end_pt = center_pts[-1]
        start_heading = _heading_at(center_pts, at_end=False)
        end_heading = _heading_at(center_pts, at_end=True)

        entry_port = RoadPort(
            point=start_pt,
            heading=start_heading,
            lane_num=start_n,
            lane_width=lane_w,
            speed_limit=speed,
        )
        exit_port = RoadPort(
            point=end_pt,
            heading=end_heading,
            lane_num=end_n,
            lane_width=lane_w,
            speed_limit=speed,
        )

        ports = {
            "entry": build_port(
                entry_port,
                kind="junction_approach_in",
                name="entry",
                lane_ids=entry_lane_ids,
                metadata={"module": "junction_approach", "lane_num": start_n},
            ),
            "exit": build_port(
                exit_port,
                kind="junction_approach_out",
                name="exit",
                lane_ids=exit_lane_ids,
                metadata={"module": "junction_approach", "lane_num": end_n},
            ),
        }

        return RoadModuleResult(
            lanes=lanes, roadlines=roadlines, ports=ports, id_counter=id_counter
        )


def _lane_role(start_width: float, end_width: float) -> str:
    """Classify a lane's role in the approach transition."""
    eps = 1e-3
    if start_width <= eps and end_width > eps:
        return "added"
    if start_width > eps and end_width <= eps:
        return "dropped"
    return "through"
