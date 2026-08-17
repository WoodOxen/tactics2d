# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Junction approach road segment generator implementation."""

from __future__ import annotations

import logging

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
    apply_lane_side_shift,
    build_lane_from_boundaries,
    build_roadline_from_points,
    lane_role,
    variable_offset_polyline,
)
from .road_segment import RoadSegment

logger = logging.getLogger(__name__)


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
    change_side: str = "both",
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
        change_side: Which side lane-count changes happen on.
            ``"left"`` — extra lane(s) on the left (innermost).
            ``"right"`` — extra lane(s) on the right (outermost).
            ``"both"`` — symmetric (default, matches the old behaviour).

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

    def _pad_side(n_extra: int, side: str) -> tuple[int, int]:
        if side == "left":
            return (n_extra, 0)
        if side == "right":
            return (0, n_extra)
        # "both" — symmetric
        left = n_extra // 2
        return (left, n_extra - left)

    # ---- end offsets: pad to max_n + 1 boundaries -----
    if len(end_offsets_arr) < max_n + 1:
        n_extra = max_n - end_lane_num
        n_left, n_right = _pad_side(n_extra, change_side)

        padded = np.empty(max_n + 1, dtype=float)
        padded[n_left : n_left + len(end_offsets_arr)] = end_offsets_arr
        padded[:n_left] = end_offsets_arr[0]  # collapse outermost left
        padded[n_left + len(end_offsets_arr) :] = end_offsets_arr[-1]  # collapse outermost right
        end_offsets_arr = padded

    # ---- pad start offsets if N < M -------------------
    if len(start_offsets) < max_n + 1:
        n_extra = max_n - start_lane_num
        n_left, n_right = _pad_side(n_extra, change_side)

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
        change_side: Which side lane-count changes happen on.
        lane_side: Which side of the centreline the road-end lane group
            occupies.  ``"center"`` (default) places lanes symmetrically;
            ``"right"`` pins the leftmost boundary to the centreline (TwoWay
            forward convention); ``"left"`` pins the rightmost boundary
            (TwoWay backward convention).  The junction-end boundaries are
            controlled by ``end_boundary_offsets`` and are never shifted.
        min_taper_length: Minimum centreline length for the taper transition in
            metres.  When the provided centreline is shorter a warning is emitted
            but generation continues.
    """

    def __init__(
        self,
        step_size: float = 0.1,
        change_side: str = "both",
        lane_side: str = "center",
        min_taper_length: float | None = None,
        centerline_marking_token: str | None = None,
    ) -> None:
        """Initialise the generator.

        Args:
            step_size: Reference-line sampling interval in metres.
            change_side: Which side lane-count changes happen on.
                ``"left"`` — extra lane(s) on the left (innermost side, towards
                the junction centre).  ``"right"`` — extra lane(s) on the right
                (outermost side).  ``"both"`` — symmetric padding.
            lane_side: Which side of the centreline the road-end lane group
                occupies.  ``"center"`` (default), ``"left"``, or ``"right"``.
            min_taper_length: Minimum centreline length in metres for smooth
                width transitions.  When ``None``, the minimum is computed
                automatically as ``max(lane_width * |Δlane| * 3, 10)``.
            centerline_marking_token: When ``lane_side`` is ``"right"`` or
                ``"left"``, the road-end boundary pinned to the centreline
                continues a TwoWay centre divider.  Pass a marking token string
                to use that marking for the centreline boundary instead of the
                default one-way edge rule.

        Raises:
            ValueError: If ``step_size <= 0``, ``change_side`` is invalid, or
                ``lane_side`` is invalid.
        """
        if step_size <= 0.0:
            raise ValueError("step_size must be positive.")
        if change_side not in ("left", "right", "both"):
            raise ValueError("change_side must be 'left', 'right', or 'both'.")
        if lane_side not in ("center", "left", "right"):
            raise ValueError("lane_side must be 'center', 'left', or 'right'.")
        self.step_size = step_size
        self.change_side = change_side
        self.lane_side = lane_side
        self.min_taper_length = min_taper_length
        self.centerline_marking_token = centerline_marking_token

    def build(  # type: ignore[override]
        self,
        centerline: np.ndarray,
        start_lane_num: int,
        end_lane_num: int,
        end_boundary_offsets: np.ndarray,
        lane_width: float = 3.5,
        speed_limit: float = 50.0,
        change_side: str | None = None,
        start_lane_side: str | None = None,
        centerline_marking_token: str | None = None,
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
            change_side: Per-call override for ``self.change_side``.
                ``None`` uses the generator default.
            start_lane_side: Per-call override for ``self.lane_side``.
                ``None`` uses the generator default.  Only affects the road-end
                boundary offsets; the junction end is controlled by
                ``end_boundary_offsets`` and is never shifted.
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
        if end_offs_arr.ndim != 1 or len(end_offs_arr) != end_n + 1:
            raise ValueError(
                f"end_boundary_offsets must have end_lane_num + 1 = {end_n + 1} "
                f"elements, got {len(end_offs_arr)}."
            )
        if not np.all(np.isfinite(end_offs_arr)):
            raise ValueError("end_boundary_offsets must contain only finite values.")
        if np.any(np.diff(end_offs_arr) >= -1e-6):
            raise ValueError(
                "end_boundary_offsets must be strictly decreasing from leftmost to rightmost."
            )

        center_pts = np.asarray(centerline, dtype=float)
        if center_pts.ndim != 2 or center_pts.shape[1] != 2:
            raise ValueError(f"centreline must have shape (N, 2), got {center_pts.shape}.")
        if len(center_pts) < 2:
            raise ValueError("centreline must have at least 2 points.")
        if not np.all(np.isfinite(center_pts)):
            raise ValueError("centreline must contain only finite values.")

        # ---- resolve change_side ---------------------------------------
        side = self.change_side if change_side is None else change_side
        if side not in ("left", "right", "both"):
            raise ValueError(f"change_side must be 'left', 'right', or 'both', got {side!r}.")

        # ---- resolve lane_side -----------------------------------------
        road_side = start_lane_side if start_lane_side is not None else self.lane_side
        if road_side not in ("center", "left", "right"):
            raise ValueError(f"lane_side must be 'center', 'left', or 'right', got {road_side!r}")
        cl_token = (
            centerline_marking_token
            if centerline_marking_token is not None
            else self.centerline_marking_token
        )

        # ---- minimum taper length check --------------------------------
        lane_delta = abs(end_n - start_n)
        min_len = self.min_taper_length
        if min_len is None:
            min_len = max(lane_w * lane_delta * 3, 10.0)

        center_len = float(cumulative_s(center_pts)[-1])
        if lane_delta > 0 and center_len < min_len:
            logger.warning(
                "Taper centreline length %.1f m is shorter than recommended "
                "minimum %.1f m for a %d→%d lane transition (Δ=%d×%.1f m). "
                "The resulting boundaries may be noticeably stiff.",
                center_len,
                min_len,
                start_n,
                end_n,
                lane_delta,
                lane_w,
            )

        # ---- boundary offsets ------------------------------------------
        raw_start_offsets, end_offsets = _approach_boundary_offsets(
            start_lane_num=start_n,
            end_lane_num=end_n,
            lane_width=lane_w,
            end_boundary_offsets=end_offs_arr,
            change_side=side,
        )

        # ---- apply lane-side shift to road end only --------------------
        start_offsets = np.asarray(
            apply_lane_side_shift(raw_start_offsets.tolist(), road_side), dtype=float
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

        # ---- identify centreline boundary (road-end only) ---------------
        cl_boundary_idx: int | None = None
        if road_side == "right":
            cl_boundary_idx = 0
        elif road_side == "left":
            cl_boundary_idx = boundary_num - 1

        # ---- generate boundaries and roadlines -------------------------
        boundary_points: list[np.ndarray] = []
        roadlines: list[RoadLine] = []
        lanes: list[Lane] = []
        id_counter = id_offset

        for b_idx in range(boundary_num):
            pts = variable_offset_polyline(
                centerline=center_pts,
                start_offset=float(start_offsets[b_idx]),
                end_offset=float(end_offsets[b_idx]),
            )
            boundary_points.append(pts)

            if b_idx == cl_boundary_idx and cl_token is not None:
                marking_token = cl_token
            else:
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
            role = lane_role(
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
            point=end_pt, heading=end_heading, lane_num=end_n, lane_width=lane_w, speed_limit=speed
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
