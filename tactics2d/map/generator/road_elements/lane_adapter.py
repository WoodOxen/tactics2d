# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Lane adapter generator for same-direction road width transitions."""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString

from tactics2d.map.element import Lane, LaneRelationship, RoadLine

from ..geometry.module_geometry import (
    curvature_stats,
    fit_reference_line,
    has_self_intersection,
    polyline_length,
)
from ..rules.lane_marking_rules import one_way_mark, roadline_render_kwargs
from ..rules.module_types import RoadModuleResult, RoadPort, make_port, ports_to_interfaces


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Smooth interpolation from 0 to 1."""
    return t * t * (3.0 - 2.0 * t)


def _polyline_t(points: np.ndarray) -> np.ndarray:
    """Return normalized cumulative arc-length parameter for a polyline."""
    if len(points) < 2:
        return np.zeros(len(points), dtype=float)

    seg_lens = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum_s = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum_s[-1]

    if total < 1e-9:
        return np.zeros(len(points), dtype=float)

    return cum_s / total


def _polyline_normals(points: np.ndarray) -> np.ndarray:
    """Return left normals along a polyline."""
    normals = np.zeros_like(points, dtype=float)

    if len(points) < 2:
        normals[:, 1] = 1.0
        return normals

    for i in range(len(points)):
        if i == 0:
            tangent = points[1] - points[0]
        elif i == len(points) - 1:
            tangent = points[-1] - points[-2]
        else:
            tangent = points[i + 1] - points[i - 1]

        norm = float(np.linalg.norm(tangent))
        if norm < 1e-9:
            normals[i] = np.array([0.0, 1.0])
        else:
            tangent = tangent / norm
            normals[i] = np.array([-tangent[1], tangent[0]])

    return normals


def _variable_offset_polyline(
    centerline: np.ndarray, start_offset: float, end_offset: float
) -> np.ndarray:
    """Offset a polyline with smoothly varying lateral offset."""
    centerline = np.asarray(centerline, dtype=float)
    t = _smoothstep(_polyline_t(centerline))
    normals = _polyline_normals(centerline)
    offsets = start_offset + (end_offset - start_offset) * t

    return centerline + offsets[:, None] * normals


def _standard_boundary_offsets(lane_num: int, lane_width: float) -> list[float]:
    """Return boundary offsets from left edge to right edge."""
    half_width = lane_num * lane_width / 2.0
    return [half_width - i * lane_width for i in range(lane_num + 1)]


def _expanded_boundary_offsets(
    lane_num: int, max_lane_num: int, lane_width: float, missing_side: str
) -> list[float]:
    """Return boundary offsets padded with a zero-width lane on one side."""
    if lane_num == max_lane_num:
        return _standard_boundary_offsets(lane_num, lane_width)

    if lane_num != max_lane_num - 1:
        raise ValueError("lane_adapter v1 only supports lane count difference of 1.")

    offsets = _standard_boundary_offsets(lane_num, lane_width)

    if missing_side == "right":
        return offsets + [offsets[-1]]

    if missing_side == "left":
        return [offsets[0]] + offsets

    raise ValueError("missing_side must be 'left' or 'right'.")


def _boundary_offsets_for_adapter(
    start_lane_num: int, end_lane_num: int, lane_width: float, change_side: str
) -> tuple[list[float], list[float]]:
    """Return start and end boundary offsets for the adapter."""
    if change_side not in ("left", "right"):
        raise ValueError("lane_adapter v1 only supports change_side='left' or 'right'.")

    lane_delta = end_lane_num - start_lane_num
    if abs(lane_delta) > 1:
        raise ValueError(
            "lane_adapter v1 only supports lane count difference of 1. "
            "Use multiple adapters for larger changes."
        )

    max_lane_num = max(start_lane_num, end_lane_num)

    if lane_delta == 0:
        offsets = _standard_boundary_offsets(start_lane_num, lane_width)
        return offsets, offsets

    if lane_delta > 0:
        start_offsets = _expanded_boundary_offsets(
            lane_num=start_lane_num,
            max_lane_num=max_lane_num,
            lane_width=lane_width,
            missing_side=change_side,
        )
        end_offsets = _standard_boundary_offsets(end_lane_num, lane_width)
    else:
        start_offsets = _standard_boundary_offsets(start_lane_num, lane_width)
        end_offsets = _expanded_boundary_offsets(
            lane_num=end_lane_num,
            max_lane_num=max_lane_num,
            lane_width=lane_width,
            missing_side=change_side,
        )

    return start_offsets, end_offsets


def _active_lane_indices(offsets: list[float], min_width: float = 1e-3) -> list[int]:
    """Return lane indices whose width is non-zero at an adapter end."""
    active = []

    for i in range(len(offsets) - 1):
        width = abs(offsets[i] - offsets[i + 1])
        if width > min_width:
            active.append(i)

    return active


def _boundary_marking_token(boundary_index: int, boundary_num: int) -> str:
    """Return active-standard token for an adapter one-way boundary."""
    lane_num = boundary_num - 1

    if boundary_index == 0:
        return one_way_mark(0, lane_num, "left")

    if boundary_index == boundary_num - 1:
        return one_way_mark(lane_num - 1, lane_num, "right")

    return one_way_mark(boundary_index - 1, lane_num, "right")


def _lane_role(start_width: float, end_width: float) -> str:
    """Classify lane role in the adapter."""
    eps = 1e-3

    if start_width <= eps and end_width > eps:
        return "added"

    if start_width > eps and end_width <= eps:
        return "dropped"

    return "through"


def lane_adapter(
    start_port: RoadPort,
    end_port: RoadPort,
    *,
    start_lane_num: int | None = None,
    end_lane_num: int | None = None,
    lane_width: float | None = None,
    speed_limit: float | None = None,
    change_side: str = "right",
    step_size: float = 0.1,
    id_offset: int = 0,
) -> RoadModuleResult:
    """Generate a same-direction lane-count adapter."""
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
    if step_size <= 0.0:
        raise ValueError("step_size must be positive.")
    if change_side not in ("left", "right"):
        raise ValueError("change_side must be 'left' or 'right'.")
    if abs(end_n - start_n) > 1:
        raise ValueError(
            "lane_adapter v1 only supports lane count difference of 1. "
            "Use chained adapters, e.g. 2 -> 3 -> 4."
        )

    center_pts = fit_reference_line(
        start_port.point, start_port.heading, end_port.point, end_port.heading, step_size
    )

    start_offsets, end_offsets = _boundary_offsets_for_adapter(
        start_lane_num=start_n, end_lane_num=end_n, lane_width=lane_w, change_side=change_side
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

        marking_token = _boundary_marking_token(boundary_idx, boundary_num)

        roadline = RoadLine(
            id_=str(id_counter),
            geometry=LineString(pts),
            **roadline_render_kwargs(
                marking_token,
                {
                    "module": "lane_adapter",
                    "boundary_index": boundary_idx,
                    "change_side": change_side,
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

        left_line = LineString(left_pts)
        right_line = LineString(right_pts)

        start_width = abs(start_offsets[lane_idx] - start_offsets[lane_idx + 1])
        end_width = abs(end_offsets[lane_idx] - end_offsets[lane_idx + 1])
        role = _lane_role(start_width, end_width)

        lane = Lane(
            id_=str(id_counter),
            left_side=left_line,
            right_side=right_line,
            subtype="road",
            speed_limit=speed,
            speed_limit_unit="km/h",
            line_ids={"left": [roadlines[lane_idx].id_], "right": [roadlines[lane_idx + 1].id_]},
            custom_tags={
                "module": "lane_adapter",
                "lane_index": lane_idx,
                "lane_role": role,
                "change_side": change_side,
                "start_width": float(start_width),
                "end_width": float(end_width),
            },
        )
        id_counter += 1

        lanes.append(lane)

    for i, lane in enumerate(lanes):
        if i > 0:
            lane.add_related_lane(lanes[i - 1].id_, LaneRelationship.LEFT_NEIGHBOR)
        if i < len(lanes) - 1:
            lane.add_related_lane(lanes[i + 1].id_, LaneRelationship.RIGHT_NEIGHBOR)

    active_start_indices = _active_lane_indices(start_offsets)
    active_end_indices = _active_lane_indices(end_offsets)

    entry_lane_ids = tuple(lanes[i].id_ for i in active_start_indices)
    exit_lane_ids = tuple(lanes[i].id_ for i in active_end_indices)

    entry_base = RoadPort(
        point=np.asarray(start_port.point, dtype=float),
        heading=float(start_port.heading),
        lane_num=start_n,
        lane_width=lane_w,
        speed_limit=speed,
        metadata=dict(getattr(start_port, "metadata", {})),
    )
    exit_base = RoadPort(
        point=np.asarray(end_port.point, dtype=float),
        heading=float(end_port.heading),
        lane_num=end_n,
        lane_width=lane_w,
        speed_limit=speed,
        metadata=dict(getattr(end_port, "metadata", {})),
    )

    ports = {
        "entry": make_port(
            entry_base,
            kind="adapter_in",
            name="entry",
            lane_ids=entry_lane_ids,
            metadata={"module": "lane_adapter", "change_side": change_side, "lane_num": start_n},
        ),
        "exit": make_port(
            exit_base,
            kind="adapter_out",
            name="exit",
            lane_ids=exit_lane_ids,
            metadata={"module": "lane_adapter", "change_side": change_side, "lane_num": end_n},
        ),
    }

    added_lane_ids = [lane.id_ for lane in lanes if lane.custom_tags.get("lane_role") == "added"]
    dropped_lane_ids = [
        lane.id_ for lane in lanes if lane.custom_tags.get("lane_role") == "dropped"
    ]

    stats = curvature_stats(center_pts)
    self_intersection = has_self_intersection(center_pts)

    quality = {
        "module": "lane_adapter",
        "start_lane_num": start_n,
        "end_lane_num": end_n,
        "lane_delta": end_n - start_n,
        "change_side": change_side,
        "length": polyline_length(center_pts),
        "self_intersection": self_intersection,
        "added_lane_ids": added_lane_ids,
        "dropped_lane_ids": dropped_lane_ids,
        "active_start_lane_ids": list(entry_lane_ids),
        "active_end_lane_ids": list(exit_lane_ids),
        "accepted_reasons": ["self_intersection"] if self_intersection else [],
        "accepted": not self_intersection,
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
