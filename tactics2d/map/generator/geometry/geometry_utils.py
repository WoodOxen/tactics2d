# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Geometry utilities for road element generators."""

from __future__ import annotations

import numpy as np

from tactics2d.geometry import Circle


def sample_centerline(
    start: np.ndarray, heading: float, length: float, curvature: float = 0.0, step_size: float = 0.1
) -> np.ndarray:
    """Sample points along a road centre line.

    Args:
        start: Start point ``(x, y)`` in metres.
        heading: Start heading in radians.
        length: Arc length in metres.
        curvature: Signed curvature in m^-1. Positive curves left.
        step_size: Sampling interval in metres.

    Returns:
        Sampled centre-line points, shape ``(N, 2)``.
    """
    start = np.asarray(start, dtype=float)

    if abs(curvature) < 1e-9:
        n = max(2, int(length / step_size) + 1)
        s = np.linspace(0.0, length, n)
        tangent = np.array([np.cos(heading), np.sin(heading)])
        return start + np.outer(s, tangent)

    radius = abs(1.0 / curvature)
    side = "L" if curvature > 0 else "R"
    center, _ = Circle.get_circle(
        tangent_point=start, tangent_heading=heading, radius=radius, side=side
    )

    start_angle = np.arctan2(start[1] - center[1], start[0] - center[0])
    delta_angle = length / radius

    return Circle.get_arc(
        center_point=center,
        radius=radius,
        delta_angle=delta_angle,
        start_angle=start_angle,
        clockwise=(curvature < 0),
        step_size=step_size,
    )


def offset_polyline(pts: np.ndarray, offset: float) -> np.ndarray:
    """Offset a polyline laterally by a signed distance.

    Args:
        pts: Input polyline points, shape ``(N, 2)``.
        offset: Signed lateral offset in metres. Positive is left.

    Returns:
        Offset polyline points, shape ``(N, 2)``.
    """
    pts = np.asarray(pts, dtype=float)

    if len(pts) < 2:
        raise ValueError("pts must contain at least two points.")

    tangents = np.empty_like(pts)
    tangents[1:-1] = pts[2:] - pts[:-2]
    tangents[0] = pts[1] - pts[0]
    tangents[-1] = pts[-1] - pts[-2]

    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    tangents = tangents / norms

    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    return pts + offset * normals


def get_exit(pts: np.ndarray, heading: float, curvature: float) -> tuple[np.ndarray, float]:
    """Return exit point and heading of a sampled centre line."""
    pts = np.asarray(pts, dtype=float)

    if len(pts) < 2:
        raise ValueError("pts must contain at least two points.")

    if abs(curvature) < 1e-9:
        return pts[-1], float(heading)

    tangent = pts[-1] - pts[-2]
    return pts[-1], float(np.arctan2(tangent[1], tangent[0]))


def arc_pts(
    start_pt: np.ndarray,
    start_heading: float,
    angle: float,
    radius: float,
    clockwise: bool,
    step_size: float = 0.1,
) -> tuple[np.ndarray, float]:
    """Sample points along a circular arc."""
    start_pt = np.asarray(start_pt, dtype=float)

    sign = -1.0 if clockwise else 1.0
    cx = start_pt[0] - sign * radius * np.sin(start_heading)
    cy = start_pt[1] + sign * radius * np.cos(start_heading)
    center = np.array([cx, cy])

    n = max(2, int(radius * angle / step_size))
    thetas = np.linspace(0.0, sign * angle, n)

    start_theta = np.arctan2(start_pt[1] - cy, start_pt[0] - cx)
    pts = center + radius * np.column_stack(
        [np.cos(start_theta + thetas), np.sin(start_theta + thetas)]
    )

    return pts, float(start_heading + sign * angle)
