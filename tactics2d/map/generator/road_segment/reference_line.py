# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Reference-line and width-profile helpers for road segment generators."""

from __future__ import annotations

import numpy as np

from tactics2d.geometry import as_point, heading_unit, normalize_angle
from tactics2d.geometry.polyline import polyline_end_pose
from tactics2d.interpolator import Bezier
from tactics2d.interpolator.spiral import Spiral


def interpolation_count_from_step(length: float, step_size: float) -> int:
    """Return interpolation count from arc length and sampling step.

    The returned count includes both start and end points.

    Args:
        length: Arc length in metres.
        step_size: Desired sampling interval in metres.

    Returns:
        Number of interpolation points.
    """
    length = float(length)
    step_size = float(step_size)

    if length < 0.0:
        raise ValueError("length must be non-negative.")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive.")

    return max(2, int(np.ceil(length / step_size)) + 1)


def sample_centerline(
    start: np.ndarray, heading: float, length: float, curvature: float = 0.0, step_size: float = 0.1
) -> np.ndarray:
    """Sample a road centre line.

    This helper exposes a generator-friendly interface and delegates curve
    sampling to ``Spiral.get_curve``. With ``gamma == 0``, the spiral
    interpolator covers both straight lines and circular arcs.

    Args:
        start: Start point ``(x, y)`` in metres.
        heading: Start heading in radians.
        length: Arc length in metres.
        curvature: Signed curvature in ``m^-1``. Positive values turn left.
        step_size: Desired sampling interval in metres.

    Returns:
        Sampled centre-line points with shape ``(N, 2)``.
    """
    start = as_point(start, "start")
    n_interpolation = interpolation_count_from_step(length, step_size)

    return Spiral.get_curve(
        length=float(length),
        start_point=start,
        heading=float(heading),
        start_curvature=float(curvature),
        gamma=0.0,
        n_interpolation=n_interpolation,
    )


def get_exit(pts: np.ndarray, heading: float, curvature: float) -> tuple[np.ndarray, float]:
    """Return exit point and heading of a sampled centre line.

    Args:
        pts: Sampled centre-line points with shape ``(N, 2)``.
        heading: Input heading in radians.
        curvature: Centre-line curvature.

    Returns:
        A tuple ``(exit_point, exit_heading)``.
    """
    pts = np.asarray(pts, dtype=float)

    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"pts must have shape (N, 2), got {pts.shape}.")
    if len(pts) < 2:
        raise ValueError("pts must contain at least two points.")

    if abs(float(curvature)) < 1e-9:
        return pts[-1].copy(), float(heading)

    return polyline_end_pose(pts)


def arc_pts(
    start_pt: np.ndarray,
    start_heading: float,
    angle: float,
    radius: float,
    clockwise: bool,
    step_size: float = 0.1,
) -> tuple[np.ndarray, float]:
    """Sample points along a circular arc.

    This helper exposes an arc-specific generator interface and delegates the
    actual sampling to ``Spiral.get_curve``.

    Args:
        start_pt: Start point ``(x, y)`` in metres.
        start_heading: Start heading in radians.
        angle: Non-negative arc angle in radians.
        radius: Positive arc radius in metres.
        clockwise: Whether the arc turns clockwise.
        step_size: Desired sampling interval in metres.

    Returns:
        A tuple ``(points, end_heading)``.
    """
    start_pt = as_point(start_pt, "start_pt")
    angle = float(angle)
    radius = float(radius)

    if angle < 0.0:
        raise ValueError("angle must be non-negative.")
    if radius <= 0.0:
        raise ValueError("radius must be positive.")

    sign = -1.0 if clockwise else 1.0
    length = radius * angle
    curvature = sign / radius
    n_interpolation = interpolation_count_from_step(length, step_size)

    pts = Spiral.get_curve(
        length=length,
        start_point=start_pt,
        heading=float(start_heading),
        start_curvature=curvature,
        gamma=0.0,
        n_interpolation=n_interpolation,
    )

    return pts, float(start_heading + sign * angle)


def bezier_connection(
    p0: np.ndarray, h0: float, p3: np.ndarray, h3: float, step_size: float, min_tangent: float = 6.0
) -> np.ndarray:
    """Generate a C1-continuous cubic Bezier curve between two oriented points.

    This helper is generator-specific: it converts endpoint poses into cubic
    Bezier control points, then delegates interpolation to ``Bezier.get_curve``.

    Args:
        p0: Start point ``(x, y)``.
        h0: Start heading in radians.
        p3: End point ``(x, y)``.
        h3: End heading in radians.
        step_size: Desired sampling interval in metres.
        min_tangent: Minimum tangent handle length.

    Returns:
        Sampled Bezier curve points with shape ``(N, 2)``.
    """
    p0 = as_point(p0, "p0")
    p3 = as_point(p3, "p3")
    h0 = float(h0)
    h3 = float(h3)
    step_size = float(step_size)
    min_tangent = float(min_tangent)

    if step_size <= 0.0:
        raise ValueError("step_size must be positive.")
    if min_tangent < 0.0:
        raise ValueError("min_tangent must be non-negative.")

    chord = float(np.linalg.norm(p3 - p0))
    if chord < 1e-6:
        return np.vstack([p0, p3])

    delta_heading = abs(normalize_angle(h3 - h0))
    tension = 1.0 / 3.0 + (delta_heading / np.pi) * 0.20
    tangent_len = max(min_tangent, chord * tension)

    p1 = p0 + tangent_len * heading_unit(h0)
    p2 = p3 - tangent_len * heading_unit(h3)

    n_interpolation = max(8, int(np.ceil(chord / step_size)) + 1)

    return Bezier.get_curve(
        control_points=np.array([p0, p1, p2, p3]), n_interpolation=n_interpolation, order=3
    )


def fit_reference_line(
    start_point: np.ndarray,
    start_heading: float,
    end_point: np.ndarray,
    end_heading: float,
    step_size: float,
) -> np.ndarray:
    """Fit a reference line between two oriented road-module ports.

    If both endpoint headings align with the chord direction, a straight line is
    used. Otherwise a cubic Bezier connection is generated.

    Args:
        start_point: Start point ``(x, y)``.
        start_heading: Start heading in radians.
        end_point: End point ``(x, y)``.
        end_heading: End heading in radians.
        step_size: Desired sampling interval in metres.

    Returns:
        Reference-line points with shape ``(N, 2)``.
    """
    p0 = as_point(start_point, "start_point")
    p1 = as_point(end_point, "end_point")
    start_heading = float(start_heading)
    end_heading = float(end_heading)
    step_size = float(step_size)

    if step_size <= 0.0:
        raise ValueError("step_size must be positive.")

    chord = p1 - p0
    length = float(np.linalg.norm(chord))
    if length < 1e-6:
        raise ValueError("start and end points must be distinct.")

    chord_heading = float(np.arctan2(chord[1], chord[0]))
    aligned = (
        abs(normalize_angle(start_heading - chord_heading)) < 1e-3
        and abs(normalize_angle(end_heading - chord_heading)) < 1e-3
    )

    if aligned:
        n_interpolation = interpolation_count_from_step(length, step_size)
        s = np.linspace(0.0, length, n_interpolation)
        return p0 + np.outer(s, chord / length)

    return bezier_connection(p0=p0, h0=start_heading, p3=p1, h3=end_heading, step_size=step_size)


def hermite_cubic_width(s_local: np.ndarray, length: float, target_width: float) -> np.ndarray:
    """Return a zero-slope cubic width transition (smoothstep Hermite cubic).

    Args:
        s_local: Local arc-length coordinates.
        length: Transition length. Must be positive.
        target_width: Target width or width delta.

    Returns:
        Width values with the same shape as ``s_local``.
    """
    length = float(length)
    target_width = float(target_width)

    if length <= 0.0:
        raise ValueError("length must be positive.")

    s_local = np.asarray(s_local, dtype=float)
    t = np.clip(s_local / length, 0.0, 1.0)

    return target_width * (3.0 * t**2 - 2.0 * t**3)
