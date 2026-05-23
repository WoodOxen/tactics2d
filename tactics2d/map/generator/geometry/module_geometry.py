# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Geometry helpers for socket-driven road modules."""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Point

from tactics2d.interpolator import Bezier


def as_point(point: np.ndarray | list[float] | tuple[float, float]) -> np.ndarray:
    """Return a validated 2D point."""
    arr = np.asarray(point, dtype=float)
    if arr.shape != (2,):
        raise ValueError(f"point must have shape (2,), got {arr.shape}.")
    return arr.copy()


def unit(heading: float) -> np.ndarray:
    """Return unit vector for heading."""
    return np.array([np.cos(heading), np.sin(heading)], dtype=float)


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def polyline_length(pts: np.ndarray) -> float:
    """Return arc length of a polyline."""
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def cumulative_s(pts: np.ndarray) -> np.ndarray:
    """Return cumulative arc length of a polyline."""
    if len(pts) < 2:
        return np.zeros(len(pts), dtype=float)
    seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_lens)])


def nearest_s(polyline: np.ndarray, point: np.ndarray) -> float:
    """Project point to polyline and return arc-length coordinate."""
    return float(LineString(polyline).project(Point(float(point[0]), float(point[1]))))


def bezier_connection(
    p0: np.ndarray, h0: float, p3: np.ndarray, h3: float, step_size: float, min_tangent: float = 6.0
) -> np.ndarray:
    """Generate a C1-continuous cubic Bezier curve between two oriented points.

    This is the socket-driven equivalent of OpenDRIVE parametric cubic style.
    It exactly satisfies endpoint position and heading constraints.
    """
    p0 = as_point(p0)
    p3 = as_point(p3)

    chord = float(np.linalg.norm(p3 - p0))
    if chord < 1e-6:
        return np.vstack([p0, p3])

    delta = abs(wrap_angle(h3 - h0))
    tension = 1.0 / 3.0 + (delta / np.pi) * 0.20
    tangent_len = max(min_tangent, chord * tension)

    p1 = p0 + tangent_len * unit(h0)
    p2 = p3 - tangent_len * unit(h3)

    n = max(8, int(chord / max(step_size, 1e-3)) + 1)
    return Bezier.get_curve(np.array([p0, p1, p2, p3]), n, order=3)


def fit_reference_line(
    start_point: np.ndarray,
    start_heading: float,
    end_point: np.ndarray,
    end_heading: float,
    step_size: float,
) -> np.ndarray:
    """Fit a reference line between two RoadPorts.

    Straight line is used when endpoint headings align with the chord.
    Otherwise a cubic Bezier is used to satisfy endpoint tangents.
    """
    p0 = as_point(start_point)
    p1 = as_point(end_point)

    chord = p1 - p0
    length = float(np.linalg.norm(chord))
    if length < 1e-6:
        raise ValueError("start and end points must be distinct.")

    chord_heading = float(np.arctan2(chord[1], chord[0]))
    aligned = (
        abs(wrap_angle(start_heading - chord_heading)) < 1e-3
        and abs(wrap_angle(end_heading - chord_heading)) < 1e-3
    )

    if aligned:
        n = max(2, int(length / max(step_size, 1e-3)) + 1)
        s = np.linspace(0.0, length, n)
        return p0 + np.outer(s, chord / length)

    return bezier_connection(p0, start_heading, p1, end_heading, step_size)


def sample_by_s(
    pts: np.ndarray, s_start: float, s_end: float, n_samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample positions, headings, and right normals by arc length."""
    if len(pts) < 2:
        raise ValueError("pts must contain at least two points.")

    cum = cumulative_s(pts)
    total = cum[-1]

    s_start = float(np.clip(s_start, 0.0, total))
    s_end = float(np.clip(s_end, s_start, total))
    n_samples = max(2, int(n_samples))

    s_query = np.linspace(s_start, s_end, n_samples)
    xs = np.interp(s_query, cum, pts[:, 0])
    ys = np.interp(s_query, cum, pts[:, 1])
    positions = np.column_stack([xs, ys])

    seg_lens = np.diff(cum)
    headings = np.empty(n_samples, dtype=float)

    for i, s in enumerate(s_query):
        idx = int(np.searchsorted(cum, s, side="right") - 1)
        idx = int(np.clip(idx, 0, len(seg_lens) - 1))
        tangent = pts[idx + 1] - pts[idx]
        headings[i] = float(np.arctan2(tangent[1], tangent[0]))

    right_normals = np.column_stack([np.sin(headings), -np.cos(headings)])
    return positions, headings, right_normals


def hermite_cubic_width(s_local: np.ndarray, length: float, target_width: float) -> np.ndarray:
    """Zero-slope lane-width transition.

    w(s) = 3W/L^2 * s^2 - 2W/L^3 * s^3
    """
    length = max(float(length), 1e-9)
    c = 3.0 * target_width / (length * length)
    d = -2.0 * target_width / (length * length * length)
    return c * s_local**2 + d * s_local**3


def curvature_stats(pts: np.ndarray) -> dict[str, float]:
    """Estimate maximum absolute curvature and curvature rate."""
    if len(pts) < 4:
        return {"max_abs_curvature": 0.0, "max_abs_curvature_rate": 0.0}

    diffs = np.diff(pts, axis=0)
    ds = np.linalg.norm(diffs, axis=1)
    if int(np.count_nonzero(ds > 1e-9)) < 3:
        return {"max_abs_curvature": 0.0, "max_abs_curvature_rate": 0.0}

    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    dtheta = np.array([wrap_angle(headings[i + 1] - headings[i]) for i in range(len(headings) - 1)])

    ds_mid = 0.5 * (ds[:-1] + ds[1:])
    ds_mid = np.where(ds_mid < 1e-9, 1e-9, ds_mid)
    curvature = dtheta / ds_mid

    if len(curvature) < 2:
        curvature_rate = np.array([0.0])
    else:
        rate_ds = 0.5 * (ds_mid[:-1] + ds_mid[1:])
        rate_ds = np.where(rate_ds < 1e-9, 1e-9, rate_ds)
        curvature_rate = np.diff(curvature) / rate_ds

    return {
        "max_abs_curvature": float(np.max(np.abs(curvature))),
        "max_abs_curvature_rate": float(np.max(np.abs(curvature_rate))),
    }


def has_self_intersection(pts: np.ndarray) -> bool:
    """Return whether a polyline self-intersects."""
    if len(pts) < 4:
        return False
    return not LineString(pts).is_simple
