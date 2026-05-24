# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Polyline geometry utilities."""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Point

from .utils import normalize_angle


def _as_polyline(pts: np.ndarray, min_points: int = 2) -> np.ndarray:
    """Convert input points to a valid 2D polyline array."""
    arr = np.asarray(pts, dtype=float)

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"pts must have shape (N, 2), got {arr.shape}.")

    if len(arr) < min_points:
        raise ValueError(f"pts must contain at least {min_points} points.")

    return arr


def offset_polyline(pts: np.ndarray, offset: float) -> np.ndarray:
    """Offset a polyline laterally by a signed distance.

    Positive offset is to the left of the polyline direction. Negative offset
    is to the right.

    Args:
        pts: Input polyline points with shape ``(N, 2)``.
        offset: Signed lateral offset distance in metres.

    Returns:
        Offset polyline points with shape ``(N, 2)``.
    """
    pts = _as_polyline(pts)

    tangents = np.empty_like(pts)
    tangents[1:-1] = pts[2:] - pts[:-2]
    tangents[0] = pts[1] - pts[0]
    tangents[-1] = pts[-1] - pts[-2]

    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    tangents = tangents / norms

    left_normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    return pts + float(offset) * left_normals


def polyline_length(pts: np.ndarray) -> float:
    """Return the arc length of a polyline.

    Args:
        pts: Input polyline points with shape ``(N, 2)``.

    Returns:
        Total polyline length.
    """
    pts = np.asarray(pts, dtype=float)

    if len(pts) < 2:
        return 0.0

    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def cumulative_s(pts: np.ndarray) -> np.ndarray:
    """Return cumulative arc-length coordinates of a polyline.

    Args:
        pts: Input polyline points with shape ``(N, 2)``.

    Returns:
        Cumulative arc-length coordinates with shape ``(N,)``.
    """
    pts = np.asarray(pts, dtype=float)

    if len(pts) == 0:
        return np.empty(0, dtype=float)

    if len(pts) < 2:
        return np.zeros(len(pts), dtype=float)

    seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_lens)])


def nearest_s(polyline: np.ndarray, point: np.ndarray) -> float:
    """Project a point to a polyline and return the arc-length coordinate.

    Args:
        polyline: Input polyline points with shape ``(N, 2)``.
        point: Query point with shape ``(2,)``.

    Returns:
        Arc-length coordinate of the projected point.
    """
    polyline = _as_polyline(polyline)
    point = np.asarray(point, dtype=float)

    if point.shape != (2,):
        raise ValueError(f"point must have shape (2,), got {point.shape}.")

    return float(LineString(polyline).project(Point(float(point[0]), float(point[1]))))


def sample_by_s(
    pts: np.ndarray, s_start: float, s_end: float, n_samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample positions, headings, and right normals by arc length.

    Args:
        pts: Input polyline points with shape ``(N, 2)``.
        s_start: Start arc-length coordinate.
        s_end: End arc-length coordinate.
        n_samples: Number of sampled points.

    Returns:
        A tuple ``(positions, headings, right_normals)``:
            - positions: sampled points with shape ``(n_samples, 2)``.
            - headings: tangent headings with shape ``(n_samples,)``.
            - right_normals: right normal vectors with shape ``(n_samples, 2)``.
    """
    pts = _as_polyline(pts)
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
        if np.linalg.norm(tangent) < 1e-12:
            headings[i] = 0.0
        else:
            headings[i] = float(np.arctan2(tangent[1], tangent[0]))

    right_normals = np.column_stack([np.sin(headings), -np.cos(headings)])
    return positions, headings, right_normals


def polyline_end_pose(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the end point and tangent heading of a polyline.

    Args:
        pts: Input polyline points with shape ``(N, 2)``.

    Returns:
        A tuple ``(end_point, end_heading)``.
    """
    pts = _as_polyline(pts)

    tangent = pts[-1] - pts[-2]
    if np.linalg.norm(tangent) < 1e-12:
        heading = 0.0
    else:
        heading = float(np.arctan2(tangent[1], tangent[0]))

    return pts[-1].copy(), heading


def curvature_stats(pts: np.ndarray) -> dict[str, float]:
    """Estimate maximum absolute curvature and curvature-rate of a polyline.

    Args:
        pts: Input polyline points with shape ``(N, 2)``.

    Returns:
        Dictionary containing ``max_abs_curvature`` and
        ``max_abs_curvature_rate``.
    """
    pts = np.asarray(pts, dtype=float)

    if len(pts) < 4:
        return {"max_abs_curvature": 0.0, "max_abs_curvature_rate": 0.0}

    diffs = np.diff(pts, axis=0)
    ds = np.linalg.norm(diffs, axis=1)

    if int(np.count_nonzero(ds > 1e-9)) < 3:
        return {"max_abs_curvature": 0.0, "max_abs_curvature_rate": 0.0}

    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    dtheta = np.array(
        [normalize_angle(headings[i + 1] - headings[i]) for i in range(len(headings) - 1)]
    )

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
    """Return whether a polyline self-intersects.

    Args:
        pts: Input polyline points with shape ``(N, 2)``.

    Returns:
        Whether the polyline has self-intersection.
    """
    pts = np.asarray(pts, dtype=float)

    if len(pts) < 4:
        return False

    return not LineString(pts).is_simple


def point_at_s(points: np.ndarray, s: float) -> np.ndarray:
    """Sample one point from a polyline by arc-length coordinate.

    Args:
        points: Polyline points with shape ``(N, 2)``.
        s: Arc-length coordinate.

    Returns:
        Sampled point with shape ``(2,)``.
    """
    positions, _, _ = sample_by_s(points, s, s, 2)
    return positions[0].copy()


def point_heading_at_s(points: np.ndarray, s: float) -> tuple[np.ndarray, float]:
    """Sample one point and heading from a polyline by arc-length coordinate.

    Args:
        points: Polyline points with shape ``(N, 2)``.
        s: Arc-length coordinate.

    Returns:
        A tuple ``(point, heading)``.
    """
    positions, headings, _ = sample_by_s(points, s, s, 2)
    return positions[0].copy(), float(headings[0])


def resample_polyline(points: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline to ``n`` points by arc length.

    Args:
        points: Polyline points with shape ``(N, 2)``.
        n: Number of output points.

    Returns:
        Resampled polyline points with shape ``(n, 2)``.
    """
    points = np.asarray(points, dtype=float)

    if len(points) == 0:
        return np.empty((0, 2), dtype=float)
    if len(points) == 1 or n <= 1:
        return points[:1].copy()

    total = polyline_length(points)
    if total < 1e-9:
        return np.repeat(points[:1], int(n), axis=0)

    positions, _, _ = sample_by_s(points, 0.0, total, int(n))
    return positions


def _intersection_points(geometry) -> list[Point]:
    """Extract representative points from a Shapely intersection geometry.

    Args:
        geometry: Shapely geometry returned by an intersection operation.

    Returns:
        Representative intersection points.
    """
    if geometry.is_empty:
        return []

    geom_type = geometry.geom_type

    if geom_type == "Point":
        return [geometry]

    if geom_type == "MultiPoint":
        return [point for point in geometry.geoms]

    if geom_type == "LineString":
        coords = list(geometry.coords)
        if len(coords) == 0:
            return []
        return [Point(coords[0]), Point(coords[-1])]

    if geom_type == "MultiLineString":
        points: list[Point] = []
        for line in geometry.geoms:
            coords = list(line.coords)
            if len(coords) > 0:
                points.append(Point(coords[0]))
                points.append(Point(coords[-1]))
        return points

    if geom_type == "GeometryCollection":
        points: list[Point] = []
        for sub_geometry in geometry.geoms:
            points.extend(_intersection_points(sub_geometry))
        return points

    return []


def find_intersection_point(
    line1: LineString, line2: LineString, *, pick: str = "first_on_line1"
) -> Point | None:
    """Find a representative Point intersection between two LineStrings.

    Args:
        line1: First line.
        line2: Second line.
        pick: Selection rule. One of ``first_on_line1``, ``last_on_line1``,
            ``first_on_line2`` or ``last_on_line2``.

    Returns:
        A representative intersection point, or ``None``.
    """
    points = _intersection_points(line1.intersection(line2))

    if len(points) == 0:
        return None

    if pick == "first_on_line1":
        return min(points, key=lambda pt: float(line1.project(pt)))
    if pick == "last_on_line1":
        return max(points, key=lambda pt: float(line1.project(pt)))
    if pick == "first_on_line2":
        return min(points, key=lambda pt: float(line2.project(pt)))
    if pick == "last_on_line2":
        return max(points, key=lambda pt: float(line2.project(pt)))

    return points[0]


def cut_polyline(line: LineString, pt: Point, keep: str) -> np.ndarray:
    """Cut a polyline at a point and keep one side.

    Args:
        line: Source line.
        pt: Cutting point.
        keep: ``before`` or ``after``.

    Returns:
        Cut polyline points.
    """
    if keep not in ("before", "after"):
        raise ValueError("keep must be 'before' or 'after'.")

    coords = np.array(line.coords, dtype=float)
    if len(coords) == 0:
        return np.empty((0, 2), dtype=float)
    if len(coords) == 1:
        return coords.copy()

    dist = float(line.project(pt))

    dists = np.zeros(len(coords), dtype=float)
    dists[1:] = np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))

    idx = int(np.searchsorted(dists, dist))
    if idx == 0:
        idx = 1

    pt_coords = np.array([[pt.x, pt.y]], dtype=float)

    if keep == "before":
        return np.vstack([coords[:idx], pt_coords])

    return np.vstack([pt_coords, coords[idx:]])
