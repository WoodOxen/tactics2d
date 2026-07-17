# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Polyline geometry utilities."""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Point

from .spatial import angle_between, cubic_hermite_points, normalize_angle


def _as_polyline(pts: np.ndarray, min_points: int = 2) -> np.ndarray:
    """Convert input points to a valid 2D polyline array."""
    arr = np.asarray(pts, dtype=float)

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"pts must have shape (N, 2), got {arr.shape}.")

    if len(arr) < min_points:
        raise ValueError(f"pts must contain at least {min_points} points.")

    return arr


def offset(pts: np.ndarray, offset: float) -> np.ndarray:
    """Offset a polyline laterally by a signed distance.

    Positive offset is to the left of the polyline direction; negative offset
    is to the right.

    Args:
        pts: Input polyline points with shape ``(N, 2)``, ``N >= 2``.
        offset: Signed lateral offset distance in metres.

    Returns:
        Offset polyline points with shape ``(N, 2)``.

    Raises:
        ValueError: If ``pts`` does not have shape ``(N, 2)`` or contains
            fewer than 2 points.
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


def arc_length(pts: np.ndarray) -> float:
    """Return the arc length of a polyline.

    Args:
        pts: Input polyline points with shape ``(N, 2)``. Arrays with fewer
            than 2 points return ``0.0`` without raising.

    Returns:
        Total arc length in the same unit as the coordinate values. Returns
        ``0.0`` for empty or single-point inputs.
    """
    pts = np.asarray(pts, dtype=float)

    if len(pts) < 2:
        return 0.0

    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def arc_lengths(pts: np.ndarray) -> np.ndarray:
    """Return cumulative arc-length coordinates of a polyline.

    Args:
        pts: Input polyline points with shape ``(N, 2)``. Empty and
            single-point arrays are handled gracefully.

    Returns:
        Cumulative arc-length array with shape ``(N,)``, starting at ``0.0``.
        Returns an empty array for empty input.
    """
    pts = np.asarray(pts, dtype=float)

    if len(pts) == 0:
        return np.empty(0, dtype=float)

    if len(pts) < 2:
        return np.zeros(len(pts), dtype=float)

    seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_lens)])


def project_arc_length(polyline: np.ndarray, point: np.ndarray) -> float:
    """Project a point onto a polyline and return the arc-length coordinate.

    Args:
        polyline: Input polyline points with shape ``(N, 2)``, ``N >= 2``.
        point: Query point with shape ``(2,)``.

    Returns:
        Arc-length coordinate (``s``) of the closest point on the polyline,
        in the same unit as the coordinate values.

    Raises:
        ValueError: If ``polyline`` does not have shape ``(N, 2)`` or
            contains fewer than 2 points, or if ``point`` shape is not
            ``(2,)``.
    """
    polyline = _as_polyline(polyline)
    point = np.asarray(point, dtype=float)

    if point.shape != (2,):
        raise ValueError(f"point must have shape (2,), got {point.shape}.")

    return float(LineString(polyline).project(Point(float(point[0]), float(point[1]))))


def sample_uniformly(
    pts: np.ndarray, s_start: float, s_end: float, n_samples: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample positions, headings, and right normals uniformly by arc length.

    ``s_start`` and ``s_end`` are clamped to ``[0, total_length]``.
    ``n_samples`` is clamped to at least 2.

    Args:
        pts: Input polyline points with shape ``(N, 2)``, ``N >= 2``.
        s_start: Start arc-length coordinate.
        s_end: End arc-length coordinate. Must be ``>= s_start`` after
            clamping; equal values produce a degenerate single-point
            segment.
        n_samples: Number of output points. Values below 2 are raised to 2.

    Returns:
        A tuple ``(positions, headings, right_normals)``:

        - **positions**: sampled points with shape ``(n_samples, 2)``.
        - **headings**: tangent heading angles in radians with shape
          ``(n_samples,)``.
        - **right_normals**: unit right-normal vectors with shape
          ``(n_samples, 2)``.

    Raises:
        ValueError: If ``pts`` does not have shape ``(N, 2)`` or contains
            fewer than 2 points.
    """
    pts = _as_polyline(pts)
    cumulative_lengths = arc_lengths(pts)
    total = cumulative_lengths[-1]

    s_start = float(np.clip(s_start, 0.0, total))
    s_end = float(np.clip(s_end, s_start, total))
    n_samples = max(2, int(n_samples))

    s_query = np.linspace(s_start, s_end, n_samples)
    xs = np.interp(s_query, cumulative_lengths, pts[:, 0])
    ys = np.interp(s_query, cumulative_lengths, pts[:, 1])
    positions = np.column_stack([xs, ys])

    seg_lens = np.diff(cumulative_lengths)
    headings = np.empty(n_samples, dtype=float)

    for i, s in enumerate(s_query):
        idx = int(np.searchsorted(cumulative_lengths, s, side="right") - 1)
        idx = int(np.clip(idx, 0, len(seg_lens) - 1))

        tangent = pts[idx + 1] - pts[idx]
        if np.linalg.norm(tangent) < 1e-12:
            headings[i] = 0.0
        else:
            headings[i] = float(np.arctan2(tangent[1], tangent[0]))

    right_normals = np.column_stack([np.sin(headings), -np.cos(headings)])
    return positions, headings, right_normals


def end_pose(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the end point and tangent heading of a polyline.

    Args:
        pts: Input polyline points with shape ``(N, 2)``, ``N >= 2``.

    Returns:
        A tuple ``(end_point, end_heading)`` where ``end_point`` has shape
        ``(2,)`` and ``end_heading`` is the tangent angle in radians.

    Raises:
        ValueError: If ``pts`` does not have shape ``(N, 2)`` or contains
            fewer than 2 points.
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

    Uses finite-difference heading changes between consecutive segments.
    Returns zero statistics for polylines with fewer than 4 points or fewer
    than 3 non-degenerate segments.

    Args:
        pts: Input polyline points with shape ``(N, 2)``.

    Returns:
        Dictionary with keys:

        - ``"max_abs_curvature"``: maximum absolute curvature in
          radians/metre.
        - ``"max_abs_curvature_rate"``: maximum absolute rate of change of
          curvature in radians/metre².
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

    Polylines with fewer than 4 points cannot self-intersect and always
    return ``False``.

    Args:
        pts: Input polyline points with shape ``(N, 2)``.

    Returns:
        ``True`` if any two non-adjacent segments cross each other.
    """
    pts = np.asarray(pts, dtype=float)

    if len(pts) < 4:
        return False

    return not LineString(pts).is_simple


def point_at_arc_length(points: np.ndarray, s: float) -> np.ndarray:
    """Sample one point from a polyline by arc-length coordinate.

    Args:
        points: Polyline points with shape ``(N, 2)``, ``N >= 2``.
        s: Arc-length coordinate. Clamped to ``[0, total_length]``.

    Returns:
        Interpolated point at arc-length ``s`` with shape ``(2,)``.

    Raises:
        ValueError: If ``points`` does not have shape ``(N, 2)`` or contains
            fewer than 2 points.
    """
    positions, _, _ = sample_uniformly(points, s, s, 2)
    return positions[0].copy()


def heading_at_arc_length(points: np.ndarray, s: float) -> tuple[np.ndarray, float]:
    """Sample one point and tangent heading from a polyline by arc-length coordinate.

    Args:
        points: Polyline points with shape ``(N, 2)``, ``N >= 2``.
        s: Arc-length coordinate. Clamped to ``[0, total_arc_length]``.

    Returns:
        Tangent heading in radians at arc-length coordinate ``s``.

    Raises:
        ValueError: If ``points`` does not have shape ``(N, 2)`` or contains
            fewer than 2 points.
    """
    _, headings, _ = sample_uniformly(points, s, s, 2)
    return float(headings[0])


def resample(points: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline to exactly ``n`` uniformly spaced points by arc length.

    Args:
        points: Polyline points with shape ``(N, 2)``.
        n: Number of output points. Values ``<= 1`` return the first point
            only.

    Returns:
        Resampled polyline points with shape ``(n, 2)``.  Returns an empty
        array for empty input.
    """
    points = np.asarray(points, dtype=float)

    if len(points) == 0:
        return np.empty((0, 2), dtype=float)
    if len(points) == 1 or n <= 1:
        return points[:1].copy()

    total = arc_length(points)
    if total < 1e-9:
        return np.repeat(points[:1], int(n), axis=0)

    positions, _, _ = sample_uniformly(points, 0.0, total, int(n))
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
    """Find a representative intersection point between two LineStrings.

    Args:
        line1: First Shapely LineString.
        line2: Second Shapely LineString.
        pick: Selection rule controlling which intersection point is
            returned. One of:

            - ``"first_on_line1"``: closest to the start of ``line1``.
            - ``"last_on_line1"``: closest to the end of ``line1``.
            - ``"first_on_line2"``: closest to the start of ``line2``.
            - ``"last_on_line2"``: closest to the end of ``line2``.

            Any other value returns the first intersection found.

    Returns:
        A Shapely :class:`~shapely.geometry.Point` representing the chosen
        intersection, or ``None`` if the lines do not intersect.
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


def cut(line: LineString, pt: Point, keep: str) -> np.ndarray:
    """Cut a Shapely LineString at a point and return one side as a numpy array.

    Args:
        line: Source Shapely :class:`~shapely.geometry.LineString`.
        pt: Cutting point as a Shapely :class:`~shapely.geometry.Point`.
            The nearest projection of ``pt`` onto ``line`` is used as the
            cut location.
        keep: Which side of the cut to return. ``"before"`` returns the
            segment from the start up to the cut; ``"after"`` returns the
            segment from the cut to the end.

    Returns:
        Cut polyline points as an ``(M, 2)`` numpy array. Returns an empty
        ``(0, 2)`` array if ``line`` has no coordinates.

    Raises:
        ValueError: If ``keep`` is not ``"before"`` or ``"after"``.
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


def smooth_joint(
    prev_tail: np.ndarray, next_head: np.ndarray, kink_angle_threshold_deg: float = 15.0
) -> np.ndarray | None:
    """Return interpolated arc points when two segments meet at a kink.

    ``prev_tail`` is the last *k* points of the previous polyline (k >= 2).
    ``next_head`` is the first *k* points of the next polyline (k >= 2).
    Returns ``None`` when the joint is smooth enough to skip.

    Args:
        prev_tail: Last points of the previous polyline with shape ``(K, 2)``.
        next_head: First points of the next polyline with shape ``(K, 2)``.
        kink_angle_threshold_deg: Angle threshold in degrees above which a
            smoothing arc is inserted. Defaults to 15.0.

    Returns:
        Interpolated arc points with shape ``(M, 2)``, or ``None`` if the
        joint is smooth enough.
    """
    p_joint = prev_tail[-1]
    q_joint = next_head[0]

    gap = np.linalg.norm(q_joint - p_joint)
    if gap > 0.5:
        return None

    v_in = p_joint - prev_tail[-2]
    v_out = next_head[1] - q_joint
    norm_in = np.linalg.norm(v_in)
    norm_out = np.linalg.norm(v_out)
    if norm_in < 1e-6 or norm_out < 1e-6:
        return None

    v_in_unit = v_in / norm_in
    v_out_unit = v_out / norm_out

    angle = angle_between(v_in_unit, v_out_unit)
    if angle < np.deg2rad(kink_angle_threshold_deg):
        return None

    arc_len = min(4.0, norm_in, norm_out)
    p_start = p_joint - v_in_unit * arc_len
    p_end = q_joint + v_out_unit * arc_len
    num_points = max(4, int(arc_len / 0.5))
    arc = cubic_hermite_points(
        p_start, v_in_unit * arc_len, p_end, v_out_unit * arc_len, num_points
    )

    mask = (np.linalg.norm(arc - p_joint, axis=1) >= 0.01) & (
        np.linalg.norm(arc - q_joint, axis=1) >= 0.01
    )
    interior = arc[mask]
    return interior if len(interior) >= 2 else None


def concatenate(
    polylines: list[np.ndarray], kink_angle_threshold_deg: float = 15.0
) -> np.ndarray | None:
    """Concatenate multiple polylines with C1-smooth joints.

    When two consecutive polylines meet at a kink (angle > threshold),
    transitional Hermite arc points are inserted to produce a smooth joint.

    Args:
        polylines: Sequence of polyline point arrays, each ``(N_i, 2)``.
        kink_angle_threshold_deg: Angle threshold in degrees. Defaults to 15.0.

    Returns:
        Concatenated polyline points with shape ``(M, 2)``, or ``None`` if
        no valid polylines are provided.
    """
    merged: list[np.ndarray] = []
    for polyline in polylines:
        if polyline is None or len(polyline) == 0:
            continue
        if not merged:
            merged.append(polyline.copy())
            continue

        previous = merged[-1]
        if np.allclose(previous[-1], polyline[0]):
            n_tail = min(3, len(previous))
            n_head = min(3, len(polyline))
            arc = smooth_joint(previous[-n_tail:], polyline[:n_head], kink_angle_threshold_deg)
            if arc is not None:
                merged.append(arc)
            merged.append(polyline[1:].copy())
        else:
            merged.append(polyline.copy())

    if not merged:
        return None

    return np.vstack(merged)
