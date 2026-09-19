# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""General geometry utility functions."""

import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import LinearRing, Polygon


def as_point(point, name: str = "point") -> np.ndarray:
    """Convert input to a validated 2D point array.

    Args:
        point: Array-like with exactly 2 elements.
        name: Field name used in the error message.

    Returns:
        A copy of the input as a ``(2,)`` float64 array.

    Raises:
        ValueError: If the resulting array does not have shape ``(2,)``.
    """
    arr = np.asarray(point, dtype=float)

    if arr.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), got {arr.shape}.")

    return arr.copy()


def heading_unit(heading: float) -> np.ndarray:
    """Return the unit direction vector of a heading angle.

    Args:
        heading: Heading angle in radians, measured counter-clockwise from the
            positive x-axis.

    Returns:
        Unit vector ``[cos(heading), sin(heading)]`` with shape ``(2,)``.
    """
    heading = float(heading)
    return np.array([np.cos(heading), np.sin(heading)], dtype=float)


def normalize_angle(angle: float) -> float:
    """Normalize an angle to the range ``[-pi, pi]``.

    Args:
        angle: Input angle in radians.

    Returns:
        Equivalent angle in ``[-pi, pi]``.
    """
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def euclidean_distance(a, b) -> float:
    """Compute the Euclidean distance between two 2D points.

    Args:
        a: First point as an array-like with at least 2 elements.
        b: Second point as an array-like with at least 2 elements.

    Returns:
        Euclidean distance between the first two coordinates of ``a`` and ``b``.
    """
    return float(np.linalg.norm(np.asarray(a, dtype=float)[:2] - np.asarray(b, dtype=float)[:2]))


def transform_point(point, transform: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homogeneous transform to a 2D point.

    Args:
        point: Array-like point with at least two coordinates.
        transform: Homogeneous transform matrix with shape ``(3, 3)``.

    Returns:
        Transformed 2D point with shape ``(2,)``.
    """

    transformed = np.asarray(transform, dtype=float) @ np.asarray(
        [point[0], point[1], 1.0], dtype=float
    )
    return transformed[:2]


def oriented_box(x: float, y: float, heading: float, length: float, width: float) -> Polygon:
    """Build an oriented rectangular polygon from a centre pose and dimensions.

    Args:
        x: Centre x-coordinate in metres.
        y: Centre y-coordinate in metres.
        heading: Orientation in radians, measured counter-clockwise from the
            positive x-axis.
        length: Box length along the heading direction in metres.  Values below
            ``0.1`` are clamped to ``0.1``.
        width: Box width perpendicular to the heading direction in metres.
            Values below ``0.1`` are clamped to ``0.1``.

    Returns:
        Shapely :class:`~shapely.geometry.Polygon` representing the oriented box.
    """
    length = max(float(length), 0.1)
    width = max(float(width), 0.1)
    bbox = LinearRing(
        [
            [0.5 * length, -0.5 * width],
            [0.5 * length, 0.5 * width],
            [-0.5 * length, 0.5 * width],
            [-0.5 * length, -0.5 * width],
        ]
    )
    transform = [np.cos(heading), -np.sin(heading), np.sin(heading), np.cos(heading), x, y]
    return Polygon(affine_transform(bbox, transform))


def boxes_overlap(box_a, box_b, margin: float = 1.0) -> bool:
    """Check whether two oriented boxes overlap.

    Uses the separating-axis test over the four edge normals, so the answer is
    exact for rectangles: overlapping is reported whenever the two boxes share
    area, including the cross case where no corner of either box lies inside
    the other.

    Both boxes are scaled about their own centres by ``margin`` before testing,
    so ``margin=1.0`` (the default) tests the full-size boxes, while
    ``margin=0.7`` tests a 70% shrink and lets bodies that merely graze pass as
    collision-free. A ``margin`` of 0 collapses both boxes to points, which
    makes every pair overlap.

    Args:
        box_a: First box as ``(x, y, yaw, length, width)``. Units are m and rad.
        box_b: Second box as ``(x, y, yaw, length, width)``.
        margin: Multiplicative scale applied to both boxes before testing.
            Defaults to 1.0, which keeps them at full size.

    Returns:
        True when the scaled boxes overlap.
    """

    f1, l1, hl1, hw1 = _box_axes(box_a[2], box_a[3], box_a[4], margin)
    f2, l2, hl2, hw2 = _box_axes(box_b[2], box_b[3], box_b[4], margin)
    delta = np.array([box_b[0] - box_a[0], box_b[1] - box_a[1]])

    for axis in (f1, l1, f2, l2):
        half1 = hl1 * abs(float(np.dot(f1, axis))) + hw1 * abs(float(np.dot(l1, axis)))
        half2 = hl2 * abs(float(np.dot(f2, axis))) + hw2 * abs(float(np.dot(l2, axis)))
        if abs(float(np.dot(axis, delta))) > half1 + half2:
            return False
    return True


def _box_axes(yaw: float, length: float, width: float, margin: float):
    """Return the two separating-axis bases and half extents of one box."""
    forward = np.array([np.cos(yaw), np.sin(yaw)])
    lateral = np.array([-np.sin(yaw), np.cos(yaw)])
    return forward, lateral, 0.5 * length * margin, 0.5 * width * margin


def central_diff(values: np.ndarray, pad_value: float = np.nan) -> np.ndarray:
    """Return the central difference of a signal along its last axis.

    The difference is ``(f[i+1] - f[i-1]) / 2``, per sample rather than per second.

    Args:
        values: Signal of shape ``(..., steps)``.
        pad_value: Value written to the first and last slot. Defaults to NaN.

    Returns:
        An array shaped like ``values`` holding the central differences.
    """

    values = np.asarray(values, dtype=float)
    differences = (values[..., 2:] - values[..., :-2]) / 2.0
    pad_shape = (*values.shape[:-1], 1)
    pad = np.full(pad_shape, pad_value, dtype=float)
    return np.concatenate([pad, differences, pad], axis=-1)


def central_logical_and(flags: np.ndarray, pad_value: bool = False) -> np.ndarray:
    """Return the ``and`` of each flag and its two neighbours along the last axis.

    Args:
        flags: Boolean array of shape ``(..., steps)``.
        pad_value: Value written to the first and last slot. Defaults to False.

    Returns:
        A boolean array shaped like ``flags``.
    """

    flags = np.asarray(flags, dtype=bool)
    combined = flags[..., 2:] & flags[..., :-2]
    pad_shape = (*flags.shape[:-1], 1)
    pad = np.full(pad_shape, pad_value, dtype=bool)
    return np.concatenate([pad, combined, pad], axis=-1)


def oriented_box_corners(x, y, heading, length, width) -> np.ndarray:
    """Return the four corners of an oriented box, in counter-clockwise order.

    Broadcasts over the inputs.

    Args:
        x: Centre x-coordinate in metres.
        y: Centre y-coordinate in metres.
        heading: Orientation in radians, counter-clockwise from the positive
            x-axis.
        length: Box length along the heading direction in metres.
        width: Box width perpendicular to the heading direction in metres.

    Returns:
        An array of shape ``(..., 4, 2)`` holding the corner coordinates,
        starting at the front-right corner.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    heading = np.asarray(heading, dtype=float)
    half_length = np.asarray(length, dtype=float) / 2.0
    half_width = np.asarray(width, dtype=float) / 2.0
    cos, sin = np.cos(heading), np.sin(heading)

    signs = np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]])
    long_offsets = signs[:, 0] * half_length[..., np.newaxis]
    lat_offsets = signs[:, 1] * half_width[..., np.newaxis]

    corners = np.stack(
        [
            cos[..., np.newaxis] * long_offsets - sin[..., np.newaxis] * lat_offsets,
            sin[..., np.newaxis] * long_offsets + cos[..., np.newaxis] * lat_offsets,
        ],
        axis=-1,
    )
    return corners + np.stack([x, y], axis=-1)[..., np.newaxis, :]


def rounded_box_distance(box_a, box_b, rounding: float = 0.7) -> np.ndarray:
    """Return the signed distance between two corner-rounded boxes.

    Corners are rounded with a radius of ``rounding * min(length, width) / 2``:
    ``0`` gives rectangles and ``1`` gives capsules. Negative when the outlines
    overlap, by the penetration depth.

    Args:
        box_a: Box or boxes as ``(..., 5)`` holding ``(x, y, heading, length,
            width)`` in metres and radians.
        box_b: Second box or boxes, broadcastable against ``box_a``.
        rounding: Corner rounding factor in ``[0, 1]``. Defaults to 0.7.

    Returns:
        An array of the broadcast shape holding the signed distance in metres.
    """

    box_a = np.asarray(box_a, dtype=float)
    box_b = np.asarray(box_b, dtype=float)
    shape = np.broadcast_shapes(box_a.shape[:-1], box_b.shape[:-1])

    corners_a, radius_a = _rounded_box_core(box_a, rounding)
    corners_b, radius_b = _rounded_box_core(box_b, rounding)
    gap = radius_a + radius_b

    separations = np.stack(
        [
            _separating_axis_gaps(corners_a, corners_b, _edge_normals(corners_a)),
            _separating_axis_gaps(corners_a, corners_b, _edge_normals(corners_b)),
        ]
    )
    # A positive gap separates the boxes; otherwise it is the penetration depth.
    widest = np.max(separations, axis=(0, -1))
    centre_distance = np.where(widest > 0.0, _polygon_gap(corners_a, corners_b), widest)
    return np.broadcast_to(centre_distance, shape) - gap


def _rounded_box_core(box: np.ndarray, rounding: float):
    """Return the rectangle a rounded box collapses to, and its corner radius."""
    radius = np.minimum(box[..., 3], box[..., 4]) * rounding / 2.0
    corners = oriented_box_corners(
        box[..., 0],
        box[..., 1],
        box[..., 2],
        box[..., 3] - 2.0 * radius,
        box[..., 4] - 2.0 * radius,
    )
    return corners, radius


def _edge_normals(corners: np.ndarray) -> np.ndarray:
    """Return unit normals of the four edges of each box, shape ``(..., 4, 2)``."""
    edges = np.roll(corners, -1, axis=-2) - corners
    normals = np.stack([edges[..., 1], -edges[..., 0]], axis=-1)
    length = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / np.where(length > 0.0, length, 1.0)


def _separating_axis_gaps(corners_a, corners_b, axes) -> np.ndarray:
    """Return the projection gap of ``b`` behind ``a``, positive when separated."""
    projections_a = np.einsum("...ik,...nk->...in", corners_a, axes)
    projections_b = np.einsum("...ik,...nk->...in", corners_b, axes)
    return np.min(projections_b, axis=-2) - np.max(projections_a, axis=-2)


def _polygon_gap(corners_a: np.ndarray, corners_b: np.ndarray) -> np.ndarray:
    """Return the distance between two disjoint convex quadrilaterals."""
    starts_a = corners_a[..., :, np.newaxis, :]
    ends_a = np.roll(corners_a, -1, axis=-2)[..., :, np.newaxis, :]
    starts_b = corners_b[..., np.newaxis, :, :]
    ends_b = np.roll(corners_b, -1, axis=-2)[..., np.newaxis, :, :]
    return np.min(_segment_distance(starts_a, ends_a, starts_b, ends_b), axis=(-2, -1))


def _segment_distance(p1, q1, p2, q2) -> np.ndarray:
    """Return the distance between two segments, broadcasting over their shapes."""
    d1 = q1 - p1
    d2 = q2 - p2
    offset = p1 - p2
    a = np.sum(d1 * d1, axis=-1)
    e = np.sum(d2 * d2, axis=-1)
    b = np.sum(d1 * d2, axis=-1)
    c = np.sum(d1 * offset, axis=-1)
    f = np.sum(d2 * offset, axis=-1)

    denominator = a * e - b * b
    scale = np.where(denominator > 1e-12 * np.maximum(a * e, 1e-12), 1.0, 0.0)
    s = np.clip(
        np.where(scale > 0, (b * f - c * e) / np.where(scale > 0, denominator, 1.0), 0.0), 0.0, 1.0
    )
    t = np.where(e > 0.0, (b * s + f) / np.where(e > 0.0, e, 1.0), 0.0)

    t_clamped = np.clip(t, 0.0, 1.0)
    s = np.where(t < 0.0, np.clip(-c / np.where(a > 0.0, a, 1.0), 0.0, 1.0), s)
    s = np.where(t > 1.0, np.clip((b - c) / np.where(a > 0.0, a, 1.0), 0.0, 1.0), s)
    t = t_clamped

    return np.linalg.norm((p1 + s[..., np.newaxis] * d1) - (p2 + t[..., np.newaxis] * d2), axis=-1)


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Return the absolute angle in radians between two 2D direction vectors.

    Args:
        v1: First direction vector with shape ``(2,)``.
        v2: Second direction vector with shape ``(2,)``.

    Returns:
        Absolute angle between the two vectors in radians, in ``[0, pi]``.
    """
    dot = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12), -1.0, 1.0)
    return float(np.arccos(dot))


def cubic_hermite_points(
    p0: np.ndarray, t0: np.ndarray, p1: np.ndarray, t1: np.ndarray, num_points: int = 8
) -> np.ndarray:
    """Sample points along a cubic Hermite curve.

    Args:
        p0: Start point with shape ``(2,)``.
        t0: Start tangent vector with shape ``(2,)``.
        p1: End point with shape ``(2,)``.
        t1: End tangent vector with shape ``(2,)``.
        num_points: Number of sample points. Defaults to 8.

    Returns:
        Sampled points with shape ``(num_points, 2)``.
    """
    ts = np.linspace(0, 1, num_points)
    h00 = 2 * ts**3 - 3 * ts**2 + 1
    h10 = ts**3 - 2 * ts**2 + ts
    h01 = -2 * ts**3 + 3 * ts**2
    h11 = ts**3 - ts**2
    return (
        p0[np.newaxis, :] * h00[:, np.newaxis]
        + t0[np.newaxis, :] * h10[:, np.newaxis]
        + p1[np.newaxis, :] * h01[:, np.newaxis]
        + t1[np.newaxis, :] * h11[:, np.newaxis]
    )
