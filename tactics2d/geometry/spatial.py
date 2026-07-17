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
