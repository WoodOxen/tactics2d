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
