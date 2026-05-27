# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""General geometry utility functions."""

import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import LinearRing, Polygon


def as_point(point, name: str = "point") -> np.ndarray:
    """Convert input to a validated 2D point array."""
    arr = np.asarray(point, dtype=float)

    if arr.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), got {arr.shape}.")

    return arr.copy()


def heading_unit(heading: float) -> np.ndarray:
    """Return the unit direction vector of a heading angle."""
    heading = float(heading)
    return np.array([np.cos(heading), np.sin(heading)], dtype=float)


def normalize_angle(angle: float) -> float:
    """Normalize an angle to the range [-pi, pi]."""

    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def euclidean_distance(a, b) -> float:
    """Compute the Euclidean distance between two 2D points."""

    return float(np.linalg.norm(np.asarray(a, dtype=float)[:2] - np.asarray(b, dtype=float)[:2]))


def oriented_box(x: float, y: float, heading: float, length: float, width: float) -> Polygon:
    """Build an oriented rectangular polygon from center pose and dimensions."""

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
