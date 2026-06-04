# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Geometry module."""


from .circle import Circle
from .direction import CardinalDirection, RelativeDirection
from .frenet import FrenetPoint, ReferencePath
from .polyline import (
    cumulative_s,
    curvature_stats,
    cut_polyline,
    find_intersection_point,
    has_self_intersection,
    nearest_s,
    offset_polyline,
    point_at_s,
    point_heading_at_s,
    polyline_end_pose,
    polyline_length,
    resample_polyline,
    sample_by_s,
)
from .utils import as_point, euclidean_distance, heading_unit, normalize_angle, oriented_box

__all__ = [
    "cut_polyline",
    "find_intersection_point",
    "resample_polyline",
    "point_heading_at_s",
    "point_at_s",
    "Circle",
    "RelativeDirection",
    "CardinalDirection",
    "FrenetPoint",
    "ReferencePath",
    "offset_polyline",
    "polyline_length",
    "cumulative_s",
    "nearest_s",
    "sample_by_s",
    "polyline_end_pose",
    "curvature_stats",
    "has_self_intersection",
    "euclidean_distance",
    "normalize_angle",
    "oriented_box",
    "as_point",
    "heading_unit",
]
