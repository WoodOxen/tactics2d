# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Geometry utilities for map generators."""

from .geometry_utils import offset_polyline, sample_centerline
from .module_geometry import (
    as_point,
    bezier_connection,
    curvature_stats,
    fit_reference_line,
    has_self_intersection,
    nearest_s,
    polyline_length,
    sample_by_s,
)

__all__ = [
    "offset_polyline",
    "sample_centerline",
    "as_point",
    "bezier_connection",
    "curvature_stats",
    "fit_reference_line",
    "has_self_intersection",
    "nearest_s",
    "polyline_length",
    "sample_by_s",
]
