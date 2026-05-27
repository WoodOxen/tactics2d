# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Rule definitions and shared types for road segment generators."""

from .lane_marking_config import GB_RULES, MARKING_SPECS, MUTCD_RULES, STANDARD_TABLES, MarkingSpec
from .lane_marking_rules import (
    get_standard,
    one_way_boundary_token,
    one_way_mark,
    ramp_mark,
    roadline_render_kwargs,
    set_standard,
    two_way_backward,
    two_way_centerline,
    two_way_forward,
)
from .module_types import MainRoadType, RampKind, RampSide, RoadModuleResult, RoadPort, make_port

__all__ = [
    "MarkingSpec",
    "MARKING_SPECS",
    "MUTCD_RULES",
    "GB_RULES",
    "STANDARD_TABLES",
    "MainRoadType",
    "RampKind",
    "RampSide",
    "RoadModuleResult",
    "RoadPort",
    "make_port",
    "get_standard",
    "set_standard",
    "roadline_render_kwargs",
    "one_way_boundary_token",
    "one_way_mark",
    "ramp_mark",
    "two_way_backward",
    "two_way_centerline",
    "two_way_forward",
]
