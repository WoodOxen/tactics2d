# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Rule definitions and shared types for map generators."""

from .lane_marking_rules import (
    MarkingSpec,
    get_standard,
    one_way_mark,
    one_way_mark_kwargs,
    ramp_mark,
    ramp_mark_kwargs,
    roadline_render_kwargs,
    set_standard,
    two_way_backward_kwargs,
    two_way_centerline_kwargs,
    two_way_forward_kwargs,
)
from .module_types import RoadModuleResult, RoadPort, make_port, ports_to_interfaces

__all__ = [
    "RoadPort",
    "RoadModuleResult",
    "make_port",
    "ports_to_interfaces",
    "MarkingSpec",
    "set_standard",
    "get_standard",
    "roadline_render_kwargs",
    "one_way_mark",
    "ramp_mark",
    "one_way_mark_kwargs",
    "ramp_mark_kwargs",
    "two_way_centerline_kwargs",
    "two_way_forward_kwargs",
    "two_way_backward_kwargs",
]
