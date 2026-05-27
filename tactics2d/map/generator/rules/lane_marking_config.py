# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Lane marking configuration data: types, token specs, and standard rule tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Standard = Literal["MUTCD", "GB"]

Pattern = Literal[
    "solid",
    "dashed",
    "dotted",
    "double_solid",
    "double_dashed",
    "solid_dashed",
    "dashed_solid",
    "virtual",
]

Color = Literal["white", "yellow", "none"]

Role = Literal[
    "centerline",
    "lane_divider",
    "edge",
    "ramp_aux",
    "ramp_edge",
    "intersection_connection",
    "roundabout_ring",
    "virtual",
]


@dataclass(frozen=True)
class MarkingSpec:
    """Semantic description of one road marking token.

    Attributes:
        token: Unique string identifier for this marking.
        color: Rendered color (``"white"``, ``"yellow"``, or ``"none"``).
        pattern: Line pattern (``"solid"``, ``"dashed"``, etc.).
        lane_change: ``(allow_left, allow_right)`` lane-change permissions.
        role: Semantic road role (``"centerline"``, ``"edge"``, etc.).
        opendrive_type: OpenDRIVE roadMark ``type`` attribute string.
        opendrive_color: OpenDRIVE roadMark ``color`` attribute string.
        sumo_change_left: SUMO left lane-change permission flag.
        sumo_change_right: SUMO right lane-change permission flag.
        render_style: Renderer-compatible style string.
        render_width: Rendered line width in metres.
    """

    token: str
    color: Color
    pattern: Pattern
    lane_change: tuple[bool, bool]
    role: Role
    opendrive_type: str
    opendrive_color: str
    sumo_change_left: bool
    sumo_change_right: bool
    render_style: str
    render_width: float = 0.5


MARKING_SPECS: dict[str, MarkingSpec] = {
    "virtual": MarkingSpec(
        token="virtual",
        color="none",
        pattern="virtual",
        lane_change=(True, True),
        role="virtual",
        opendrive_type="none",
        opendrive_color="standard",
        sumo_change_left=True,
        sumo_change_right=True,
        render_style="solid",
        render_width=0.0,
    ),
    "solid": MarkingSpec(
        token="solid",
        color="white",
        pattern="solid",
        lane_change=(False, False),
        role="lane_divider",
        opendrive_type="solid",
        opendrive_color="white",
        sumo_change_left=False,
        sumo_change_right=False,
        render_style="solid",
    ),
    "dashed": MarkingSpec(
        token="dashed",
        color="white",
        pattern="dashed",
        lane_change=(True, True),
        role="lane_divider",
        opendrive_type="broken",
        opendrive_color="white",
        sumo_change_left=True,
        sumo_change_right=True,
        render_style="dashed",
    ),
    "solid_dashed": MarkingSpec(
        token="solid_dashed",
        color="white",
        pattern="solid_dashed",
        lane_change=(False, True),
        role="lane_divider",
        opendrive_type="solid broken",
        opendrive_color="white",
        sumo_change_left=False,
        sumo_change_right=True,
        render_style="solid_dashed",
    ),
    "dashed_solid": MarkingSpec(
        token="dashed_solid",
        color="white",
        pattern="dashed_solid",
        lane_change=(True, False),
        role="lane_divider",
        opendrive_type="broken solid",
        opendrive_color="white",
        sumo_change_left=True,
        sumo_change_right=False,
        render_style="dashed_solid",
    ),
    "solid_white": MarkingSpec(
        token="solid_white",
        color="white",
        pattern="solid",
        lane_change=(False, False),
        role="lane_divider",
        opendrive_type="solid",
        opendrive_color="white",
        sumo_change_left=False,
        sumo_change_right=False,
        render_style="solid",
    ),
    "dashed_white": MarkingSpec(
        token="dashed_white",
        color="white",
        pattern="dashed",
        lane_change=(True, True),
        role="lane_divider",
        opendrive_type="broken",
        opendrive_color="white",
        sumo_change_left=True,
        sumo_change_right=True,
        render_style="dashed",
    ),
    "dotted_white": MarkingSpec(
        token="dotted_white",
        color="white",
        pattern="dotted",
        lane_change=(True, True),
        role="intersection_connection",
        opendrive_type="broken",
        opendrive_color="white",
        sumo_change_left=True,
        sumo_change_right=True,
        render_style="dashed",
    ),
    "solid_white_edge": MarkingSpec(
        token="solid_white_edge",
        color="white",
        pattern="solid",
        lane_change=(False, False),
        role="edge",
        opendrive_type="solid",
        opendrive_color="white",
        sumo_change_left=False,
        sumo_change_right=False,
        render_style="solid",
    ),
    "dashed_white_edge": MarkingSpec(
        token="dashed_white_edge",
        color="white",
        pattern="dashed",
        lane_change=(True, True),
        role="edge",
        opendrive_type="broken",
        opendrive_color="white",
        sumo_change_left=True,
        sumo_change_right=True,
        render_style="dashed",
    ),
    "solid_white_ramp": MarkingSpec(
        token="solid_white_ramp",
        color="white",
        pattern="solid",
        lane_change=(False, False),
        role="ramp_edge",
        opendrive_type="solid",
        opendrive_color="white",
        sumo_change_left=False,
        sumo_change_right=False,
        render_style="solid",
    ),
    "dashed_white_ramp": MarkingSpec(
        token="dashed_white_ramp",
        color="white",
        pattern="dashed",
        lane_change=(True, True),
        role="ramp_aux",
        opendrive_type="broken",
        opendrive_color="white",
        sumo_change_left=True,
        sumo_change_right=True,
        render_style="dashed",
    ),
    "solid_yellow": MarkingSpec(
        token="solid_yellow",
        color="yellow",
        pattern="solid",
        lane_change=(False, False),
        role="centerline",
        opendrive_type="solid",
        opendrive_color="yellow",
        sumo_change_left=False,
        sumo_change_right=False,
        render_style="solid",
    ),
    "solid_yellow_edge": MarkingSpec(
        token="solid_yellow_edge",
        color="yellow",
        pattern="solid",
        lane_change=(False, False),
        role="edge",
        opendrive_type="solid",
        opendrive_color="yellow",
        sumo_change_left=False,
        sumo_change_right=False,
        render_style="solid",
    ),
    "dashed_yellow": MarkingSpec(
        token="dashed_yellow",
        color="yellow",
        pattern="dashed",
        lane_change=(True, True),
        role="centerline",
        opendrive_type="broken",
        opendrive_color="yellow",
        sumo_change_left=True,
        sumo_change_right=True,
        render_style="dashed",
    ),
    "solid_double_yellow": MarkingSpec(
        token="solid_double_yellow",
        color="yellow",
        pattern="double_solid",
        lane_change=(False, False),
        role="centerline",
        opendrive_type="solid solid",
        opendrive_color="yellow",
        sumo_change_left=False,
        sumo_change_right=False,
        render_style="solid",
    ),
    "dashed_double_yellow": MarkingSpec(
        token="dashed_double_yellow",
        color="yellow",
        pattern="double_dashed",
        lane_change=(True, True),
        role="centerline",
        opendrive_type="broken broken",
        opendrive_color="yellow",
        sumo_change_left=True,
        sumo_change_right=True,
        render_style="dashed",
    ),
    "solid_dashed_yellow": MarkingSpec(
        token="solid_dashed_yellow",
        color="yellow",
        pattern="solid_dashed",
        lane_change=(False, True),
        role="centerline",
        opendrive_type="solid broken",
        opendrive_color="yellow",
        sumo_change_left=False,
        sumo_change_right=True,
        render_style="solid_dashed",
    ),
    "dashed_solid_yellow": MarkingSpec(
        token="dashed_solid_yellow",
        color="yellow",
        pattern="dashed_solid",
        lane_change=(True, False),
        role="centerline",
        opendrive_type="broken solid",
        opendrive_color="yellow",
        sumo_change_left=True,
        sumo_change_right=False,
        render_style="dashed_solid",
    ),
}

MUTCD_RULES: dict[str, str] = {
    "centerline_2lane_passing": "dashed_yellow",
    "centerline_2lane_no_passing": "solid_double_yellow",
    "centerline_4lane_plus": "solid_double_yellow",
    "tw_forward_interior": "dashed_white",
    "tw_forward_outer": "solid_white_edge",
    "tw_backward_interior": "dashed_white",
    "tw_backward_outer": "solid_white_edge",
    "ow_left_edge": "solid_yellow_edge",
    "ow_right_edge": "solid_white_edge",
    "ow_interior": "dashed_white",
    "ramp_aux_left": "dashed_white_ramp",
    "ramp_aux_right": "solid_white_ramp",
    "ramp_spiral_edge": "solid_white_ramp",
    "ramp_departure_edge": "solid_white_ramp",
    "ramp_approach_edge": "solid_white_ramp",
    "ramp_gore": "solid_white_ramp",
    "ramp_interior": "dashed_white_ramp",
    "ramp_left_edge": "solid_yellow_edge",
    "ramp_right_edge": "solid_white_ramp",
    "int_connection": "dotted_white",
    "rab_ring_outer": "solid_white_edge",
    "rab_ring_inner": "solid_yellow",
    "rab_ring_interior": "dashed_white",
    "rab_connection": "dotted_white",
}

GB_RULES: dict[str, str] = {
    "centerline_2lane_passing": "dashed_yellow",
    "centerline_2lane_no_passing": "solid_double_yellow",
    "centerline_4lane_plus": "solid_double_yellow",
    "tw_forward_interior": "dashed_white",
    "tw_forward_outer": "solid_white_edge",
    "tw_backward_interior": "dashed_white",
    "tw_backward_outer": "solid_white_edge",
    "ow_left_edge": "solid_white_edge",
    "ow_right_edge": "solid_white_edge",
    "ow_interior": "dashed_white",
    "ramp_aux_left": "dashed_white_ramp",
    "ramp_aux_right": "solid_white_ramp",
    "ramp_spiral_edge": "solid_white_ramp",
    "ramp_departure_edge": "solid_white_ramp",
    "ramp_approach_edge": "solid_white_ramp",
    "ramp_gore": "solid_white_ramp",
    "ramp_interior": "dashed_white_ramp",
    "ramp_left_edge": "solid_white_edge",
    "ramp_right_edge": "solid_white_ramp",
    "int_connection": "dotted_white",
    "rab_ring_outer": "solid_white_edge",
    "rab_ring_inner": "solid_white_edge",
    "rab_ring_interior": "dashed_white",
    "rab_connection": "dotted_white",
}

STANDARD_TABLES: dict[str, dict[str, str]] = {"MUTCD": MUTCD_RULES, "GB": GB_RULES}
