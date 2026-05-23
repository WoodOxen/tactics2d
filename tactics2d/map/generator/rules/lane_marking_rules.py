# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Lane marking rules for road element generators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Standard = Literal["MUTCD", "GB"]

_STANDARD: Standard = "MUTCD"


def set_standard(standard: Standard) -> None:
    """Set the active lane marking standard globally."""
    global _STANDARD

    if standard not in ("MUTCD", "GB"):
        raise ValueError(f"Unknown marking standard '{standard}'. Use 'MUTCD' or 'GB'.")

    _STANDARD = standard


def get_standard() -> str:
    """Return the active lane marking standard."""
    return _STANDARD


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
    """Semantic description of one road marking token."""

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


_MARKINGS: dict[str, MarkingSpec] = {
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


def get_marking_spec(token: str | None) -> MarkingSpec:
    """Return the semantic marking specification for a token."""
    if token is None:
        return _MARKINGS["solid"]

    return _MARKINGS.get(token, _MARKINGS["solid"])


def roadline_render_color(token: str | None) -> str:
    """Return renderer color key for a roadline token."""
    spec = get_marking_spec(token)

    if spec.color == "yellow":
        return "yellow"
    if spec.color == "white":
        return "white"

    return "white"


def roadline_render_style(token: str | None) -> str:
    """Return renderer-compatible line style for a roadline token."""
    return get_marking_spec(token).render_style


def roadline_render_subtype(token: str | None) -> str:
    """Return renderer-compatible RoadLine subtype."""
    spec = get_marking_spec(token)

    if spec.role == "virtual":
        return "virtual"

    if spec.pattern in ("dashed", "double_dashed", "dotted"):
        return "dashed"

    if "dashed" in spec.pattern:
        return "dashed"

    return "solid"


def roadline_render_type(token: str | None) -> str:
    """Return renderer-compatible RoadLine type."""
    spec = get_marking_spec(token)

    if spec.role == "virtual":
        return "virtual"

    if spec.role in ("edge", "ramp_edge"):
        return "road_border"

    return "line_thin"


def roadline_render_kwargs(
    token: str | None, custom_tags: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return RoadLine kwargs for a semantic marking token."""
    spec = get_marking_spec(token)

    tags = {
        "marking_token": spec.token,
        "marking_role": spec.role,
        "marking_color": spec.color,
        "marking_pattern": spec.pattern,
        "marking_standard": get_standard(),
        "opendrive_type": spec.opendrive_type,
        "opendrive_color": spec.opendrive_color,
    }

    if custom_tags is not None:
        tags.update(custom_tags)

    return {
        "type_": roadline_render_type(token),
        "subtype": roadline_render_subtype(token),
        "color": roadline_render_color(token),
        "custom_tags": tags,
    }


def roadline_lane_change(token: str | None) -> tuple[bool, bool]:
    """Return lane-change permission for a roadline token."""
    return get_marking_spec(token).lane_change


def is_crossable(token: str | None) -> bool:
    """Return whether a marking is crossable from both sides."""
    return roadline_lane_change(token) == (True, True)


def is_solid(token: str | None) -> bool:
    """Return whether the marking has a solid component."""
    return "solid" in get_marking_spec(token).pattern


def is_dashed(token: str | None) -> bool:
    """Return whether the marking has a dashed or broken component."""
    pattern = get_marking_spec(token).pattern
    return "dashed" in pattern or pattern == "dotted"


def get_all_marking_specs() -> dict[str, MarkingSpec]:
    """Return all registered marking specifications."""
    return dict(_MARKINGS)


def _get_mutcd_rules() -> dict[str, str]:
    """Return lane marking mapping rules for the US MUTCD standard.

    Core principles:
    - Yellow lines separate opposing traffic or mark the left edge of one-way
      roads, divided highways, and ramps (solid_yellow_edge, role='edge').
    - White lines separate same-direction traffic or mark the right edge.
    - Gore areas use solid white ramp markings.
    - Intersection and roundabout connections use dotted white lines.
    """
    return {
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


def _get_gb_rules() -> dict[str, str]:
    """Return lane marking mapping rules for the Chinese GB 5768 standard.

    Differs from MUTCD in that one-way left edges and ramp left edges use
    solid white rather than solid yellow.
    """
    return {
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


_MUTCD: dict[str, str] = _get_mutcd_rules()

_GB: dict[str, str] = _get_gb_rules()

_TABLES: dict[str, dict[str, str]] = {"MUTCD": _MUTCD, "GB": _GB}


def _table(standard: str | None) -> dict[str, str]:
    """Return rule table for a standard."""
    return _TABLES.get(standard or _STANDARD, _MUTCD)


def two_way_centerline(
    forward_lane_num: int,
    backward_lane_num: int,
    *,
    no_passing: bool | None = None,
    standard: str | None = None,
) -> str:
    """Subtype for the centre line of a two-way road."""
    total = forward_lane_num + backward_lane_num
    t = _table(standard)

    if total >= 4:
        return t["centerline_4lane_plus"]

    if no_passing is None:
        no_passing = False

    key = "centerline_2lane_no_passing" if no_passing else "centerline_2lane_passing"
    return t[key]


def two_way_forward(
    lane_index: int,
    total_forward: int,
    side: Literal["left", "right"],
    *,
    standard: str | None = None,
) -> str:
    """Subtype for a forward-lane boundary in a two-way road."""
    t = _table(standard)

    if side == "right" and lane_index == total_forward - 1:
        return t["tw_forward_outer"]

    return t["tw_forward_interior"]


def two_way_backward(
    lane_index: int,
    total_backward: int,
    side: Literal["left", "right"],
    *,
    standard: str | None = None,
) -> str:
    """Subtype for a backward-lane boundary in a two-way road."""
    t = _table(standard)

    if side == "left" and lane_index == total_backward - 1:
        return t["tw_backward_outer"]

    return t["tw_backward_interior"]


def one_way_mark(
    lane_index: int,
    total_lanes: int,
    side: Literal["left", "right"],
    *,
    standard: str | None = None,
) -> str:
    """Subtype for a lane boundary in a one-way road."""
    t = _table(standard)

    if side == "left" and lane_index == 0:
        return t["ow_left_edge"]
    if side == "right" and lane_index == total_lanes - 1:
        return t["ow_right_edge"]

    return t["ow_interior"]


def ramp_mark(
    role: Literal[
        "aux_left",
        "aux_right",
        "spiral_edge",
        "departure_edge",
        "approach_edge",
        "gore",
        "interior",
        "left_edge",
        "right_edge",
    ],
    *,
    standard: str | None = None,
) -> str:
    """Subtype for a ramp road line."""
    return _table(standard)[f"ramp_{role}"]


def intersection_mark(
    role: Literal["connection"] = "connection", *, standard: str | None = None
) -> str:
    """Subtype for an intersection connection lane boundary."""
    return _table(standard)[f"int_{role}"]


def roundabout_mark(
    role: Literal["ring_outer", "ring_inner", "ring_interior", "connection"],
    *,
    standard: str | None = None,
) -> str:
    """Subtype for a roundabout road line."""
    return _table(standard)[f"rab_{role}"]


def one_way_mark_kwargs(
    lane_index: int,
    total_lanes: int,
    side: Literal["left", "right"],
    *,
    standard: str | None = None,
    custom_tags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return RoadLine kwargs for a one-way lane boundary."""
    token = one_way_mark(lane_index, total_lanes, side, standard=standard)

    tags = {"module": "one_way", "lane_index": lane_index, "side": side}

    if custom_tags is not None:
        tags.update(custom_tags)

    return roadline_render_kwargs(token, tags)


def two_way_centerline_kwargs(
    forward_lane_num: int,
    backward_lane_num: int,
    *,
    no_passing: bool | None = None,
    standard: str | None = None,
    custom_tags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return RoadLine kwargs for a two-way road centerline."""
    token = two_way_centerline(
        forward_lane_num, backward_lane_num, no_passing=no_passing, standard=standard
    )

    tags = {"module": "two_way", "role": "centerline"}

    if custom_tags is not None:
        tags.update(custom_tags)

    return roadline_render_kwargs(token, tags)


def two_way_forward_kwargs(
    lane_index: int,
    total_forward: int,
    side: Literal["left", "right"],
    *,
    standard: str | None = None,
    custom_tags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return RoadLine kwargs for a forward two-way lane boundary."""
    token = two_way_forward(lane_index, total_forward, side, standard=standard)

    tags = {"module": "two_way", "direction": "forward", "lane_index": lane_index, "side": side}

    if custom_tags is not None:
        tags.update(custom_tags)

    return roadline_render_kwargs(token, tags)


def two_way_backward_kwargs(
    lane_index: int,
    total_backward: int,
    side: Literal["left", "right"],
    *,
    standard: str | None = None,
    custom_tags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return RoadLine kwargs for a backward two-way lane boundary."""
    token = two_way_backward(lane_index, total_backward, side, standard=standard)

    tags = {"module": "two_way", "direction": "backward", "lane_index": lane_index, "side": side}

    if custom_tags is not None:
        tags.update(custom_tags)

    return roadline_render_kwargs(token, tags)


def ramp_mark_kwargs(
    role: Literal[
        "aux_left",
        "aux_right",
        "spiral_edge",
        "departure_edge",
        "approach_edge",
        "gore",
        "interior",
        "left_edge",
        "right_edge",
    ],
    *,
    standard: str | None = None,
    custom_tags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return RoadLine kwargs for a ramp marking."""
    token = ramp_mark(role, standard=standard)

    tags = {"module": "ramp", "role": role}

    if custom_tags is not None:
        tags.update(custom_tags)

    return roadline_render_kwargs(token, tags)


def intersection_mark_kwargs(
    role: Literal["connection"] = "connection",
    *,
    standard: str | None = None,
    custom_tags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return RoadLine kwargs for an intersection marking."""
    token = intersection_mark(role, standard=standard)

    tags = {"module": "intersection", "role": role}

    if custom_tags is not None:
        tags.update(custom_tags)

    return roadline_render_kwargs(token, tags)


def roundabout_mark_kwargs(
    role: Literal["ring_outer", "ring_inner", "ring_interior", "connection"],
    *,
    standard: str | None = None,
    custom_tags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return RoadLine kwargs for a roundabout marking."""
    token = roundabout_mark(role, standard=standard)

    tags = {"module": "roundabout", "role": role}

    if custom_tags is not None:
        tags.update(custom_tags)

    return roadline_render_kwargs(token, tags)
