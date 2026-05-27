# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Lane marking rule functions for road segment generators."""

from __future__ import annotations

from typing import Any, Literal

from .lane_marking_config import MARKING_SPECS, STANDARD_TABLES, Standard

_STANDARD: Standard = "MUTCD"


def set_standard(standard: Standard) -> None:
    """Set the active lane marking standard globally.

    Args:
        standard: Target standard. Must be ``"MUTCD"`` (US) or ``"GB"``
            (Chinese GB 5768).

    Raises:
        ValueError: If ``standard`` is not ``"MUTCD"`` or ``"GB"``.
    """
    global _STANDARD

    if standard not in ("MUTCD", "GB"):
        raise ValueError(f"Unknown marking standard '{standard}'. Use 'MUTCD' or 'GB'.")

    _STANDARD = standard


def get_standard() -> str:
    """Return the active lane marking standard."""
    return _STANDARD


def _table(standard: str | None) -> dict[str, str]:
    """Return rule table for a standard, falling back to MUTCD."""
    return STANDARD_TABLES.get(standard or _STANDARD, STANDARD_TABLES["MUTCD"])


def get_marking_spec(token: str | None):
    """Return the semantic marking specification for a token.

    Args:
        token: Marking token string (e.g. ``"dashed_white"``), or ``None``.

    Returns:
        The corresponding :class:`MarkingSpec`. Returns the ``"solid"`` spec
        when ``token`` is ``None`` or unrecognised.
    """
    if token is None:
        return MARKING_SPECS["solid"]

    return MARKING_SPECS.get(token, MARKING_SPECS["solid"])


def roadline_render_kwargs(
    token: str | None, custom_tags: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return RoadLine constructor kwargs for a semantic marking token.

    Args:
        token: Marking token string (e.g. ``"solid_yellow"``), or ``None``
            (falls back to ``"solid"`` spec).
        custom_tags: Optional additional tags merged into ``custom_tags`` with
            higher priority than the built-in marking metadata.

    Returns:
        A dict with keys ``type_``, ``subtype``, ``color``, and
        ``custom_tags``.
    """
    spec = get_marking_spec(token)

    if spec.role == "virtual":
        type_ = "virtual"
        subtype = "virtual"
    elif spec.role in ("edge", "ramp_edge"):
        type_ = "road_border"
        subtype = "dashed" if spec.pattern in ("dashed", "double_dashed", "dotted") else "solid"
    else:
        type_ = "line_thin"
        subtype = "dashed" if "dashed" in spec.pattern or spec.pattern == "dotted" else "solid"

    color = "yellow" if spec.color == "yellow" else "white"

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

    return {"type_": type_, "subtype": subtype, "color": color, "custom_tags": tags}


def roadline_lane_change(token: str | None) -> tuple[bool, bool]:
    """Return the lane-change permission tuple for a marking token.

    Args:
        token: Marking token string, or ``None`` (falls back to solid spec).

    Returns:
        Tuple ``(allow_left, allow_right)`` where each element is ``True``
        when a vehicle may cross the marking in that direction.
    """
    return get_marking_spec(token).lane_change


def is_crossable(token: str | None) -> bool:
    """Return ``True`` when a marking may be crossed from either side.

    Args:
        token: Marking token string, or ``None``.

    Returns:
        ``True`` if lane changes are permitted in both directions.
    """
    return roadline_lane_change(token) == (True, True)


def is_solid(token: str | None) -> bool:
    """Return ``True`` when the marking pattern contains a solid component.

    Args:
        token: Marking token string, or ``None``.

    Returns:
        ``True`` for ``"solid"``, ``"solid_dashed"``, ``"double_solid"``, etc.
    """
    return "solid" in get_marking_spec(token).pattern


def is_dashed(token: str | None) -> bool:
    """Return ``True`` when the marking pattern contains a dashed or dotted component.

    Args:
        token: Marking token string, or ``None``.

    Returns:
        ``True`` for ``"dashed"``, ``"dotted"``, ``"double_dashed"``, etc.
    """
    pattern = get_marking_spec(token).pattern
    return "dashed" in pattern or pattern == "dotted"


def two_way_centerline(
    forward_lane_num: int,
    backward_lane_num: int,
    *,
    no_passing: bool | None = None,
    standard: str | None = None,
) -> str:
    """Return the marking token for the centre line of a two-way road."""
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
    """Return the marking token for a forward-lane boundary in a two-way road."""
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
    """Return the marking token for a backward-lane boundary in a two-way road."""
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
    """Return the marking token for a lane boundary in a one-way road."""
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
    """Return the marking token for a ramp road line."""
    return _table(standard)[f"ramp_{role}"]


def intersection_mark(
    role: Literal["connection"] = "connection", *, standard: str | None = None
) -> str:
    """Return the marking token for an intersection connection lane boundary."""
    return _table(standard)[f"int_{role}"]


def roundabout_mark(
    role: Literal["ring_outer", "ring_inner", "ring_interior", "connection"],
    *,
    standard: str | None = None,
) -> str:
    """Return the marking token for a roundabout road line."""
    return _table(standard)[f"rab_{role}"]


def one_way_boundary_token(boundary_index: int, boundary_num: int) -> str:
    """Return the active-standard marking token for a one-way road boundary.

    Boundary indices run left-to-right in the driving direction.

    Args:
        boundary_index: 0 = leftmost edge, ``boundary_num - 1`` = rightmost edge.
        boundary_num: Total number of boundaries (``lane_num + 1``).

    Returns:
        Marking token string.
    """
    lane_num = boundary_num - 1

    if boundary_index == 0:
        return one_way_mark(0, lane_num, "left")
    if boundary_index == boundary_num - 1:
        return one_way_mark(lane_num - 1, lane_num, "right")
    return one_way_mark(boundary_index - 1, lane_num, "right")
