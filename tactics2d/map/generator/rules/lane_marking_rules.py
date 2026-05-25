# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Lane marking rule functions for road element generators."""

from __future__ import annotations

from typing import Any, Literal

from .lane_marking_config import (
    MARKING_SPECS,
    STANDARD_TABLES,
    Color,
    MarkingSpec,
    Pattern,
    Role,
    Standard,
)

__all__ = [
    "Standard",
    "Pattern",
    "Color",
    "Role",
    "MarkingSpec",
    "set_standard",
    "get_standard",
    "get_marking_spec",
    "get_all_marking_specs",
    "roadline_render_color",
    "roadline_render_style",
    "roadline_render_subtype",
    "roadline_render_type",
    "roadline_render_kwargs",
    "roadline_lane_change",
    "is_crossable",
    "is_solid",
    "is_dashed",
    "two_way_centerline",
    "two_way_forward",
    "two_way_backward",
    "one_way_mark",
    "ramp_mark",
    "intersection_mark",
    "roundabout_mark",
    "one_way_mark_kwargs",
    "two_way_centerline_kwargs",
    "two_way_forward_kwargs",
    "two_way_backward_kwargs",
    "ramp_mark_kwargs",
    "intersection_mark_kwargs",
    "roundabout_mark_kwargs",
]

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
    """Return the active lane marking standard.

    Returns:
        The currently active standard string (``"MUTCD"`` or ``"GB"``).
    """
    return _STANDARD


def _table(standard: str | None) -> dict[str, str]:
    """Return rule table for a standard.

    Args:
        standard: Standard name, or ``None`` to use the active global standard.

    Returns:
        Role-to-token mapping dict for the requested standard. Falls back to
        MUTCD if the name is not recognised.
    """
    return STANDARD_TABLES.get(standard or _STANDARD, STANDARD_TABLES["MUTCD"])


def get_marking_spec(token: str | None) -> MarkingSpec:
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


def roadline_render_color(token: str | None) -> str:
    """Return renderer color key for a roadline token.

    Args:
        token: Marking token string, or ``None`` (falls back to ``"solid"``).

    Returns:
        Color string: ``"yellow"`` or ``"white"``.
    """
    spec = get_marking_spec(token)

    if spec.color == "yellow":
        return "yellow"

    return "white"


def roadline_render_style(token: str | None) -> str:
    """Return renderer-compatible line style for a roadline token.

    Args:
        token: Marking token string, or ``None``.

    Returns:
        Renderer style string (e.g. ``"solid"``, ``"dashed"``).
    """
    return get_marking_spec(token).render_style


def roadline_render_subtype(token: str | None) -> str:
    """Return renderer-compatible RoadLine subtype for a token.

    Args:
        token: Marking token string, or ``None``.

    Returns:
        ``"virtual"`` for virtual markings, ``"dashed"`` for any broken or
        dotted pattern, ``"solid"`` otherwise.
    """
    spec = get_marking_spec(token)

    if spec.role == "virtual":
        return "virtual"

    if spec.pattern in ("dashed", "double_dashed", "dotted"):
        return "dashed"

    if "dashed" in spec.pattern:
        return "dashed"

    return "solid"


def roadline_render_type(token: str | None) -> str:
    """Return renderer-compatible RoadLine type for a token.

    Args:
        token: Marking token string, or ``None``.

    Returns:
        ``"virtual"`` for virtual markings, ``"road_border"`` for edge roles,
        ``"line_thin"`` for all other markings.
    """
    spec = get_marking_spec(token)

    if spec.role == "virtual":
        return "virtual"

    if spec.role in ("edge", "ramp_edge"):
        return "road_border"

    return "line_thin"


def roadline_render_kwargs(
    token: str | None, custom_tags: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return RoadLine constructor kwargs for a semantic marking token.

    Combines renderer properties (``type_``, ``subtype``, ``color``) with a
    ``custom_tags`` dict that encodes marking metadata for downstream
    serialisers.

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
    """Return lane-change permission tuple for a roadline token.

    Args:
        token: Marking token string, or ``None``.

    Returns:
        ``(allow_left, allow_right)`` boolean tuple.
    """
    return get_marking_spec(token).lane_change


def is_crossable(token: str | None) -> bool:
    """Return whether a marking is crossable from both sides.

    Args:
        token: Marking token string, or ``None``.

    Returns:
        ``True`` if both left and right lane changes are permitted.
    """
    return roadline_lane_change(token) == (True, True)


def is_solid(token: str | None) -> bool:
    """Return whether the marking has a solid component.

    Args:
        token: Marking token string, or ``None``.

    Returns:
        ``True`` if the pattern contains ``"solid"``.
    """
    return "solid" in get_marking_spec(token).pattern


def is_dashed(token: str | None) -> bool:
    """Return whether the marking has a dashed or broken component.

    Args:
        token: Marking token string, or ``None``.

    Returns:
        ``True`` if the pattern contains ``"dashed"`` or is ``"dotted"``.
    """
    pattern = get_marking_spec(token).pattern
    return "dashed" in pattern or pattern == "dotted"


def get_all_marking_specs() -> dict[str, MarkingSpec]:
    """Return all registered marking specifications.

    Returns:
        A shallow copy of the full token-to-spec mapping.
    """
    return dict(MARKING_SPECS)


def two_way_centerline(
    forward_lane_num: int,
    backward_lane_num: int,
    *,
    no_passing: bool | None = None,
    standard: str | None = None,
) -> str:
    """Return the marking token for the centre line of a two-way road.

    Args:
        forward_lane_num: Number of forward (outbound) lanes.
        backward_lane_num: Number of backward (inbound) lanes.
        no_passing: Force the ``no_passing`` variant when ``True``. Defaults
            to ``False`` for two-lane roads.
        standard: Lane-marking standard override. Uses the active global
            standard when ``None``.

    Returns:
        Marking token string for the centre line.
    """
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
    """Return the marking token for a forward-lane boundary in a two-way road.

    Args:
        lane_index: Zero-based lane index counted from the centre line outward.
        total_forward: Total number of forward lanes.
        side: Which boundary of the lane (``"left"`` or ``"right"``).
        standard: Lane-marking standard override; uses the global default when
            ``None``.

    Returns:
        Marking token string for the boundary.
    """
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
    """Return the marking token for a backward-lane boundary in a two-way road.

    Args:
        lane_index: Zero-based lane index counted from the centre line outward.
        total_backward: Total number of backward lanes.
        side: Which boundary of the lane (``"left"`` or ``"right"``).
        standard: Lane-marking standard override; uses the global default when
            ``None``.

    Returns:
        Marking token string for the boundary.
    """
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
    """Return the marking token for a lane boundary in a one-way road.

    Args:
        lane_index: Zero-based lane index from the left edge.
        total_lanes: Total number of lanes.
        side: Which boundary of the lane (``"left"`` or ``"right"``).
        standard: Lane-marking standard override; uses the global default when
            ``None``.

    Returns:
        Marking token string for the boundary.
    """
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
    """Return the marking token for a ramp road line.

    Args:
        role: Functional role of the ramp marking. One of ``"aux_left"``,
            ``"aux_right"``, ``"spiral_edge"``, ``"departure_edge"``,
            ``"approach_edge"``, ``"gore"``, ``"interior"``, ``"left_edge"``,
            ``"right_edge"``.
        standard: Lane-marking standard override; uses the global default when
            ``None``.

    Returns:
        Marking token string for the ramp line.
    """
    return _table(standard)[f"ramp_{role}"]


def intersection_mark(
    role: Literal["connection"] = "connection", *, standard: str | None = None
) -> str:
    """Return the marking token for an intersection connection lane boundary.

    Args:
        role: Connection role; currently only ``"connection"`` is supported.
        standard: Lane-marking standard override; uses the global default when
            ``None``.

    Returns:
        Marking token string.
    """
    return _table(standard)[f"int_{role}"]


def roundabout_mark(
    role: Literal["ring_outer", "ring_inner", "ring_interior", "connection"],
    *,
    standard: str | None = None,
) -> str:
    """Return the marking token for a roundabout road line.

    Args:
        role: Semantic role of the roundabout marking. One of
            ``"ring_outer"``, ``"ring_inner"``, ``"ring_interior"``,
            ``"connection"``.
        standard: Lane-marking standard override; uses the global default when
            ``None``.

    Returns:
        Marking token string.
    """
    return _table(standard)[f"rab_{role}"]


def one_way_mark_kwargs(
    lane_index: int,
    total_lanes: int,
    side: Literal["left", "right"],
    *,
    standard: str | None = None,
    custom_tags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return RoadLine kwargs for a one-way lane boundary.

    Args:
        lane_index: Zero-based lane index from the left edge.
        total_lanes: Total number of lanes.
        side: Which boundary (``"left"`` or ``"right"``).
        standard: Lane-marking standard override; uses the global default when
            ``None``.
        custom_tags: Optional additional tags merged into ``custom_tags`` with
            higher priority.

    Returns:
        Dict with ``type_``, ``subtype``, ``color``, and ``custom_tags`` keys.
    """
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
    """Return RoadLine kwargs for a two-way road centerline.

    Args:
        forward_lane_num: Number of forward lanes.
        backward_lane_num: Number of backward lanes.
        no_passing: Force the no-passing variant when ``True``.
        standard: Lane-marking standard override; uses the global default when
            ``None``.
        custom_tags: Optional additional tags merged with higher priority.

    Returns:
        Dict with ``type_``, ``subtype``, ``color``, and ``custom_tags`` keys.
    """
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
    """Return RoadLine kwargs for a forward two-way lane boundary.

    Args:
        lane_index: Zero-based lane index from the centre line outward.
        total_forward: Total forward lane count.
        side: Which boundary (``"left"`` or ``"right"``).
        standard: Lane-marking standard override; uses the global default when
            ``None``.
        custom_tags: Optional additional tags merged with higher priority.

    Returns:
        Dict with ``type_``, ``subtype``, ``color``, and ``custom_tags`` keys.
    """
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
    """Return RoadLine kwargs for a backward two-way lane boundary.

    Args:
        lane_index: Zero-based lane index from the centre line outward.
        total_backward: Total backward lane count.
        side: Which boundary (``"left"`` or ``"right"``).
        standard: Lane-marking standard override; uses the global default when
            ``None``.
        custom_tags: Optional additional tags merged with higher priority.

    Returns:
        Dict with ``type_``, ``subtype``, ``color``, and ``custom_tags`` keys.
    """
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
    """Return RoadLine kwargs for a ramp marking.

    Args:
        role: Functional role of the ramp line. See :func:`ramp_mark` for
            accepted values.
        standard: Lane-marking standard override; uses the global default when
            ``None``.
        custom_tags: Optional additional tags merged with higher priority.

    Returns:
        Dict with ``type_``, ``subtype``, ``color``, and ``custom_tags`` keys.
    """
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
    """Return RoadLine kwargs for an intersection marking.

    Args:
        role: Connection role; currently only ``"connection"`` is supported.
        standard: Lane-marking standard override; uses the global default when
            ``None``.
        custom_tags: Optional additional tags merged with higher priority.

    Returns:
        Dict with ``type_``, ``subtype``, ``color``, and ``custom_tags`` keys.
    """
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
    """Return RoadLine kwargs for a roundabout marking.

    Args:
        role: Semantic role. See :func:`roundabout_mark` for accepted values.
        standard: Lane-marking standard override; uses the global default when
            ``None``.
        custom_tags: Optional additional tags merged with higher priority.

    Returns:
        Dict with ``type_``, ``subtype``, ``color``, and ``custom_tags`` keys.
    """
    token = roundabout_mark(role, standard=standard)
    tags = {"module": "roundabout", "role": role}

    if custom_tags is not None:
        tags.update(custom_tags)

    return roadline_render_kwargs(token, tags)
