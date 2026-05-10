# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Routing cost builders for lane-level topology graphs."""

from __future__ import annotations

from math import sqrt
from typing import Callable, Literal, Optional

from tactics2d.map.element import Lane, Map

from .utils import get_lane_length

RoutingCostFunction = Callable[[Map, Lane, Lane, str], float]
RoutingCostMode = Literal[
    "distance",
    "time",
    "lanelet2_distance",
    "lanelet2_time",
    "apollo_inspired",
]

_DEFAULT_SPEED_MPS = 13.89
_EPS = 1e-6


def _lane_speed_mps(lane: Lane, default_speed_mps: float) -> float:
    speed_limit = getattr(lane, "speed_limit", None)
    if speed_limit is None or speed_limit <= 0:
        return default_speed_mps
    return float(speed_limit)


def _apollo_turn_penalty(lane: Lane, left: float, right: float, uturn: float) -> float:
    text = []
    if lane.subtype:
        text.append(str(lane.subtype).lower())
    if lane.custom_tags is not None:
        for key in ("turn", "turn_type", "maneuver", "movement"):
            value = lane.custom_tags.get(key)
            if value is not None:
                text.append(str(value).lower())

    merged = " ".join(text)
    if "u-turn" in merged or "uturn" in merged or "u_turn" in merged:
        return uturn
    if "left" in merged:
        return left
    if "right" in merged:
        return right
    return 0.0


def build_distance_cost(lane_change_penalty: float = 0.0) -> RoutingCostFunction:
    """Build a classic distance-based routing cost."""

    def cost(map_: Map, from_lane: Lane, to_lane: Lane, relation: str) -> float:
        del map_, from_lane
        traversal_cost = get_lane_length(to_lane)
        if relation == "neighbor":
            return traversal_cost + lane_change_penalty
        return traversal_cost

    return cost


def build_time_cost(
    lane_change_penalty: float = 0.0,
    default_speed_mps: float = _DEFAULT_SPEED_MPS,
) -> RoutingCostFunction:
    """Build a classic travel-time-based routing cost."""

    def cost(map_: Map, from_lane: Lane, to_lane: Lane, relation: str) -> float:
        del map_, from_lane
        speed_mps = max(_lane_speed_mps(to_lane, default_speed_mps), _EPS)
        traversal_cost = get_lane_length(to_lane) / speed_mps
        if relation == "neighbor":
            return traversal_cost + lane_change_penalty
        return traversal_cost

    return cost


def build_lanelet2_distance_cost(
    lane_change_cost: float = 2.0,
    min_lane_change_length: float = 0.0,
) -> RoutingCostFunction:
    """Build a Lanelet2-style distance cost.

    Successor edges use the average 2D length of the current and next lane,
    while lane-change edges use a fixed lane-change cost. A minimal lane-change
    length can be enforced to disable lane changes on very short lanes.
    """

    def cost(map_: Map, from_lane: Lane, to_lane: Lane, relation: str) -> float:
        del map_
        from_length = get_lane_length(from_lane)
        to_length = get_lane_length(to_lane)
        if relation == "neighbor":
            if min_lane_change_length > 0.0 and from_length < min_lane_change_length:
                return float("inf")
            return lane_change_cost
        return 0.5 * (from_length + to_length)

    return cost


def build_lanelet2_time_cost(
    lane_change_cost: float = 2.0,
    min_lane_change_length: float = 0.0,
    default_speed_mps: float = _DEFAULT_SPEED_MPS,
) -> RoutingCostFunction:
    """Build a Lanelet2-style travel-time cost."""

    def cost(map_: Map, from_lane: Lane, to_lane: Lane, relation: str) -> float:
        del map_
        from_length = get_lane_length(from_lane)
        if relation == "neighbor":
            if min_lane_change_length > 0.0 and from_length < min_lane_change_length:
                return float("inf")
            return lane_change_cost

        from_speed = max(_lane_speed_mps(from_lane, default_speed_mps), _EPS)
        to_speed = max(_lane_speed_mps(to_lane, default_speed_mps), _EPS)
        from_time = get_lane_length(from_lane) / from_speed
        to_time = get_lane_length(to_lane) / to_speed
        return 0.5 * (from_time + to_time)

    return cost


def build_apollo_inspired_cost(
    base_speed_mps: float = _DEFAULT_SPEED_MPS,
    change_penalty: float = 20.0,
    base_changing_length: float = 50.0,
    left_turn_penalty: float = 0.0,
    right_turn_penalty: float = 0.0,
    uturn_penalty: float = 0.0,
) -> RoutingCostFunction:
    """Build an Apollo-inspired node/edge cost approximation.

    Apollo's published routing cost separates lane-intrinsic node cost from
    transition edge cost. In this lane-graph implementation, the node part is
    approximated as a nonnegative lane traversal cost:

    * lane_length * speed_ratio
    * plus an optional turn penalty inferred from lane metadata.

    The edge part keeps forward transitions free and applies a lane-change
    penalty for lateral transitions. This is an approximation that preserves
    Apollo's cost decomposition while remaining compatible with generic
    nonnegative shortest-path search.
    """

    def _node_cost(lane: Lane) -> float:
        lane_length = get_lane_length(lane)
        speed_mps = _lane_speed_mps(lane, base_speed_mps)
        if speed_mps >= base_speed_mps:
            speed_ratio = 1.0 / sqrt(speed_mps / base_speed_mps)
        else:
            speed_ratio = 1.0
        return lane_length * speed_ratio + _apollo_turn_penalty(
            lane, left_turn_penalty, right_turn_penalty, uturn_penalty
        )

    def _lane_change_multiplier(from_lane: Lane) -> float:
        if base_changing_length <= 0.0:
            return 1.0
        from_length = get_lane_length(from_lane)
        if from_length <= 0.0:
            return float("inf")
        if from_length >= base_changing_length:
            return 1.0
        return (from_length / base_changing_length) ** -1.5

    def cost(map_: Map, from_lane: Lane, to_lane: Lane, relation: str) -> float:
        del map_
        node_cost = _node_cost(to_lane)
        if relation == "neighbor":
            return node_cost + change_penalty * _lane_change_multiplier(from_lane)
        return node_cost

    return cost


def build_cost_function(
    cost_mode: RoutingCostMode = "distance",
    *,
    cost_fn: Optional[RoutingCostFunction] = None,
    **kwargs,
) -> RoutingCostFunction:
    """Resolve a routing cost function from a built-in mode or custom callable."""

    if cost_fn is not None:
        return cost_fn

    if cost_mode == "distance":
        return build_distance_cost(**kwargs)
    if cost_mode == "time":
        return build_time_cost(**kwargs)
    if cost_mode == "lanelet2_distance":
        return build_lanelet2_distance_cost(**kwargs)
    if cost_mode == "lanelet2_time":
        return build_lanelet2_time_cost(**kwargs)
    if cost_mode in {"apollo_inspired", "apollo_like"}:
        return build_apollo_inspired_cost(**kwargs)

    raise ValueError(f"Unsupported cost mode: {cost_mode}")


def build_apollo_like_cost(**kwargs) -> RoutingCostFunction:
    """Backward-compatible alias for :func:`build_apollo_inspired_cost`."""

    return build_apollo_inspired_cost(**kwargs)
