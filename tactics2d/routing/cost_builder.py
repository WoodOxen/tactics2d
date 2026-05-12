# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Routing cost builders for lane-level topology graphs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import sqrt
from typing import Callable, Dict, Literal, Optional, Type

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


def _turn_penalty_from_lane_metadata(
    lane: Lane, left: float, right: float, uturn: float
) -> float:
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


class CostBuilder(ABC):
    """Base class for routing cost builders."""

    @abstractmethod
    def build(self) -> RoutingCostFunction:
        """Build a routing cost callback."""


@dataclass
class DistanceCostBuilder(CostBuilder):
    """Classic length-based traversal cost."""

    lane_change_penalty: float = 0.0

    def build(self) -> RoutingCostFunction:
        def cost(map_: Map, from_lane: Lane, to_lane: Lane, relation: str) -> float:
            del map_, from_lane
            traversal_cost = get_lane_length(to_lane)
            if relation == "neighbor":
                return traversal_cost + self.lane_change_penalty
            return traversal_cost

        return cost


@dataclass
class TravelTimeCostBuilder(CostBuilder):
    """Classic travel-time traversal cost."""

    lane_change_penalty: float = 0.0
    default_speed_mps: float = _DEFAULT_SPEED_MPS

    def build(self) -> RoutingCostFunction:
        def cost(map_: Map, from_lane: Lane, to_lane: Lane, relation: str) -> float:
            del map_, from_lane
            speed_mps = max(_lane_speed_mps(to_lane, self.default_speed_mps), _EPS)
            traversal_cost = get_lane_length(to_lane) / speed_mps
            if relation == "neighbor":
                return traversal_cost + self.lane_change_penalty
            return traversal_cost

        return cost


@dataclass
class AveragedLengthCostBuilder(CostBuilder):
    """Average source/target lane length with fixed lateral transition cost.

    This mechanism matches a Lanelet2-style relation-aware distance cost:
    successor edges use averaged traversal length, while neighbor edges use a
    fixed lane-change cost with an optional minimal source-lane length filter.
    """

    lane_change_cost: float = 2.0
    min_lane_change_length: float = 0.0

    def build(self) -> RoutingCostFunction:
        def cost(map_: Map, from_lane: Lane, to_lane: Lane, relation: str) -> float:
            del map_
            from_length = get_lane_length(from_lane)
            to_length = get_lane_length(to_lane)
            if relation == "neighbor":
                if (
                    self.min_lane_change_length > 0.0
                    and from_length < self.min_lane_change_length
                ):
                    return float("inf")
                return self.lane_change_cost
            return 0.5 * (from_length + to_length)

        return cost


@dataclass
class AveragedTravelTimeCostBuilder(CostBuilder):
    """Average source/target travel time with fixed lateral transition cost.

    This mechanism matches a Lanelet2-style relation-aware travel-time cost:
    successor edges use averaged traversal time, while neighbor edges use a
    fixed lane-change cost with an optional minimal source-lane length filter.
    """

    lane_change_cost: float = 2.0
    min_lane_change_length: float = 0.0
    default_speed_mps: float = _DEFAULT_SPEED_MPS

    def build(self) -> RoutingCostFunction:
        def cost(map_: Map, from_lane: Lane, to_lane: Lane, relation: str) -> float:
            del map_
            from_length = get_lane_length(from_lane)
            if relation == "neighbor":
                if (
                    self.min_lane_change_length > 0.0
                    and from_length < self.min_lane_change_length
                ):
                    return float("inf")
                return self.lane_change_cost

            from_speed = max(_lane_speed_mps(from_lane, self.default_speed_mps), _EPS)
            to_speed = max(_lane_speed_mps(to_lane, self.default_speed_mps), _EPS)
            from_time = get_lane_length(from_lane) / from_speed
            to_time = get_lane_length(to_lane) / to_speed
            return 0.5 * (from_time + to_time)

        return cost


@dataclass
class NodeEdgeCostBuilder(CostBuilder):
    """Lane-intrinsic cost plus lateral transition penalty.

    This mechanism follows an Apollo-inspired decomposition: each routed lane
    contributes a node-like traversal cost based on lane length, speed, and
    optional turn penalty; lateral transitions add an extra edge-like lane-
    change penalty scaled by the available source-lane length.
    """

    base_speed_mps: float = _DEFAULT_SPEED_MPS
    change_penalty: float = 20.0
    base_changing_length: float = 50.0
    left_turn_penalty: float = 0.0
    right_turn_penalty: float = 0.0
    uturn_penalty: float = 0.0

    def build(self) -> RoutingCostFunction:
        def node_cost(lane: Lane) -> float:
            lane_length = get_lane_length(lane)
            speed_mps = _lane_speed_mps(lane, self.base_speed_mps)
            if speed_mps >= self.base_speed_mps:
                speed_ratio = 1.0 / sqrt(speed_mps / self.base_speed_mps)
            else:
                speed_ratio = 1.0
            return lane_length * speed_ratio + _turn_penalty_from_lane_metadata(
                lane,
                self.left_turn_penalty,
                self.right_turn_penalty,
                self.uturn_penalty,
            )

        def lane_change_multiplier(from_lane: Lane) -> float:
            if self.base_changing_length <= 0.0:
                return 1.0
            from_length = get_lane_length(from_lane)
            if from_length <= 0.0:
                return float("inf")
            if from_length >= self.base_changing_length:
                return 1.0
            return (from_length / self.base_changing_length) ** -1.5

        def cost(map_: Map, from_lane: Lane, to_lane: Lane, relation: str) -> float:
            del map_
            traversal_cost = node_cost(to_lane)
            if relation == "neighbor":
                return traversal_cost + self.change_penalty * lane_change_multiplier(
                    from_lane
                )
            return traversal_cost

        return cost


_BUILDER_CLASSES: Dict[str, Type[CostBuilder]] = {
    "distance": DistanceCostBuilder,
    "time": TravelTimeCostBuilder,
    "lanelet2_distance": AveragedLengthCostBuilder,
    "lanelet2_time": AveragedTravelTimeCostBuilder,
    "apollo_inspired": NodeEdgeCostBuilder,
    "apollo_like": NodeEdgeCostBuilder,
}


def build_cost_builder(cost_mode: str = "distance", **kwargs) -> CostBuilder:
    """Resolve a cost-builder instance from a built-in mode."""

    try:
        builder_cls = _BUILDER_CLASSES[cost_mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported cost mode: {cost_mode}") from exc
    return builder_cls(**kwargs)


def build_cost_function(
    cost_mode: RoutingCostMode = "distance",
    *,
    cost_fn: Optional[RoutingCostFunction] = None,
    **kwargs,
) -> RoutingCostFunction:
    """Resolve a routing cost callback from a built-in mode or custom callable."""

    if cost_fn is not None:
        return cost_fn
    return build_cost_builder(cost_mode=cost_mode, **kwargs).build()


def build_distance_cost(**kwargs) -> RoutingCostFunction:
    """Build the classic distance-based routing preset."""

    return DistanceCostBuilder(**kwargs).build()


def build_time_cost(**kwargs) -> RoutingCostFunction:
    """Build the classic travel-time routing preset."""

    return TravelTimeCostBuilder(**kwargs).build()


def build_lanelet2_distance_cost(**kwargs) -> RoutingCostFunction:
    """Build the averaged-length routing preset."""

    return AveragedLengthCostBuilder(**kwargs).build()


def build_lanelet2_time_cost(**kwargs) -> RoutingCostFunction:
    """Build the averaged-travel-time routing preset."""

    return AveragedTravelTimeCostBuilder(**kwargs).build()


def build_apollo_inspired_cost(**kwargs) -> RoutingCostFunction:
    """Build the node/edge routing preset."""

    return NodeEdgeCostBuilder(**kwargs).build()


def build_apollo_like_cost(**kwargs) -> RoutingCostFunction:
    """Backward-compatible alias for :func:`build_apollo_inspired_cost`."""

    return build_apollo_inspired_cost(**kwargs)


__all__ = [
    "RoutingCostFunction",
    "RoutingCostMode",
    "CostBuilder",
    "DistanceCostBuilder",
    "TravelTimeCostBuilder",
    "AveragedLengthCostBuilder",
    "AveragedTravelTimeCostBuilder",
    "NodeEdgeCostBuilder",
    "build_cost_builder",
    "build_cost_function",
    "build_distance_cost",
    "build_time_cost",
    "build_lanelet2_distance_cost",
    "build_lanelet2_time_cost",
    "build_apollo_inspired_cost",
    "build_apollo_like_cost",
]
