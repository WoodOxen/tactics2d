# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Interaction grouping and conflict checks."""

from typing import Dict, List, Optional, Sequence, Tuple

from tactics2d.geometry import euclidean_distance, oriented_box
from tactics2d.map.element import Map

from .config import LimSimConfig
from .schema import AgentDecisionState


class InteractionGraph:
    """Find connected groups of mutually relevant agents."""

    def __init__(self, config: LimSimConfig):
        self.config = config

    def build_groups(
        self, states: Dict[object, AgentDecisionState], map_: Optional[Map] = None
    ) -> List[List[object]]:
        """Build interaction groups with LimSim-style topology-aware rules."""

        agent_ids = list(states.keys())
        adjacency = {agent_id: set() for agent_id in agent_ids}
        for index, source_id in enumerate(agent_ids):
            for target_id in agent_ids[index + 1 :]:
                if self._has_interaction(states[source_id], states[target_id], map_):
                    adjacency[source_id].add(target_id)
                    adjacency[target_id].add(source_id)

        groups = []
        unseen = set(agent_ids)
        while unseen:
            start = unseen.pop()
            stack = [start]
            group = [start]
            while stack:
                current = stack.pop()
                for neighbor in adjacency[current]:
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        stack.append(neighbor)
                        group.append(neighbor)
            groups.extend(self._split_large_group(group, states))
        return groups

    def _split_large_group(
        self, group: List[object], states: Dict[object, AgentDecisionState]
    ) -> List[List[object]]:
        if len(group) <= self.config.max_group_size:
            return [group]

        ordered = sorted(group, key=lambda agent_id: states[agent_id].speed, reverse=True)
        return [
            ordered[index : index + self.config.max_group_size]
            for index in range(0, len(ordered), self.config.max_group_size)
        ]

    def _has_interaction(
        self, source: AgentDecisionState, target: AgentDecisionState, map_: Optional[Map]
    ) -> bool:
        distance = euclidean_distance(source.location, target.location)
        if distance <= self.config.conflict_distance:
            return True

        if map_ is None or source.lane_id is None or target.lane_id is None:
            return distance <= self.config.interaction_distance

        lane_i = map_.lanes.get(source.lane_id)
        lane_j = map_.lanes.get(target.lane_id)
        if lane_i is None or lane_j is None:
            return distance <= self.config.interaction_distance

        if self._is_junction_like(lane_i) or self._is_junction_like(lane_j):
            return distance <= self.config.junction_interaction_distance

        if source.lane_id == target.lane_id:
            return self._longitudinally_close(source, target)

        if target.lane_id in lane_i.successors:
            gap = self._successor_gap(source, target, lane_i)
            return gap <= self._dynamic_interaction_distance(source)

        if source.lane_id in lane_j.successors:
            gap = self._successor_gap(target, source, lane_j)
            return gap <= self._dynamic_interaction_distance(target)

        if target.lane_id in lane_i.left_neighbors | lane_i.right_neighbors:
            if abs(source.route_progress - target.route_progress) > source.length + target.length:
                return False
            return (
                source.action.is_lane_change
                or target.action.is_lane_change
                or distance <= (source.length + target.length)
            )

        return distance <= self.config.interaction_distance

    def _longitudinally_close(self, source: AgentDecisionState, target: AgentDecisionState) -> bool:
        rear, front = (source, target)
        if source.route_progress > target.route_progress:
            rear, front = target, source
        gap = front.route_progress - rear.route_progress
        return gap <= self._dynamic_interaction_distance(rear)

    def _dynamic_interaction_distance(self, agent: AgentDecisionState) -> float:
        return self.config.same_lane_time_headway * agent.speed + agent.length

    def _successor_gap(
        self, rear: AgentDecisionState, front: AgentDecisionState, rear_lane
    ) -> float:
        lane_length = (
            float(rear_lane.geometry.length) / 2.0 if rear_lane.geometry is not None else 0.0
        )
        return max(lane_length - rear.route_progress, 0.0) + front.route_progress

    def _is_junction_like(self, lane) -> bool:
        tags = lane.custom_tags or {}
        subtype = (lane.subtype or "").lower()
        return bool(tags.get("junction")) or "junction" in subtype or "intersection" in subtype


def has_trajectory_collision(trajectories: Sequence[Sequence[AgentDecisionState]]) -> bool:
    """Check whether any predicted footprints overlap at the same future step."""

    if len(trajectories) < 2:
        return False
    steps = min(len(trajectory) for trajectory in trajectories)
    for step in range(steps):
        states = [trajectory[step] for trajectory in trajectories]
        for i, source in enumerate(states):
            source_shape = _footprint(source)  # hoisted: compute once per source
            source_radius = 0.5 * (source.length**2 + source.width**2) ** 0.5
            for j in range(i + 1, len(states)):
                target = states[j]
                target_radius = 0.5 * (target.length**2 + target.width**2) ** 0.5
                if (
                    euclidean_distance(source.location, target.location)
                    > source_radius + target_radius
                ):
                    continue
                if source_shape.intersects(_footprint(target)):
                    return True
    return False


def first_collision_info(
    trajectories: Sequence[Sequence[AgentDecisionState]],
) -> Optional[Tuple[int, AgentDecisionState, AgentDecisionState]]:
    """Return the first colliding step and states, if any."""

    if len(trajectories) < 2:
        return None
    steps = min(len(trajectory) for trajectory in trajectories)
    for step in range(steps):
        states = [trajectory[step] for trajectory in trajectories]
        for i, source in enumerate(states):
            source_shape = _footprint(source)  # hoisted: compute once per source
            source_radius = 0.5 * (source.length**2 + source.width**2) ** 0.5
            for j in range(i + 1, len(states)):
                target = states[j]
                target_radius = 0.5 * (target.length**2 + target.width**2) ** 0.5
                if (
                    euclidean_distance(source.location, target.location)
                    > source_radius + target_radius
                ):
                    continue
                if source_shape.intersects(_footprint(target)):
                    return step, source, target
    return None


def minimum_pair_distance(trajectories: Sequence[Sequence[AgentDecisionState]]) -> float:
    """Return the minimum center distance among predicted agents."""

    if len(trajectories) < 2:
        return float("inf")
    steps = min(len(trajectory) for trajectory in trajectories)
    min_distance = float("inf")
    for step in range(steps):
        for i, source in enumerate(trajectories):
            for target in trajectories[i + 1 :]:
                distance = euclidean_distance(source[step].location, target[step].location)
                min_distance = min(min_distance, distance)
    return min_distance


def _footprint(state: AgentDecisionState):
    return oriented_box(state.x, state.y, state.heading, state.length, state.width)
