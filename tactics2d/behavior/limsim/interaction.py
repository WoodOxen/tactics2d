# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Interaction grouping and conflict checks."""

from typing import Dict, List, Optional, Sequence, Tuple

from tactics2d.geometry import spatial
from tactics2d.map.element import Map

from .config import LimSimConfig
from .decision_state import AgentDecisionState


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
        distance = spatial.euclidean_distance(source.location, target.location)
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

        # Topology is available and no relation matched: treat the pair as
        # non-interacting.
        return False

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
        return max(rear_lane.length - rear.route_progress, 0.0) + front.route_progress

    def _is_junction_like(self, lane) -> bool:
        tags = lane.custom_tags or {}
        subtype = (lane.subtype or "").lower()
        return bool(tags.get("junction")) or "junction" in subtype or "intersection" in subtype


def has_trajectory_collision(trajectories: Sequence[Sequence[AgentDecisionState]]) -> bool:
    """Check whether any predicted footprints overlap at the same future step."""

    return first_collision_info(trajectories) is not None


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
            source_radius = 0.5 * (source.length**2 + source.width**2) ** 0.5
            source_shape = None
            for j in range(i + 1, len(states)):
                target = states[j]
                target_radius = 0.5 * (target.length**2 + target.width**2) ** 0.5
                if (
                    spatial.euclidean_distance(source.location, target.location)
                    > source_radius + target_radius
                ):
                    continue
                if source_shape is None:
                    source_shape = source.footprint
                if source_shape.intersects(target.footprint):
                    return step, source, target
    return None


def trajectory_safety_summary(
    trajectories: Sequence[Sequence[AgentDecisionState]],
) -> Tuple[Optional[Tuple[int, AgentDecisionState, AgentDecisionState]], float, float]:
    """Compute collision, minimum distance, and closing risk in one traversal."""

    if len(trajectories) < 2:
        return None, float("inf"), 0.0

    steps = min(len(trajectory) for trajectory in trajectories)
    min_distance = float("inf")
    closing_factor = 0.0
    for step in range(steps):
        states = [trajectory[step] for trajectory in trajectories]
        for i, source in enumerate(states):
            source_radius = 0.5 * (source.length**2 + source.width**2) ** 0.5
            source_shape = None
            for target in states[i + 1 :]:
                distance = spatial.euclidean_distance(source.location, target.location)
                min_distance = min(min_distance, distance)

                if source.lane_id == target.lane_id:
                    rear, front = source, target
                    if rear.route_progress > front.route_progress:
                        rear, front = front, rear
                    gap = front.route_progress - rear.route_progress
                    closing_speed = rear.speed - front.speed
                    safe_gap = rear.length + max(rear.speed, 0.0)
                    if closing_speed > 0.0 and gap < safe_gap:
                        closing_factor += closing_speed * (safe_gap - gap)

                target_radius = 0.5 * (target.length**2 + target.width**2) ** 0.5
                if distance > source_radius + target_radius:
                    continue
                if source_shape is None:
                    source_shape = source.footprint
                if source_shape.intersects(target.footprint):
                    return (step, source, target), min_distance, closing_factor

    return None, min_distance, closing_factor
