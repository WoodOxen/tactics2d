# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Lane-following trajectory rollout for LimSim-style actions."""

from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import LineString, Point

from tactics2d.geometry import polyline, spatial
from tactics2d.map.element import Map
from tactics2d.map.query import SemanticMapQuery

from .action import LimSimAction
from .config import LimSimConfig
from .schema import AgentDecisionState


class LaneFollower:
    """Roll out actions along lane centerlines when possible."""

    def __init__(self, config: LimSimConfig):
        self.config = config

    def rollout(
        self,
        agent: AgentDecisionState,
        action: LimSimAction,
        map_: Optional[Map],
        steps: Optional[int] = None,
    ) -> List[AgentDecisionState]:
        """Generate future states for one agent under one high-level action."""

        horizon = self.config.horizon_steps if steps is None else steps
        route_path, route_lanes, start_progress = self._select_route(agent, action, map_)

        states = []
        speed = agent.speed
        progress = start_progress
        current = agent
        for _ in range(horizon):
            speed = float(
                np.clip(
                    speed + action.acceleration * self.config.dt,
                    self.config.min_speed,
                    self.config.max_speed,
                )
            )
            travel = speed * self.config.dt
            transition = self._lane_transition(current, action, map_)
            if transition is not None:
                current = transition
                route_path, route_lanes, progress = self._select_route(current, action, map_)

            if route_path is not None and route_path.length > 1e-6:
                progress = min(progress + travel, route_path.length)
                point = route_path.interpolate(progress)
                lookahead = route_path.interpolate(min(progress + 0.5, route_path.length))
                heading = spatial.normalize_angle(
                    np.arctan2(lookahead.y - point.y, lookahead.x - point.x)
                )
                lateral_offset = self._next_lateral_offset(current, action)
                x = float(point.x - lateral_offset * np.sin(heading))
                y = float(point.y + lateral_offset * np.cos(heading))
                current = current.with_updates(
                    x=x,
                    y=y,
                    heading=heading,
                    speed=speed,
                    action=action,
                    route_lane_ids=route_lanes,
                    lane_id=current.lane_id,
                    route_progress=progress,
                    lateral_offset=lateral_offset,
                )
            else:
                current = current.with_updates(
                    x=current.x + travel * np.cos(current.heading),
                    y=current.y + travel * np.sin(current.heading),
                    speed=speed,
                    action=action,
                )
            states.append(current)

        return states

    def _lane_transition(
        self, agent: AgentDecisionState, action: LimSimAction, map_: Optional[Map]
    ) -> Optional[AgentDecisionState]:
        """Smoothly transition lane_id when the lateral offset crosses a lane boundary.

        Instead of teleporting the lateral offset to the new lane's frame (which
        produces a discontinuous jump of ~one lane width), this method projects
        the vehicle's current Cartesian position onto the neighbor lane's
        centerline to obtain a **continuous** lateral offset in the new frame.
        """

        if action not in {LimSimAction.LCL, LimSimAction.LCR}:
            return None
        if map_ is None or agent.lane_id is None or agent.lane_id not in map_.lanes:
            return None
        direction = "left" if action == LimSimAction.LCL else "right"
        if not SemanticMapQuery(map_).get_lane_change_permission(
            agent.lane_id, direction, s=agent.route_progress
        ):
            return None

        current_lane = map_.lanes[agent.lane_id]
        current_width = self._lane_width(current_lane)
        if action == LimSimAction.LCL and agent.lateral_offset <= current_width / 2.0:
            return None
        if action == LimSimAction.LCR and agent.lateral_offset >= -current_width / 2.0:
            return None

        neighbor_ids = (
            current_lane.left_neighbors
            if action == LimSimAction.LCL
            else current_lane.right_neighbors
        )
        next_lane_id = self._choose_neighbor_lane(agent, neighbor_ids, map_)
        if next_lane_id is None:
            clipped_offset = np.clip(
                agent.lateral_offset, -current_width / 2.0, current_width / 2.0
            )
            return agent.with_updates(lateral_offset=float(clipped_offset))

        next_lane = map_.lanes[next_lane_id]
        next_centerline = next_lane.centerline()
        next_centerline = (
            np.asarray(next_centerline.coords, dtype=float) if next_centerline is not None else None
        )
        if next_centerline is None or len(next_centerline) < 2:
            return agent.with_updates(lane_id=next_lane_id)

        # Project the vehicle's current Cartesian position onto the neighbor
        # lane's centerline to get a continuous lateral offset in the new frame.
        line = LineString(next_centerline)
        next_progress = float(line.project(Point(agent.x, agent.y)))
        point = line.interpolate(next_progress)
        lookahead = line.interpolate(min(next_progress + 0.5, line.length))
        heading = spatial.normalize_angle(np.arctan2(lookahead.y - point.y, lookahead.x - point.x))

        # Signed distance from the neighbor centerline = continuous lateral offset.
        # This avoids the 3.6 m arithmetic jump that the old formula produced.
        dx = agent.x - point.x
        dy = agent.y - point.y
        next_offset = float(-dx * np.sin(heading) + dy * np.cos(heading))

        return agent.with_updates(
            lane_id=next_lane_id, route_progress=next_progress, lateral_offset=next_offset
        )

    def _choose_neighbor_lane(
        self, agent: AgentDecisionState, lane_ids, map_: Map
    ) -> Optional[str]:
        candidates = [lane_id for lane_id in lane_ids if lane_id in map_.lanes]
        if not candidates:
            return None
        point = Point(agent.x, agent.y)

        def _lane_centerline_distance(lid: str) -> float:
            centerline = map_.lanes[lid].centerline()
            centerline = (
                np.asarray(centerline.coords, dtype=float) if centerline is not None else None
            )
            return (
                LineString(centerline).distance(point) if centerline is not None else float("inf")
            )

        return min(candidates, key=_lane_centerline_distance)

    def _lane_width(self, lane) -> float:
        if lane.left_side is None or lane.right_side is None:
            return self.config.default_lane_width
        left = LineString(lane.left_side)
        right = LineString(lane.right_side)
        centerline = lane.centerline()
        centerline = np.asarray(centerline.coords, dtype=float) if centerline is not None else None
        if centerline is None or len(centerline) < 2:
            return self.config.default_lane_width
        line = LineString(centerline)
        samples = np.linspace(0.0, line.length, num=5)
        widths = []
        for progress in samples:
            point = line.interpolate(progress)
            widths.append(point.distance(left) + point.distance(right))
        return float(np.mean(widths)) if widths else self.config.default_lane_width

    def _next_lateral_offset(self, agent: AgentDecisionState, action: LimSimAction) -> float:
        if action == LimSimAction.LCL:
            return agent.lateral_offset + self.config.lateral_speed * self.config.dt
        if action == LimSimAction.LCR:
            return agent.lateral_offset - self.config.lateral_speed * self.config.dt
        return agent.lateral_offset

    def _select_route(
        self, agent: AgentDecisionState, action: LimSimAction, map_: Optional[Map]
    ) -> Tuple[Optional[LineString], Tuple[str, ...], float]:
        if map_ is None or agent.lane_id is None or agent.lane_id not in map_.lanes:
            return None, agent.route_lane_ids, agent.route_progress
        if abs(agent.lateral_offset) > self.config.max_lateral_offset_for_lane_rollout:
            return None, agent.route_lane_ids, agent.route_progress

        # --- use pre-extracted route when available ---
        if len(agent.route_lane_ids) > 1:
            try:
                idx = agent.route_lane_ids.index(agent.lane_id)
            except ValueError:
                idx = -1
            if idx >= 0:
                route_lanes = list(
                    agent.route_lane_ids[idx : idx + self.config.max_routes_per_agent]
                )
            else:
                # current lane not in route – fall back to topology
                route_lanes = [agent.lane_id]
                current_lane_id = route_lanes[0]
                while len(route_lanes) < self.config.max_routes_per_agent:
                    current_lane = map_.lanes.get(current_lane_id)
                    if current_lane is None or not current_lane.successors:
                        break
                    # prefer a successor that is in the route sequence
                    candidates = sorted(current_lane.successors)
                    picked = candidates[0]
                    for c in candidates:
                        if c in agent.route_lane_ids:
                            picked = c
                            break
                    if picked in route_lanes or picked not in map_.lanes:
                        break
                    route_lanes.append(picked)
                    current_lane_id = picked
        else:
            route_lanes = [agent.lane_id]
            current_lane_id = route_lanes[0]
            while len(route_lanes) < self.config.max_routes_per_agent:
                current_lane = map_.lanes.get(current_lane_id)
                if current_lane is None or not current_lane.successors:
                    break
                next_lane_id = sorted(current_lane.successors)[0]
                if next_lane_id in route_lanes or next_lane_id not in map_.lanes:
                    break
                route_lanes.append(next_lane_id)
                current_lane_id = next_lane_id

        centerlines = []
        for lane_id in route_lanes:
            centerline = map_.lanes[lane_id].centerline()
            centerlines.append(
                np.asarray(centerline.coords, dtype=float) if centerline is not None else None
            )
        path_array = polyline.concatenate(centerlines)
        if path_array is None or len(path_array) < 2:
            return None, tuple(route_lanes), 0.0

        route_path = LineString(path_array)
        start_progress = route_path.project(Point(agent.x, agent.y))
        return route_path, tuple(route_lanes), float(start_progress)


def is_action_valid(agent: AgentDecisionState, action: LimSimAction, map_: Optional[Map]) -> bool:
    """Check map-topology feasibility for a high-level action."""

    if action not in {LimSimAction.LCL, LimSimAction.LCR}:
        return True
    if map_ is None or agent.lane_id is None or agent.lane_id not in map_.lanes:
        return False
    lane = map_.lanes[agent.lane_id]
    if action == LimSimAction.LCL:
        return len(lane.left_neighbors) > 0 and SemanticMapQuery(map_).get_lane_change_permission(
            agent.lane_id, "left", s=agent.route_progress
        )
    if action == LimSimAction.LCR:
        return len(lane.right_neighbors) > 0 and SemanticMapQuery(map_).get_lane_change_permission(
            agent.lane_id, "right", s=agent.route_progress
        )
    return False
