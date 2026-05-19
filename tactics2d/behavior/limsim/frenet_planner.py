# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tactics2D-native Frenet-style trajectory planner for LimSim actions."""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import LineString, Point

from tactics2d.geometry import euclidean_distance, normalize_angle
from tactics2d.map.element import Map
from tactics2d.map.query import SemanticMapQuery, StopTarget
from tactics2d.routing.utils import concatenate_centerlines, get_lane_centerline

from .action import LimSimAction
from .config import LimSimConfig
from .geometry import footprint
from .planner import LaneFollower
from .schema import AgentDecisionState


@dataclass(frozen=True)
class FrenetPoint:
    """A point represented in reference-path Frenet coordinates."""

    s: float
    d: float


@dataclass
class FrenetCandidate:
    """One sampled trajectory and its scalar planning cost."""

    states: List[AgentDecisionState]
    cost: float
    target_speed: float
    target_lateral_offset: float


class QuinticPolynomial:
    """Quintic polynomial with position, velocity, and acceleration boundary values."""

    def __init__(self, x0: float, dx0: float, ddx0: float, x1: float, dx1: float, ddx1: float, t: float):
        self.coeffs = np.zeros(6)
        self.coeffs[0] = x0
        self.coeffs[1] = dx0
        self.coeffs[2] = 0.5 * ddx0
        matrix = np.array(
            [
                [t**3, t**4, t**5],
                [3 * t**2, 4 * t**3, 5 * t**4],
                [6 * t, 12 * t**2, 20 * t**3],
            ],
            dtype=float,
        )
        vector = np.array(
            [
                x1 - self.coeffs[0] - self.coeffs[1] * t - self.coeffs[2] * t**2,
                dx1 - self.coeffs[1] - 2 * self.coeffs[2] * t,
                ddx1 - 2 * self.coeffs[2],
            ],
            dtype=float,
        )
        self.coeffs[3:] = np.linalg.solve(matrix, vector)

    def calc(self, t: float, order: int = 0) -> float:
        if order == 0:
            powers = np.array([1.0, t, t**2, t**3, t**4, t**5])
            return float(np.dot(self.coeffs, powers))
        if order == 1:
            powers = np.array([0.0, 1.0, 2 * t, 3 * t**2, 4 * t**3, 5 * t**4])
            return float(np.dot(self.coeffs, powers))
        if order == 2:
            powers = np.array([0.0, 0.0, 2.0, 6 * t, 12 * t**2, 20 * t**3])
            return float(np.dot(self.coeffs, powers))
        powers = np.array([0.0, 0.0, 0.0, 6.0, 24 * t, 60 * t**2])
        return float(np.dot(self.coeffs, powers))


class QuarticPolynomial:
    """Quartic polynomial with initial position and terminal speed constraints."""

    def __init__(self, x0: float, dx0: float, ddx0: float, dx1: float, ddx1: float, t: float):
        self.coeffs = np.zeros(5)
        self.coeffs[0] = x0
        self.coeffs[1] = dx0
        self.coeffs[2] = 0.5 * ddx0
        matrix = np.array(
            [
                [3 * t**2, 4 * t**3],
                [6 * t, 12 * t**2],
            ],
            dtype=float,
        )
        vector = np.array(
            [
                dx1 - self.coeffs[1] - 2 * self.coeffs[2] * t,
                ddx1 - 2 * self.coeffs[2],
            ],
            dtype=float,
        )
        self.coeffs[3:] = np.linalg.solve(matrix, vector)

    def calc(self, t: float, order: int = 0) -> float:
        if order == 0:
            powers = np.array([1.0, t, t**2, t**3, t**4])
            return float(np.dot(self.coeffs, powers))
        if order == 1:
            powers = np.array([0.0, 1.0, 2 * t, 3 * t**2, 4 * t**3])
            return float(np.dot(self.coeffs, powers))
        if order == 2:
            powers = np.array([0.0, 0.0, 2.0, 6 * t, 12 * t**2])
            return float(np.dot(self.coeffs, powers))
        powers = np.array([0.0, 0.0, 0.0, 6.0, 24 * t])
        return float(np.dot(self.coeffs, powers))


class ReferencePath:
    """A lane-route reference path with Cartesian/Frenet conversion helpers."""

    def __init__(self, path: LineString, lane_ids: Tuple[str, ...], lane_width: float):
        self.path = path
        self.lane_ids = lane_ids
        self.lane_width = lane_width

    @classmethod
    def from_agent(
        cls, agent: AgentDecisionState, map_: Optional[Map], config: LimSimConfig
    ) -> Optional["ReferencePath"]:
        if map_ is None or agent.lane_id is None or agent.lane_id not in map_.lanes:
            return None

        route_lanes = [agent.lane_id]
        current_lane_id = agent.lane_id
        while len(route_lanes) < config.max_routes_per_agent:
            current_lane = map_.lanes.get(current_lane_id)
            if current_lane is None or not current_lane.successors:
                break
            next_lane_id = sorted(current_lane.successors)[0]
            if next_lane_id in route_lanes or next_lane_id not in map_.lanes:
                break
            route_lanes.append(next_lane_id)
            current_lane_id = next_lane_id

        centerlines = [get_lane_centerline(map_.lanes[lane_id]) for lane_id in route_lanes]
        path_array = concatenate_centerlines(centerlines)
        if path_array is None or len(path_array) < 2:
            return None
        path_array = _align_path_with_heading(path_array, agent.x, agent.y, agent.heading)

        return cls(
            LineString(path_array),
            tuple(route_lanes),
            lane_width=_lane_width(map_.lanes[agent.lane_id], config.default_lane_width),
        )

    def cartesian_to_frenet(self, x: float, y: float) -> FrenetPoint:
        point = Point(x, y)
        s = float(self.path.project(point))
        ref_point = self.path.interpolate(s)
        heading = self.heading_at(s)
        dx = x - ref_point.x
        dy = y - ref_point.y
        d = float(-dx * np.sin(heading) + dy * np.cos(heading))
        return FrenetPoint(s=s, d=d)

    def frenet_to_cartesian(self, s: float, d: float) -> Tuple[float, float, float]:
        s = float(np.clip(s, 0.0, self.path.length))
        point = self.path.interpolate(s)
        heading = self.heading_at(s)
        x = float(point.x - d * np.sin(heading))
        y = float(point.y + d * np.cos(heading))
        return x, y, heading

    def heading_at(self, s: float) -> float:
        s0 = float(np.clip(s, 0.0, self.path.length))
        ahead = self.path.interpolate(min(s0 + 0.5, self.path.length))
        behind = self.path.interpolate(max(s0 - 0.5, 0.0))
        return normalize_angle(np.arctan2(ahead.y - behind.y, ahead.x - behind.x))


class FrenetTrajectoryPlanner:
    """Sample and score Frenet-style trajectories for one selected LimSim action."""

    def __init__(self, config: LimSimConfig):
        self.config = config
        self.fallback = LaneFollower(config)

    def plan(
        self,
        agent: AgentDecisionState,
        action: LimSimAction,
        map_: Optional[Map],
        obstacle_trajectories: Sequence[Sequence[AgentDecisionState]] = (),
        time_ms: Optional[int] = None,
    ) -> List[AgentDecisionState]:
        """Generate a final trajectory for one agent after the behavior action is fixed."""

        reference_path = ReferencePath.from_agent(agent, map_, self.config)
        if reference_path is None or reference_path.path.length <= 1e-6:
            return self.fallback.rollout(agent, action, map_)
        start = reference_path.cartesian_to_frenet(agent.x, agent.y)
        if abs(start.d) > self.config.max_lateral_offset_for_lane_rollout:
            return self.fallback.rollout(agent, action, map_)

        candidates = self.sample_candidates(
            agent, action, reference_path, map_, obstacle_trajectories, time_ms=time_ms
        )
        if not candidates:
            return self.fallback.rollout(agent, action, map_)
        stop_target = self._nearest_required_stop_target(agent, reference_path, map_, time_ms=time_ms)
        if stop_target is not None:
            candidates.append(
                self._stop_target_candidate(
                    agent,
                    action,
                    reference_path,
                    start,
                    stop_target,
                    obstacle_trajectories,
                    map_,
                    time_ms,
                )
            )
        best_candidate = min(candidates, key=lambda candidate: candidate.cost)
        if obstacle_trajectories and self._has_collision(best_candidate.states, obstacle_trajectories):
            candidates.append(self._stop_candidate(agent, action, obstacle_trajectories, map_))
        return min(candidates, key=lambda candidate: candidate.cost).states

    def sample_candidates(
        self,
        agent: AgentDecisionState,
        action: LimSimAction,
        reference_path: ReferencePath,
        map_: Optional[Map],
        obstacle_trajectories: Sequence[Sequence[AgentDecisionState]] = (),
        time_ms: Optional[int] = None,
    ) -> List[FrenetCandidate]:
        start = reference_path.cartesian_to_frenet(agent.x, agent.y)
        if not self._lane_change_is_allowed(agent, action, map_, start.s):
            return []

        duration = max(self.config.dt, self.config.horizon_steps * self.config.dt)
        nominal_speed = np.clip(
            agent.speed + action.acceleration * duration,
            self.config.min_speed,
            self.config.max_speed,
        )
        nominal_d = self._target_lateral_offset(agent, action, map_, reference_path)

        candidates = []
        for speed_offset in self.config.frenet_target_speed_offsets:
            target_speed = float(np.clip(nominal_speed + speed_offset, self.config.min_speed, self.config.max_speed))
            for lateral_offset in self._sample_lateral_offsets(nominal_d):
                states, accel_cost, jerk_cost = self._build_states(
                    agent,
                    action,
                    reference_path,
                    start,
                    target_speed,
                    lateral_offset,
                    duration,
                    map_,
                )
                if not states:
                    continue
                cost = self._cost(
                    states,
                    target_speed,
                    nominal_d,
                    accel_cost,
                    jerk_cost,
                    obstacle_trajectories,
                    reference_path,
                    map_,
                    time_ms,
                )
                candidates.append(
                    FrenetCandidate(
                        states=states,
                        cost=cost,
                        target_speed=target_speed,
                        target_lateral_offset=lateral_offset,
                    )
                )
        return candidates

    def _build_states(
        self,
        agent: AgentDecisionState,
        action: LimSimAction,
        reference_path: ReferencePath,
        start: FrenetPoint,
        target_speed: float,
        target_d: float,
        duration: float,
        map_: Optional[Map],
    ) -> Tuple[List[AgentDecisionState], float, float]:
        longitudinal = QuarticPolynomial(start.s, agent.speed, action.acceleration, target_speed, 0.0, duration)
        lateral = QuinticPolynomial(start.d, 0.0, 0.0, target_d, 0.0, 0.0, duration)

        raw = []
        accel_cost = 0.0
        jerk_cost = 0.0
        previous_s = start.s
        for step in range(1, self.config.horizon_steps + 1):
            t = min(step * self.config.dt, duration)
            s = float(np.clip(longitudinal.calc(t), previous_s, reference_path.path.length))
            d = lateral.calc(t)
            x, y, heading = reference_path.frenet_to_cartesian(s, d)
            speed = max(longitudinal.calc(t, order=1), self.config.min_speed)
            accel_cost += longitudinal.calc(t, order=2) ** 2 + lateral.calc(t, order=2) ** 2
            jerk_cost += longitudinal.calc(t, order=3) ** 2 + lateral.calc(t, order=3) ** 2
            lane_id = self._lane_id_for_offset(agent, d, action, map_)
            raw.append((s, d, x, y, heading, speed, lane_id))
            previous_s = s

        states = []
        for index, item in enumerate(raw):
            s, d, x, y, heading, speed, lane_id = item
            if index + 1 < len(raw):
                nx, ny = raw[index + 1][2], raw[index + 1][3]
                heading = normalize_angle(np.arctan2(ny - y, nx - x))
            elif index > 0:
                px, py = raw[index - 1][2], raw[index - 1][3]
                heading = normalize_angle(np.arctan2(y - py, x - px))
            states.append(
                agent.with_updates(
                    x=x,
                    y=y,
                    heading=heading,
                    speed=float(np.clip(speed, self.config.min_speed, self.config.max_speed)),
                    lane_id=lane_id,
                    route_lane_ids=reference_path.lane_ids,
                    route_progress=s,
                    lateral_offset=float(d),
                    action=action,
                )
            )
        return states, accel_cost, jerk_cost

    def _target_lateral_offset(
        self,
        agent: AgentDecisionState,
        action: LimSimAction,
        map_: Optional[Map],
        reference_path: ReferencePath,
    ) -> float:
        if action not in {LimSimAction.LCL, LimSimAction.LCR}:
            return 0.0
        if map_ is None or agent.lane_id is None or agent.lane_id not in map_.lanes:
            return agent.lateral_offset

        lane = map_.lanes[agent.lane_id]
        neighbor_ids = lane.left_neighbors if action == LimSimAction.LCL else lane.right_neighbors
        if not neighbor_ids:
            return 0.0
        neighbor = map_.lanes.get(sorted(neighbor_ids)[0])
        neighbor_width = _lane_width(neighbor, self.config.default_lane_width) if neighbor is not None else reference_path.lane_width
        signed_distance = 0.5 * (reference_path.lane_width + neighbor_width)
        return signed_distance if action == LimSimAction.LCL else -signed_distance

    def _sample_lateral_offsets(self, nominal_d: float) -> List[float]:
        if abs(nominal_d) < 1e-6:
            return list(self.config.frenet_lateral_offsets)
        return [float(nominal_d + offset) for offset in self.config.frenet_lateral_offsets]

    def _lane_id_for_offset(
        self,
        agent: AgentDecisionState,
        d: float,
        action: LimSimAction,
        map_: Optional[Map],
    ) -> Optional[str]:
        if action not in {LimSimAction.LCL, LimSimAction.LCR}:
            return agent.lane_id
        if map_ is None or agent.lane_id is None or agent.lane_id not in map_.lanes:
            return agent.lane_id
        lane = map_.lanes[agent.lane_id]
        width = _lane_width(lane, self.config.default_lane_width)
        if action == LimSimAction.LCL and d > width / 2.0 and lane.left_neighbors:
            return sorted(lane.left_neighbors)[0]
        if action == LimSimAction.LCR and d < -width / 2.0 and lane.right_neighbors:
            return sorted(lane.right_neighbors)[0]
        return agent.lane_id

    def _cost(
        self,
        states: Sequence[AgentDecisionState],
        target_speed: float,
        nominal_d: float,
        accel_cost: float,
        jerk_cost: float,
        obstacle_trajectories: Sequence[Sequence[AgentDecisionState]],
        reference_path: Optional[ReferencePath] = None,
        map_: Optional[Map] = None,
        time_ms: Optional[int] = None,
    ) -> float:
        cost = 0.0
        cost += self.config.frenet_accel_weight * accel_cost
        cost += self.config.frenet_jerk_weight * jerk_cost
        cost += self.config.frenet_speed_weight * sum((state.speed - target_speed) ** 2 for state in states)
        cost += self.config.frenet_lateral_weight * sum((state.lateral_offset - nominal_d) ** 2 for state in states)

        for step, state in enumerate(states):
            ego_shape = footprint(state)
            for obstacle in obstacle_trajectories:
                if not obstacle:
                    continue
                other = obstacle[min(step, len(obstacle) - 1)]
                distance = euclidean_distance(state.location, other.location)
                if ego_shape.intersects(footprint(other)):
                    cost += self.config.frenet_collision_penalty
                elif distance < self.config.frenet_obstacle_buffer:
                    cost += self.config.frenet_proximity_weight * (
                        self.config.frenet_obstacle_buffer - distance
                    ) ** 2
        if reference_path is not None and map_ is not None:
            cost += self._stop_rule_cost(states, reference_path, map_, time_ms=time_ms)
            cost += self._junction_conflict_cost(states, reference_path, map_, obstacle_trajectories)
        return float(cost)

    def _stop_candidate(
        self,
        agent: AgentDecisionState,
        action: LimSimAction,
        obstacle_trajectories: Sequence[Sequence[AgentDecisionState]],
        map_: Optional[Map] = None,
        time_ms: Optional[int] = None,
    ) -> FrenetCandidate:
        states = [
            agent.with_updates(speed=0.0, action=action)
            for _ in range(self.config.horizon_steps)
        ]
        cost = self._cost(
            states,
            target_speed=0.0,
            nominal_d=agent.lateral_offset,
            accel_cost=0.0,
            jerk_cost=0.0,
            obstacle_trajectories=obstacle_trajectories,
            map_=map_,
            time_ms=time_ms,
        )
        return FrenetCandidate(
            states=states,
            cost=cost,
            target_speed=0.0,
            target_lateral_offset=agent.lateral_offset,
        )

    def _has_collision(
        self,
        states: Sequence[AgentDecisionState],
        obstacle_trajectories: Sequence[Sequence[AgentDecisionState]],
    ) -> bool:
        for step, state in enumerate(states):
            ego_shape = footprint(state)
            for obstacle in obstacle_trajectories:
                if not obstacle:
                    continue
                other = obstacle[min(step, len(obstacle) - 1)]
                if ego_shape.intersects(footprint(other)):
                    return True
        return False

    def _lane_change_is_allowed(
        self,
        agent: AgentDecisionState,
        action: LimSimAction,
        map_: Optional[Map],
        s: float,
    ) -> bool:
        if action not in {LimSimAction.LCL, LimSimAction.LCR}:
            return True
        if map_ is None or agent.lane_id is None:
            return False
        direction = "left" if action == LimSimAction.LCL else "right"
        return SemanticMapQuery(map_).get_lane_change_permission(agent.lane_id, direction, s=s)

    def _nearest_required_stop_target(
        self,
        agent: AgentDecisionState,
        reference_path: ReferencePath,
        map_: Optional[Map],
        time_ms: Optional[int] = None,
    ) -> Optional[Tuple[StopTarget, float]]:
        if map_ is None:
            return None
        query = SemanticMapQuery(map_)
        start = reference_path.cartesian_to_frenet(agent.x, agent.y)
        max_s = min(
            reference_path.path.length,
            start.s + max(agent.speed * self.config.horizon_steps * self.config.dt, 5.0),
        )

        candidates = []
        for lane_id in reference_path.lane_ids:
            for target in query.get_stop_targets(lane_id, time_ms=time_ms):
                if not self._target_requires_stop(target):
                    continue
                target_s = float(reference_path.path.project(target.point))
                if start.s < target_s <= max_s + self.config.frenet_stop_distance_buffer:
                    candidates.append((target_s, target))
        if not candidates:
            return None
        target_s, target = min(candidates, key=lambda item: item[0])
        return target, target_s

    def _target_requires_stop(self, target: StopTarget) -> bool:
        if target.reason == "stop_sign":
            return True
        if target.reason != "traffic_light":
            return False
        state = (target.state or "").upper()
        return any(stop_state.upper() in state for stop_state in self.config.traffic_light_stop_states)

    def _stop_target_candidate(
        self,
        agent: AgentDecisionState,
        action: LimSimAction,
        reference_path: ReferencePath,
        start: FrenetPoint,
        stop_target_info: Tuple[StopTarget, float],
        obstacle_trajectories: Sequence[Sequence[AgentDecisionState]],
        map_: Optional[Map],
        time_ms: Optional[int] = None,
    ) -> FrenetCandidate:
        _, stop_s = stop_target_info
        target_s = max(start.s, stop_s - self.config.frenet_stop_distance_buffer)
        distance_to_stop = max(target_s - start.s, 0.0)
        if distance_to_stop <= 1e-6:
            deceleration = self.config.frenet_stop_deceleration
        else:
            deceleration = min(
                self.config.frenet_stop_deceleration,
                max(agent.speed**2 / (2.0 * distance_to_stop), 0.0),
            )

        states = []
        current_s = start.s
        previous_speed = agent.speed
        accel_cost = 0.0
        for _ in range(1, self.config.horizon_steps + 1):
            speed = max(previous_speed - deceleration * self.config.dt, 0.0)
            travel = 0.5 * (previous_speed + speed) * self.config.dt
            current_s = min(current_s + travel, target_s)
            if current_s >= target_s - 1e-6:
                speed = 0.0
            x, y, heading = reference_path.frenet_to_cartesian(current_s, 0.0)
            states.append(
                agent.with_updates(
                    x=x,
                    y=y,
                    heading=heading,
                    speed=speed,
                    lane_id=agent.lane_id,
                    route_lane_ids=reference_path.lane_ids,
                    route_progress=current_s,
                    lateral_offset=0.0,
                    action=action,
                )
            )
            accel_cost += deceleration**2
            previous_speed = speed

        cost = self._cost(
            states,
            target_speed=0.0,
            nominal_d=0.0,
            accel_cost=accel_cost,
            jerk_cost=0.0,
            obstacle_trajectories=obstacle_trajectories,
            reference_path=reference_path,
            map_=map_,
            time_ms=time_ms,
        )
        return FrenetCandidate(states=states, cost=cost, target_speed=0.0, target_lateral_offset=0.0)

    def _stop_rule_cost(
        self,
        states: Sequence[AgentDecisionState],
        reference_path: ReferencePath,
        map_: Map,
        time_ms: Optional[int] = None,
    ) -> float:
        stop_target_info = (
            self._nearest_required_stop_target(states[0], reference_path, map_, time_ms=time_ms)
            if states
            else None
        )
        if stop_target_info is None:
            return 0.0
        _, stop_s = stop_target_info
        stop_s = max(0.0, stop_s - self.config.frenet_stop_distance_buffer)
        cost = 0.0
        for state in states:
            frenet = reference_path.cartesian_to_frenet(state.x, state.y)
            if frenet.s >= stop_s and state.speed > self.config.frenet_stop_speed_threshold:
                cost += self.config.frenet_stop_line_penalty * (
                    1.0 + state.speed - self.config.frenet_stop_speed_threshold
                )
        return cost

    def _junction_conflict_cost(
        self,
        states: Sequence[AgentDecisionState],
        reference_path: ReferencePath,
        map_: Map,
        obstacle_trajectories: Sequence[Sequence[AgentDecisionState]],
    ) -> float:
        if not states or not obstacle_trajectories:
            return 0.0
        query = SemanticMapQuery(map_)
        lane_ids = set(reference_path.lane_ids)
        cost = 0.0
        for obstacle in obstacle_trajectories:
            if not obstacle:
                continue
            obstacle_lane_id = next((state.lane_id for state in obstacle if state.lane_id), None)
            if obstacle_lane_id is None:
                continue
            conflict_points = []
            for lane_id in lane_ids:
                conflict_points.extend(query.get_conflict_points(lane_id, obstacle_lane_id))
            for point in conflict_points:
                ego_step = self._first_step_near_point(states, point)
                other_step = self._first_step_near_point(obstacle, point)
                if ego_step is None or other_step is None:
                    continue
                time_gap = abs(ego_step - other_step) * self.config.dt
                if time_gap <= self.config.frenet_junction_conflict_time_window:
                    cost += self.config.frenet_junction_conflict_penalty * (
                        self.config.frenet_junction_conflict_time_window - time_gap + self.config.dt
                    )
        return cost

    def _first_step_near_point(self, states: Sequence[AgentDecisionState], point: Point) -> Optional[int]:
        for step, state in enumerate(states):
            if euclidean_distance(state.location, (point.x, point.y)) <= self.config.frenet_junction_conflict_distance:
                return step
        return None


def _lane_width(lane, default_width: float) -> float:
    if lane is None:
        return default_width
    width = lane.get_width(samples=5, default=default_width)
    return float(width) if width is not None else default_width


def _align_path_with_heading(path_array: np.ndarray, x: float, y: float, heading: float) -> np.ndarray:
    line = LineString(path_array)
    progress = float(line.project(Point(x, y)))
    point = line.interpolate(progress)
    ahead = line.interpolate(min(progress + 0.5, line.length))
    if ahead.distance(point) < 1e-6:
        ahead = point
        point = line.interpolate(max(progress - 0.5, 0.0))
    path_heading = normalize_angle(np.arctan2(ahead.y - point.y, ahead.x - point.x))
    if np.cos(normalize_angle(path_heading - heading)) < 0.0:
        return path_array[::-1].copy()
    return path_array
