# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tactics2D-native Frenet-style trajectory planner for LimSim actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import substring

from tactics2d.geometry import frenet, spatial
from tactics2d.map.element import Map
from tactics2d.map.query import SemanticMapQuery, StopTarget

from .action import LimSimAction
from .config import LimSimConfig
from .lane_follower import (
    LaneFollower,
    _continuous_route_lanes,
    _route_path_from_lanes,
    route_has_continuation,
    route_lanes_from_agent,
)
from .schema import AgentDecisionState, DecisionStep, decision_sequence_covers_horizon


@dataclass
class FrenetCandidate:
    """One sampled trajectory and its scalar planning cost."""

    states: list[AgentDecisionState]
    cost: float


class _QuinticPolynomial:
    """Quintic polynomial with position, velocity, and acceleration boundary values."""

    def __init__(
        self, x0: float, dx0: float, ddx0: float, x1: float, dx1: float, ddx1: float, t: float
    ):
        self.coeffs = np.zeros(6)
        self.coeffs[0] = x0
        self.coeffs[1] = dx0
        self.coeffs[2] = 0.5 * ddx0
        matrix = np.array(
            [[t**3, t**4, t**5], [3 * t**2, 4 * t**3, 5 * t**4], [6 * t, 12 * t**2, 20 * t**3]],
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

    def calculate(self, t: float, order: int = 0) -> float:
        c0, c1, c2, c3, c4, c5 = self.coeffs
        if order == 0:
            return float(c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * c5)))))
        if order == 1:
            return float(c1 + t * (2.0 * c2 + t * (3.0 * c3 + t * (4.0 * c4 + t * 5.0 * c5))))
        if order == 2:
            return float(2.0 * c2 + t * (6.0 * c3 + t * (12.0 * c4 + t * 20.0 * c5)))
        return float(6.0 * c3 + t * (24.0 * c4 + t * 60.0 * c5))


class _QuarticPolynomial:
    """Quartic polynomial with initial position and terminal speed constraints."""

    def __init__(self, x0: float, dx0: float, ddx0: float, dx1: float, ddx1: float, t: float):
        self.coeffs = np.zeros(5)
        self.coeffs[0] = x0
        self.coeffs[1] = dx0
        self.coeffs[2] = 0.5 * ddx0
        matrix = np.array([[3 * t**2, 4 * t**3], [6 * t, 12 * t**2]], dtype=float)
        vector = np.array(
            [dx1 - self.coeffs[1] - 2 * self.coeffs[2] * t, ddx1 - 2 * self.coeffs[2]], dtype=float
        )
        self.coeffs[3:] = np.linalg.solve(matrix, vector)

    def calculate(self, t: float, order: int = 0) -> float:
        c0, c1, c2, c3, c4 = self.coeffs
        if order == 0:
            return float(c0 + t * (c1 + t * (c2 + t * (c3 + t * c4))))
        if order == 1:
            return float(c1 + t * (2.0 * c2 + t * (3.0 * c3 + t * 4.0 * c4)))
        if order == 2:
            return float(2.0 * c2 + t * (6.0 * c3 + t * 12.0 * c4))
        return float(6.0 * c3 + t * 24.0 * c4)


class FrenetTrajectoryPlanner:
    """Sample and score Frenet-style trajectories for one selected LimSim action."""

    def __init__(self, config: LimSimConfig):
        self.config = config
        self.fallback = LaneFollower(config)
        self._ref_path_cache = {}

    def plan(
        self,
        agent: AgentDecisionState,
        action: LimSimAction,
        map_: Map | None,
        obstacle_trajectories: Sequence[Sequence[AgentDecisionState]] = (),
        time_ms: int | None = None,
        decision_sequence: Sequence[DecisionStep] = (),
    ) -> list[AgentDecisionState]:
        """Generate a final trajectory for one agent after the behavior action is fixed."""

        current_frame = int(time_ms or 0)
        if decision_sequence_covers_horizon(
            decision_sequence, current_frame, self.config.horizon_steps, self.config.step_ms
        ):
            return self._plan_decision_sequence(
                agent, action, decision_sequence, map_, obstacle_trajectories, time_ms
            )
        return self._plan_action(
            agent, action, map_, obstacle_trajectories, time_ms, steps=self.config.horizon_steps
        )

    def _plan_decision_sequence(
        self,
        agent: AgentDecisionState,
        fallback_action: LimSimAction,
        decision_sequence: Sequence[DecisionStep],
        map_: Map | None,
        obstacle_trajectories: Sequence[Sequence[AgentDecisionState]],
        time_ms: int | None,
    ) -> list[AgentDecisionState]:
        """Generate piecewise trajectories toward the remaining MCTS targets."""

        current_frame = int(time_ms or 0)
        step_ms = int(round(self.config.dt * 1000))
        horizon_end = current_frame + self.config.horizon_steps * step_ms
        remaining = [
            decision for decision in decision_sequence if decision.expected_frame > current_frame
        ]
        if not remaining:
            return self._plan_action(
                agent,
                fallback_action,
                map_,
                obstacle_trajectories,
                time_ms,
                steps=self.config.horizon_steps,
            )

        # The official generator skips an intermediate target when the next
        # decision keeps the same action, while retaining the full cached list.
        targets = [
            decision
            for index, decision in enumerate(remaining)
            if index + 1 == len(remaining) or remaining[index + 1].action != decision.action
        ]

        planned = []
        current = agent
        for decision in targets:
            segment_end = min(decision.expected_frame, horizon_end)
            segment_steps = int(round((segment_end - current_frame) / step_ms))
            if segment_steps <= 0:
                continue
            target_state = (
                decision.expected_state if segment_end == decision.expected_frame else None
            )
            segment_obstacles = self._slice_obstacles(obstacle_trajectories, len(planned))
            segment = self._plan_action(
                current,
                decision.action,
                map_,
                segment_obstacles,
                current_frame,
                steps=segment_steps,
                target_state=target_state,
            )
            if not segment:
                break
            planned.extend(segment[:segment_steps])
            current = planned[-1]
            completed_steps = min(len(segment), segment_steps)
            current_frame += completed_steps * step_ms
            if completed_steps < segment_steps or current_frame >= horizon_end:
                break
        return planned

    def _plan_action(
        self,
        agent: AgentDecisionState,
        action: LimSimAction,
        map_: Map | None,
        obstacle_trajectories: Sequence[Sequence[AgentDecisionState]],
        time_ms: int | None,
        steps: int,
        target_state: AgentDecisionState | None = None,
    ) -> list[AgentDecisionState]:
        """Plan one action segment with an optional MCTS endpoint target."""

        steps = max(1, int(steps))

        reference_path = reference_path_from_agent(
            agent, map_, self.config, cache=self._ref_path_cache
        )
        if reference_path is None or reference_path.path.length <= 1e-6:
            return self.fallback.rollout(agent, action, map_, steps=steps)
        start = reference_path.cartesian_to_frenet(
            agent.x, agent.y, hint_s=reference_path.initial_s
        )
        if abs(start.d) > self.config.max_lateral_offset_for_lane_rollout:
            return self.fallback.rollout(agent, action, map_, steps=steps)

        # --- pre-compute map queries shared across all candidates ---
        stop_target_info = self._nearest_required_stop_target(
            agent, reference_path, map_, time_ms=time_ms, steps=steps
        )
        conflict_points_cache = self._build_conflict_cache(
            reference_path, map_, obstacle_trajectories
        )

        # --- pre-compute obstacle footprints once for all candidates ---
        obstacle_footprints = self._precompute_obstacle_footprints(obstacle_trajectories)

        candidates = self.sample_candidates(
            agent,
            action,
            reference_path,
            map_,
            obstacle_trajectories,
            time_ms=time_ms,
            stop_target_info=stop_target_info,
            conflict_points_cache=conflict_points_cache,
            obstacle_footprints=obstacle_footprints,
            start=start,
            steps=steps,
            target_state=target_state,
        )
        if not candidates:
            return self.fallback.rollout(agent, action, map_, steps=steps)
        if stop_target_info is not None:
            candidates.append(
                self._stop_target_candidate(
                    agent,
                    action,
                    reference_path,
                    start,
                    stop_target_info,
                    obstacle_trajectories,
                    map_,
                    time_ms,
                    conflict_points_cache=conflict_points_cache,
                    obstacle_footprints=obstacle_footprints,
                    steps=steps,
                )
            )
        feasible_candidates = [candidate for candidate in candidates if np.isfinite(candidate.cost)]
        if not feasible_candidates and not action.is_lane_change:
            nudge_candidates = self.sample_candidates(
                agent,
                action,
                reference_path,
                map_,
                obstacle_trajectories,
                time_ms=time_ms,
                stop_target_info=stop_target_info,
                conflict_points_cache=conflict_points_cache,
                obstacle_footprints=obstacle_footprints,
                start=start,
                steps=steps,
                target_state=target_state,
                lateral_offsets=self._nudge_lateral_offsets(reference_path.lane_width),
            )
            feasible_candidates = [
                candidate for candidate in nudge_candidates if np.isfinite(candidate.cost)
            ]
        if not feasible_candidates:
            return self._stop_candidate(
                agent,
                action,
                reference_path,
                start,
                obstacle_trajectories,
                map_,
                time_ms,
                stop_target_info=stop_target_info,
                conflict_points_cache=conflict_points_cache,
                obstacle_footprints=obstacle_footprints,
                steps=steps,
            ).states
        return min(feasible_candidates, key=lambda candidate: candidate.cost).states

    def _slice_obstacles(
        self, obstacle_trajectories: Sequence[Sequence[AgentDecisionState]], offset: int
    ) -> list[Sequence[AgentDecisionState]]:
        """Align obstacle predictions with a later piecewise planning segment."""

        return [
            trajectory[offset:] for trajectory in obstacle_trajectories if len(trajectory) > offset
        ]

    def _precompute_obstacle_footprints(
        self, obstacle_trajectories: Sequence[Sequence[AgentDecisionState]]
    ) -> list[list[object]]:
        """Pre-compute Shapely polygons for every obstacle at every step.

        Returns a list parallel to *obstacle_trajectories*, where each element
        is a list of ``_footprint(state)`` results (one per step).
        """
        footprints: list[list[object]] = []
        for obs in obstacle_trajectories:
            if not obs:
                footprints.append([])
            else:
                footprints.append([_footprint(state) for state in obs])
        return footprints

    def _build_conflict_cache(
        self,
        reference_path: ReferencePath,
        map_: Map | None,
        obstacle_trajectories: Sequence[Sequence[AgentDecisionState]],
    ) -> dict[tuple[str, str], list[Point]]:
        """Pre-compute conflict points for all (ref_lane, obs_lane) pairs."""
        cache: dict[tuple[str, str], list[Point]] = {}
        if map_ is None or not obstacle_trajectories:
            return cache
        query = SemanticMapQuery(map_)
        ref_lane_ids = set(reference_path.lane_ids)
        for obstacle in obstacle_trajectories:
            if not obstacle:
                continue
            obs_lane_id = next((s.lane_id for s in obstacle if s.lane_id), None)
            if obs_lane_id is None:
                continue
            for lane_id in ref_lane_ids:
                key = (lane_id, obs_lane_id)
                if key not in cache:
                    cache[key] = query.get_conflict_points(lane_id, obs_lane_id)
        return cache

    def sample_candidates(
        self,
        agent: AgentDecisionState,
        action: LimSimAction,
        reference_path: ReferencePath,
        map_: Map | None,
        obstacle_trajectories: Sequence[Sequence[AgentDecisionState]] = (),
        time_ms: int | None = None,
        stop_target_info: tuple[StopTarget, float] | None = None,
        conflict_points_cache: dict[tuple[str, str], list[Point]] | None = None,
        obstacle_footprints: list[list[object]] | None = None,
        start: frenet.FrenetPoint | None = None,
        steps: int | None = None,
        target_state: AgentDecisionState | None = None,
        lateral_offsets: Sequence[float] | None = None,
    ) -> list[FrenetCandidate]:
        if start is None:
            start = reference_path.cartesian_to_frenet(
                agent.x, agent.y, hint_s=reference_path.initial_s
            )
        if not self._lane_change_is_allowed(agent, action, map_, start.s):
            return []

        planning_steps = self.config.horizon_steps if steps is None else max(1, int(steps))
        duration = max(self.config.dt, planning_steps * self.config.dt)
        desired_speed = float(
            np.clip(agent.target_speed, self.config.min_speed, self.config.max_speed)
        )
        if target_state is None:
            nominal_d = self._target_lateral_offset(agent, action, map_, reference_path)
        else:
            target = reference_path.cartesian_to_frenet(
                target_state.x, target_state.y, hint_s=max(start.s, target_state.route_progress)
            )
            nominal_d = target.d

        candidates = []
        for sampled_speed in self._sample_target_speeds(agent, target_state, map_):
            for lateral_offset in self._sample_lateral_offsets(
                nominal_d,
                has_decision_target=target_state is not None,
                explicit_offsets=lateral_offsets,
            ):
                states, accel_cost, jerk_cost = self._build_states(
                    agent,
                    action,
                    reference_path,
                    start,
                    sampled_speed,
                    lateral_offset,
                    duration,
                    map_,
                    steps=planning_steps,
                )
                if not states:
                    continue
                cost = self._cost(
                    states,
                    desired_speed,
                    nominal_d,
                    accel_cost,
                    jerk_cost,
                    obstacle_trajectories,
                    reference_path,
                    map_,
                    time_ms,
                    stop_target_info=stop_target_info,
                    conflict_points_cache=conflict_points_cache,
                    obstacle_footprints=obstacle_footprints,
                )
                candidates.append(FrenetCandidate(states=states, cost=cost))
        return candidates

    def _sample_target_speeds(
        self,
        agent: AgentDecisionState,
        target_state: AgentDecisionState | None,
        map_: Map | None,
    ) -> list[float]:
        """Return LimSim's lane-keeping or decision endpoint speed samples."""

        speed_limit = self.config.max_speed
        if map_ is not None and agent.lane_id is not None:
            speed_limit = map_.get_speed_limit(agent.lane_id, default=speed_limit)
        speed_limit = float(np.clip(speed_limit, self.config.min_speed, self.config.max_speed))
        desired_speed = float(
            np.clip(agent.target_speed, self.config.min_speed, self.config.max_speed)
        )

        if self.config.frenet_target_speed_offsets:
            center_speed = target_state.speed if target_state is not None else desired_speed
            return sorted(
                {
                    float(
                        np.clip(
                            center_speed + offset,
                            self.config.min_speed,
                            speed_limit,
                        )
                    )
                    for offset in self.config.frenet_target_speed_offsets
                }
            )

        speed_step = 2.5 / 3.6
        if target_state is not None:
            lower = max(1e-9, target_state.speed - 3.0 * speed_step)
            upper = min(target_state.speed + speed_step, speed_limit)
            sample_count = 10
        else:
            lower = max(1e-9, agent.speed - 2.0 * speed_step)
            upper = min(
                max(agent.speed, desired_speed) + 2.0 * speed_step * 1.01,
                speed_limit,
            )
            sample_count = 5
        return [float(speed) for speed in np.linspace(lower, upper, sample_count)]

    def _build_states(
        self,
        agent: AgentDecisionState,
        action: LimSimAction,
        reference_path: ReferencePath,
        start: FrenetPoint,
        target_speed: float,
        target_d: float,
        duration: float,
        map_: Map | None,
        steps: int | None = None,
    ) -> tuple[list[AgentDecisionState], float, float]:
        longitudinal = _QuarticPolynomial(
            start.s,
            agent.speed,
            self._action_acceleration(action),
            target_speed,
            0.0,
            duration,
        )
        lateral = _QuinticPolynomial(start.d, 0.0, 0.0, target_d, 0.0, 0.0, duration)

        raw = []
        accel_cost = 0.0
        jerk_cost = 0.0
        previous_s = start.s
        planning_steps = self.config.horizon_steps if steps is None else max(1, int(steps))
        for step in range(1, planning_steps + 1):
            t = min(step * self.config.dt, duration)
            # A recorded route can be shorter than the planning horizon. Stop
            # this rollout at a true route endpoint instead of stacking states.
            s = float(max(longitudinal.calculate(t), previous_s))
            if reference_path.terminal and s > reference_path.path.length:
                break
            d = lateral.calculate(t)
            x, y, heading = reference_path.frenet_to_cartesian(s, d)
            speed = max(longitudinal.calculate(t, order=1), self.config.min_speed)
            accel_cost += (
                longitudinal.calculate(t, order=2) ** 2 + lateral.calculate(t, order=2) ** 2
            )
            jerk_cost += (
                longitudinal.calculate(t, order=3) ** 2 + lateral.calculate(t, order=3) ** 2
            )
            lane_id = self._lane_id_for_offset(agent, d, action, map_)
            raw.append((s, d, x, y, heading, speed, lane_id))
            previous_s = s

        states = []
        previous_heading = spatial.normalize_angle(agent.heading)
        for index, item in enumerate(raw):
            s, d, x, y, heading, speed, lane_id = item
            if index + 1 < len(raw):
                nx, ny = raw[index + 1][2], raw[index + 1][3]
                if spatial.euclidean_distance((x, y), (nx, ny)) > 1e-4:
                    heading = spatial.normalize_angle(np.arctan2(ny - y, nx - x))
                else:
                    heading = previous_heading
            elif index > 0:
                px, py = raw[index - 1][2], raw[index - 1][3]
                if spatial.euclidean_distance((px, py), (x, y)) > 1e-4:
                    heading = spatial.normalize_angle(np.arctan2(y - py, x - px))
                else:
                    heading = previous_heading
            else:
                heading = spatial.normalize_angle(heading)
            previous_heading = heading
            states.append(
                agent.with_updates(
                    x=x,
                    y=y,
                    heading=heading,
                    speed=float(np.clip(speed, self.config.min_speed, self.config.max_speed)),
                    lane_id=lane_id,
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
        map_: Map | None,
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
        neighbor_width = (
            _lane_width(neighbor, self.config.default_lane_width)
            if neighbor is not None
            else reference_path.lane_width
        )
        signed_distance = 0.5 * (reference_path.lane_width + neighbor_width)
        return signed_distance if action == LimSimAction.LCL else -signed_distance

    def _sample_lateral_offsets(
        self,
        nominal_d: float,
        has_decision_target: bool = False,
        explicit_offsets: Sequence[float] | None = None,
    ) -> list[float]:
        if explicit_offsets is not None:
            return [float(offset) for offset in explicit_offsets]
        if self.config.frenet_lateral_offsets:
            return [float(nominal_d + offset) for offset in self.config.frenet_lateral_offsets]
        if has_decision_target:
            return [float(offset) for offset in np.linspace(nominal_d - 0.75, nominal_d + 0.75, 5)]
        return [float(nominal_d)]

    @staticmethod
    def _nudge_lateral_offsets(lane_width: float) -> list[float]:
        sample_count = max(2, int(lane_width / 0.75) + 1)
        return [
            float(offset)
            for offset in np.linspace(-lane_width / 3.0, lane_width / 3.0, sample_count)
            if abs(offset) > 1e-9
        ]

    def _action_acceleration(self, action: LimSimAction) -> float:
        if action == LimSimAction.AC:
            return self.config.acceleration
        if action == LimSimAction.DC:
            return self.config.deceleration
        return 0.0

    def _lane_id_for_offset(
        self, agent: AgentDecisionState, d: float, action: LimSimAction, map_: Map | None
    ) -> str | None:
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
        reference_path: ReferencePath | None = None,
        map_: Map | None = None,
        time_ms: int | None = None,
        stop_target_info: tuple[StopTarget, float] | None = None,
        conflict_points_cache: dict[tuple[str, str], list[Point]] | None = None,
        obstacle_footprints: list[list[object]] | None = None,
    ) -> float:
        cost = 0.0
        cost += self.config.frenet_accel_weight * accel_cost
        cost += self.config.frenet_jerk_weight * jerk_cost
        cost += self.config.frenet_speed_weight * sum(
            (state.speed - target_speed) ** 2 for state in states
        )
        cost += self.config.frenet_lateral_weight * sum(
            (state.lateral_offset - nominal_d) ** 2 for state in states
        )

        ego_radius = (
            0.5 * ((states[0].length * 1.5) ** 2 + (states[0].width * 1.1) ** 2) ** 0.5
            if states
            else 0.0
        )
        for step, state in enumerate(states):
            ego_shape = _footprint(
                state,
                length=state.length * 1.5,
                width=state.width * 1.1,
            )
            for obs_idx, obstacle in enumerate(obstacle_trajectories):
                if step >= len(obstacle):
                    continue
                other = obstacle[step]
                distance = spatial.euclidean_distance(state.location, other.location)
                other_radius = 0.5 * (other.length**2 + other.width**2) ** 0.5
                if distance > ego_radius + other_radius:
                    # Bounding circles do not overlap, so collision is impossible.
                    if distance < self.config.frenet_obstacle_buffer:
                        cost += (
                            self.config.frenet_proximity_weight
                            * (self.config.frenet_obstacle_buffer - distance) ** 2
                        )
                    continue
                # use pre-computed obstacle footprint when available
                other_shape = (
                    obstacle_footprints[obs_idx][step]
                    if obstacle_footprints is not None
                    and obs_idx < len(obstacle_footprints)
                    and step < len(obstacle_footprints[obs_idx])
                    else _footprint(other)
                )
                if ego_shape.intersects(other_shape):
                    return float("inf")
                elif distance < self.config.frenet_obstacle_buffer:
                    cost += (
                        self.config.frenet_proximity_weight
                        * (self.config.frenet_obstacle_buffer - distance) ** 2
                    )
        if reference_path is not None and map_ is not None:
            cost += self._stop_rule_cost(
                states, reference_path, map_, time_ms=time_ms, stop_target_info=stop_target_info
            )
            cost += self._junction_conflict_cost(
                states,
                reference_path,
                map_,
                obstacle_trajectories,
                conflict_points_cache=conflict_points_cache,
            )
        return float(cost)

    def _stop_candidate(
        self,
        agent: AgentDecisionState,
        action: LimSimAction,
        reference_path: ReferencePath,
        start: FrenetPoint,
        obstacle_trajectories: Sequence[Sequence[AgentDecisionState]],
        map_: Map | None = None,
        time_ms: int | None = None,
        stop_target_info: tuple[StopTarget, float] | None = None,
        conflict_points_cache: dict[tuple[str, str], list[Point]] | None = None,
        obstacle_footprints: list[list[object]] | None = None,
        steps: int | None = None,
    ) -> FrenetCandidate:
        states, accel_cost, jerk_cost = self._build_deceleration_states(
            agent, action, reference_path, start, steps=steps
        )
        cost = self._cost(
            states,
            target_speed=0.0,
            nominal_d=agent.lateral_offset,
            accel_cost=accel_cost,
            jerk_cost=jerk_cost,
            obstacle_trajectories=obstacle_trajectories,
            reference_path=reference_path,
            map_=map_,
            time_ms=time_ms,
            stop_target_info=stop_target_info,
            conflict_points_cache=conflict_points_cache,
            obstacle_footprints=obstacle_footprints,
        )
        return FrenetCandidate(states=states, cost=cost)

    def _lane_change_is_allowed(
        self, agent: AgentDecisionState, action: LimSimAction, map_: Map | None, s: float
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
        map_: Map | None,
        time_ms: int | None = None,
        steps: int | None = None,
    ) -> tuple[StopTarget, float] | None:
        if map_ is None:
            return None
        query = SemanticMapQuery(map_)
        start = reference_path.cartesian_to_frenet(
            agent.x, agent.y, hint_s=reference_path.initial_s
        )
        planning_steps = self.config.horizon_steps if steps is None else max(1, int(steps))
        max_s = min(
            reference_path.path.length,
            start.s + max(agent.speed * planning_steps * self.config.dt, 5.0),
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
        return any(
            stop_state.upper() in state for stop_state in self.config.traffic_light_stop_states
        )

    def _stop_target_candidate(
        self,
        agent: AgentDecisionState,
        action: LimSimAction,
        reference_path: ReferencePath,
        start: FrenetPoint,
        stop_target_info: tuple[StopTarget, float],
        obstacle_trajectories: Sequence[Sequence[AgentDecisionState]],
        map_: Map | None,
        time_ms: int | None = None,
        conflict_points_cache: dict[tuple[str, str], list[Point]] | None = None,
        obstacle_footprints: list[list[object]] | None = None,
        steps: int | None = None,
    ) -> FrenetCandidate:
        _, stop_s = stop_target_info
        target_s = max(start.s, stop_s - self.config.frenet_stop_distance_buffer)
        states, accel_cost, jerk_cost = self._build_deceleration_states(
            agent, action, reference_path, start, target_s=target_s, lateral_offset=0.0, steps=steps
        )

        cost = self._cost(
            states,
            target_speed=0.0,
            nominal_d=0.0,
            accel_cost=accel_cost,
            jerk_cost=jerk_cost,
            obstacle_trajectories=obstacle_trajectories,
            reference_path=reference_path,
            map_=map_,
            time_ms=time_ms,
            stop_target_info=stop_target_info,
            conflict_points_cache=conflict_points_cache,
            obstacle_footprints=obstacle_footprints,
        )
        return FrenetCandidate(states=states, cost=cost)

    def _build_deceleration_states(
        self,
        agent: AgentDecisionState,
        action: LimSimAction,
        reference_path: ReferencePath,
        start: FrenetPoint,
        target_s: float | None = None,
        lateral_offset: float | None = None,
        steps: int | None = None,
    ) -> tuple[list[AgentDecisionState], float, float]:
        """Roll out a physically continuous braking trajectory."""

        if target_s is None:
            deceleration = self.config.frenet_stop_deceleration
        else:
            distance_to_stop = max(target_s - start.s, 0.0)
            deceleration = (
                self.config.frenet_stop_deceleration
                if distance_to_stop <= 1e-6
                else min(
                    self.config.frenet_stop_deceleration, agent.speed**2 / (2.0 * distance_to_stop)
                )
            )

        offset = agent.lateral_offset if lateral_offset is None else lateral_offset
        states = []
        current_s = start.s
        previous_speed = agent.speed
        previous_acceleration = self._action_acceleration(action)
        accel_cost = 0.0
        jerk_cost = 0.0
        planning_steps = self.config.horizon_steps if steps is None else max(1, int(steps))
        for _ in range(planning_steps):
            speed = max(previous_speed - deceleration * self.config.dt, 0.0)
            travel = 0.5 * (previous_speed + speed) * self.config.dt
            if reference_path.terminal and current_s + travel > reference_path.path.length:
                break
            current_s += travel
            x, y, heading = reference_path.frenet_to_cartesian(current_s, offset)
            acceleration = (speed - previous_speed) / self.config.dt
            jerk = (acceleration - previous_acceleration) / self.config.dt
            accel_cost += acceleration**2
            jerk_cost += jerk**2
            states.append(
                agent.with_updates(
                    x=x,
                    y=y,
                    heading=heading,
                    speed=speed,
                    lane_id=agent.lane_id,
                    route_progress=current_s,
                    lateral_offset=offset,
                    action=action,
                )
            )
            previous_speed = speed
            previous_acceleration = acceleration
        return states, accel_cost, jerk_cost

    def _stop_rule_cost(
        self,
        states: Sequence[AgentDecisionState],
        reference_path: ReferencePath,
        map_: Map,
        time_ms: int | None = None,
        stop_target_info: tuple[StopTarget, float] | None = None,
    ) -> float:
        if stop_target_info is None:
            if not states:
                return 0.0
            # fallback: compute on demand (backward-compatible path)
            stop_target_info = self._nearest_required_stop_target(
                states[0], reference_path, map_, time_ms=time_ms
            )
        if stop_target_info is None:
            return 0.0
        _, stop_s = stop_target_info
        stop_s = max(0.0, stop_s - self.config.frenet_stop_distance_buffer)
        cost = 0.0
        for state in states:
            frenet = reference_path.cartesian_to_frenet(
                state.x, state.y, hint_s=state.route_progress
            )
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
        conflict_points_cache: dict[tuple[str, str], list[Point]] | None = None,
    ) -> float:
        if not states or not obstacle_trajectories:
            return 0.0
        lane_ids = set(reference_path.lane_ids)
        cost = 0.0
        for obstacle in obstacle_trajectories:
            if not obstacle:
                continue
            obstacle_lane_id = next((state.lane_id for state in obstacle if state.lane_id), None)
            if obstacle_lane_id is None:
                continue
            conflict_points: list[Point] = []
            for lane_id in lane_ids:
                if conflict_points_cache is not None:
                    conflict_points.extend(
                        conflict_points_cache.get((lane_id, obstacle_lane_id), [])
                    )
                else:
                    # fallback: query on demand (backward-compatible path)
                    conflict_points.extend(
                        SemanticMapQuery(map_).get_conflict_points(lane_id, obstacle_lane_id)
                    )
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

    def _first_step_near_point(
        self, states: Sequence[AgentDecisionState], point: Point
    ) -> int | None:
        for step, state in enumerate(states):
            if (
                spatial.euclidean_distance(state.location, (point.x, point.y))
                <= self.config.frenet_junction_conflict_distance
            ):
                return step
        return None


def _lane_width(lane, default_width: float) -> float:
    if lane is None:
        return default_width
    width = lane.get_width(samples=5, default=default_width)
    return float(width) if width is not None else default_width


def reference_path_from_agent(
    agent: AgentDecisionState, map_: Map | None, config: LimSimConfig, cache: dict | None = None
) -> ReferencePath | None:
    """Build a generic reference path from LimSim agent state and map context.

    Args:
        cache: Optional ``dict`` keyed by ``(id(map_), tuple(route_lanes))``
            that stores ``(path_array, route_lanes_tuple, lane_width)`` to
            avoid rebuilding identical lane centerline concatenations.
    """

    if map_ is None or agent.lane_id is None or agent.lane_id not in map_.lanes:
        return None

    # A known multi-lane route is the low-level navigation constraint.  The
    # local available-lane approximation can lag on short lanelets, so it must
    # not truncate that route.  Topology-only fallback remains constrained.
    route_agent = (
        agent.with_updates(available_lane_ids=frozenset())
        if len(agent.route_lane_ids) > 1
        else agent
    )
    route_lanes = route_lanes_from_agent(route_agent, map_, config.max_routes_per_agent)

    # Never concatenate disconnected lane centerlines.  A missing or
    # incorrect topology relation must shorten the reference path rather than
    # create a long straight bridge across the map.
    route_lanes = _continuous_route_lanes(route_lanes, map_, agent.route_lane_ids)
    route_lanes_tuple = tuple(route_lanes)
    lane_width = _lane_width(map_.lanes[agent.lane_id], config.default_lane_width)

    # --- check cache for pre-built centerline concatenation ---
    cache_key = (id(map_), route_lanes_tuple, tuple(agent.route_lane_ids))
    if cache is not None and cache_key in cache:
        cached_array, _, cached_width = cache[cache_key]
        path_array = cached_array.copy()  # copy before alignment may reverse
        lane_width = cached_width
    else:
        path_array = _route_path_from_lanes(route_lanes, map_, agent.route_lane_ids)
        if path_array is None or len(path_array) < 2:
            return None
        if cache is not None:
            cache[cache_key] = (path_array.copy(), route_lanes_tuple, lane_width)

    path_line = LineString(path_array)
    initial_s = None
    if route_lanes and route_lanes[0] == agent.lane_id:
        current_lane = map_.lanes.get(agent.lane_id)
        current_centerline = current_lane.centerline() if current_lane is not None else None
        if current_centerline is not None and current_centerline.length > 1e-6:
            prefix_limit = min(path_line.length, float(current_centerline.length) + 4.0)
            prefix = substring(path_line, 0.0, prefix_limit)
            initial_s = float(prefix.project(Point(agent.x, agent.y)))

    aligned = frenet.align_path_with_heading(
        path_array, agent.x, agent.y, agent.heading, progress_hint=initial_s
    )
    if initial_s is not None and np.allclose(aligned, path_array[::-1]):
        initial_s = float(path_line.length - initial_s)

    terminal = not route_has_continuation(
        route_lanes, map_, agent.route_lane_ids, route_agent.available_lane_ids
    )
    return frenet.ReferencePath(
        LineString(aligned),
        route_lanes_tuple,
        lane_width=lane_width,
        initial_s=initial_s,
        terminal=terminal,
    )


def _footprint(state: AgentDecisionState, length: float | None = None, width: float | None = None):
    return spatial.oriented_box(
        state.x,
        state.y,
        state.heading,
        state.length if length is None else length,
        state.width if width is None else width,
    )
