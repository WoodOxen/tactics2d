# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Closed-loop replay execution for LimSim planning results.

The behavior model produces candidate trajectories. This module owns the
simulation-side state machine that feeds the first planned step back into
participants while replaying vehicles that have not been taken over yet.
The controller separates candidate planning from takeover: ``is_involved``
decides which planned vehicles switch away from dataset replay, while already
owned vehicles remain under planner control.
"""

import statistics
import time
from bisect import bisect_left, insort
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Dict, Mapping, Optional, Tuple

import numpy as np

from tactics2d.geometry import spatial
from tactics2d.participant.element import Vehicle

from .model import LimSimBehaviorModel


@dataclass
class InteractiveReplayResult:
    """Recorded output and diagnostics from one closed-loop run."""

    ego_plans: Dict[object, list] = field(default_factory=dict)
    simulation_frames: list = field(default_factory=list)
    rollout_states: Dict[object, Dict[int, object]] = field(default_factory=dict)
    controlled_ids_by_frame: Dict[int, set] = field(default_factory=dict)
    plan_times: list = field(default_factory=list)
    num_cycles: int = 0
    expected_cycles: int = 0
    owned_vehicle_count: int = 0
    retired_vehicle_count: int = 0
    avg_roi_vehicles: float = 0.0
    avg_background_vehicles: float = 0.0
    avg_planned_vehicles: float = 0.0
    avg_controlled_vehicles: float = 0.0
    avg_new_takeovers: float = 0.0
    avg_replayed_candidates: float = 0.0
    avg_interaction_groups: float = 0.0
    multi_agent_cycles: int = 0
    collision_events: int = 0
    collision_steps: int = 0
    status: str = "not_started"
    error_message: str = ""


def _set_trajectory_state(trajectory, state):
    """Insert or replace a state without duplicating trajectory frames."""

    if state.frame not in trajectory._history_states:
        insort(trajectory._frames, state.frame)
    trajectory._history_states[state.frame] = state
    if trajectory._frames and state.frame == trajectory._frames[-1]:
        trajectory._current_state = state


def _replace_trajectory_states(trajectory, states):
    """Replace a trajectory's states while keeping its internal indexes consistent."""

    trajectory._history_states = dict(states)
    trajectory._frames = sorted(states)
    trajectory._current_state = (
        trajectory._history_states[trajectory._frames[-1]] if trajectory._frames else None
    )


def snapshot_vehicle_trajectories(participants):
    """Copy recorded states for every vehicle in a participant mapping."""

    return {
        participant_id: dict(participant.trajectory.history_states)
        for participant_id, participant in participants.items()
        if isinstance(participant, Vehicle)
    }


def restore_recorded_snapshots(participants, recorded_snapshots):
    """Restore vehicle trajectories after a replay configuration finishes."""

    for participant_id, states in recorded_snapshots.items():
        _replace_trajectory_states(participants[participant_id].trajectory, states)


def apply_rollout_states(participants, rollout_states):
    """Apply saved closed-loop states to freshly restored participants."""

    for participant_id, states in rollout_states.items():
        trajectory = participants[participant_id].trajectory
        for frame in sorted(states):
            _set_trajectory_state(trajectory, states[frame])
        if states:
            trajectory._current_state = trajectory.get_state(max(states))


def collision_pairs_at_frame(participants, participant_ids, frame):
    """Return colliding vehicle pairs at one simulated frame."""

    active = []
    for participant_id in participant_ids:
        participant = participants.get(participant_id)
        if not isinstance(participant, Vehicle) or not participant.trajectory.has_state(frame):
            continue
        state = participant.trajectory.get_state(frame)
        active.append(
            (
                participant_id,
                spatial.oriented_box(
                    state.x,
                    state.y,
                    state.heading,
                    participant.length or 4.8,
                    participant.width or 1.9,
                ),
            )
        )

    collisions = set()
    for (first_id, first_shape), (second_id, second_shape) in combinations(active, 2):
        if first_shape.intersects(second_shape):
            collisions.add(frozenset((first_id, second_id)))
    return collisions


class InteractiveReplayController:
    """Execute LimSim plans against replayed Tactics2D participants.

    The controller is deliberately independent of a particular dataset. The
    caller supplies the parsed scene and a snapshot of its recorded vehicle
    trajectories. A dataset-specific route updater can be injected when routes
    need to be discovered lazily. The simulation advances at ``config.dt``;
    planning runs at ``config.planning_interval`` and cached trajectory states
    are consumed between planning frames.
    """

    def __init__(
        self,
        model: LimSimBehaviorModel,
        participants: Dict[object, object],
        map_,
        ego_id: object,
        route_map: Optional[Dict[object, Tuple[str, ...]]] = None,
        recorded_snapshots: Optional[Mapping[object, Mapping[int, object]]] = None,
        roi_radius: float = 50.0,
        roi_outer_radius: float = 100.0,
        replay_max_gap_ms: int = 500,
        short_replay_state_threshold: int = 10,
        involvement_check_stride: int = 3,
        route_updater: Optional[Callable[..., None]] = None,
    ):
        if roi_radius <= 0:
            raise ValueError("roi_radius must be positive.")
        if roi_outer_radius < roi_radius:
            raise ValueError("roi_outer_radius must be greater than or equal to roi_radius.")
        if short_replay_state_threshold < 0:
            raise ValueError("short_replay_state_threshold must be non-negative.")
        if involvement_check_stride <= 0:
            raise ValueError("involvement_check_stride must be positive.")
        if ego_id not in participants or not isinstance(participants[ego_id], Vehicle):
            raise ValueError("ego_id must identify a vehicle in participants.")

        self.model = model
        self.participants = participants
        self.map = map_
        self.ego_id = ego_id
        self.route_map = route_map if route_map is not None else {}
        self.roi_radius = float(roi_radius)
        self.roi_outer_radius = float(roi_outer_radius)
        self.replay_max_gap_ms = int(replay_max_gap_ms)
        self.short_replay_state_threshold = int(short_replay_state_threshold)
        self.involvement_check_stride = int(involvement_check_stride)
        self.route_updater = route_updater
        if recorded_snapshots is None:
            recorded_snapshots = snapshot_vehicle_trajectories(participants)
        self.recorded_snapshots = {
            participant_id: dict(states) for participant_id, states in recorded_snapshots.items()
        }
        self.recorded_frames = {
            participant_id: sorted(states)
            for participant_id, states in self.recorded_snapshots.items()
        }
        interval = getattr(self.model.config, "planning_interval", 0.5)
        try:
            interval_ms = int(round(float(interval) * 1000))
        except (TypeError, ValueError) as error:
            raise ValueError("planning_interval must be a positive number of seconds.") from error
        if interval_ms <= 0:
            raise ValueError("planning_interval must be positive.")
        self._planning_interval_ms = max(self.step_ms, interval_ms)
        self.reset(0)

    @property
    def step_ms(self) -> int:
        """Return the planner's closed-loop step in milliseconds."""

        return self.model.config.step_ms

    def reset(self, start_frame: int):
        """Reset controller state before running a new configuration."""

        reset_decisions = getattr(self.model, "_reset_decision_cache", None)
        if callable(reset_decisions):
            reset_decisions()
        self.current_frame = start_frame
        self.last_planning_frame = None
        self.owned_ids = set()
        self.retired_ids = set()
        self.last_plans = {}
        self.ego_plans = {}
        self.simulation_frames = []
        self.rollout_states = {}
        self.controlled_ids_by_frame = {}
        self.plan_times = []
        self.roi_counts = []
        self.background_counts = []
        self.planned_counts = []
        self.controlled_counts = []
        self.new_takeover_counts = []
        self.replayed_candidate_counts = []
        self.group_counts = []
        self.multi_agent_cycles = 0
        self.collision_events = 0
        self.collision_steps = 0
        self.previous_collisions = set()
        self.status = "running"
        self.error_message = ""

    def run(self, start_frame: int, steps: int) -> InteractiveReplayResult:
        """Run up to ``steps`` receding-horizon cycles."""

        if steps < 0:
            raise ValueError("steps must be non-negative.")
        self.reset(start_frame)
        for _ in range(steps):
            if not self.step():
                break
        if self.status == "running":
            self.status = "completed" if len(self.simulation_frames) == steps else "incomplete"
        return self.result(expected_cycles=steps)

    def step(self) -> bool:
        """Plan and execute one cycle at the current controller frame."""

        if self.status != "running":
            return False
        if not self.participants[self.ego_id].trajectory.has_state(self.current_frame):
            return self._fail("ego has no state at the current replay frame")

        next_frame = self.current_frame + self.step_ms
        planning_result = None
        planned_ids = set()
        controlled_plan_ids = set()
        new_takeovers = set()
        planning_due = (
            self.last_planning_frame is None
            or self.current_frame - self.last_planning_frame >= self._planning_interval_ms
            or any(
                self.last_plans.get(participant_id) is None
                or not self.last_plans[participant_id].has_state(next_frame)
                for participant_id in self.owned_ids - self.retired_ids
            )
        )
        if planning_due:
            try:
                self._update_local_route_map()
            except Exception as error:
                return self._fail(f"route update failed: {type(error).__name__}: {error}")
            started = time.perf_counter()
            try:
                planning_result = self.model.plan(
                    participants=self.participants,
                    map_=self.map,
                    frame=self.current_frame,
                    route_map=self.route_map,
                    ego_id=self.ego_id,
                    roi_radius=self.roi_radius,
                    roi_outer_radius=self.roi_outer_radius,
                    last_planned_trajectories=self.last_plans,
                )
            except Exception as error:
                return self._fail(f"planning failed: {type(error).__name__}: {error}")
            self.plan_times.append(time.perf_counter() - started)
            self.last_planning_frame = self.current_frame

            ego_trajectory = planning_result.trajectories.get(self.ego_id)
            if ego_trajectory is None or not ego_trajectory.has_state(next_frame):
                return self._fail("planning result did not contain the next ego state")

            self.roi_counts.append(len(planning_result.roi_agent_ids))
            self.background_counts.append(len(planning_result.background_agent_ids))
            self.planned_counts.append(len(planning_result.trajectories))
            self.group_counts.append(len(planning_result.groups))
            if any(len(group) > 1 for group in planning_result.groups):
                self.multi_agent_cycles += 1

            planned_ids = {
                participant_id
                for participant_id, trajectory in planning_result.trajectories.items()
                if trajectory.has_state(next_frame)
            }
            controlled_plan_ids = {
                participant_id
                for participant_id in planned_ids
                if self.is_involved(participant_id, planning_result, self.current_frame)
            }
            if self.ego_id not in controlled_plan_ids:
                return self._fail("planning result did not contain the next ego state")

            new_takeovers = controlled_plan_ids - self.owned_ids
            for participant_id in new_takeovers:
                self._truncate_trajectory_after(
                    self.participants[participant_id].trajectory, self.current_frame
                )
            self.owned_ids.update(new_takeovers)
            self.last_plans.update(
                {
                    participant_id: planning_result.trajectories[participant_id]
                    for participant_id in controlled_plan_ids
                }
            )
            self.new_takeover_counts.append(len(new_takeovers))
            self.replayed_candidate_counts.append(len(planned_ids - controlled_plan_ids))

        cached_ego = self.last_plans.get(self.ego_id)
        if cached_ego is not None:
            self.ego_plans[self.current_frame] = [
                (cached_ego.get_state(frame).x, cached_ego.get_state(frame).y)
                for frame in cached_ego.frames
                if frame > self.current_frame
            ]

        for participant_id in self.owned_ids - self.retired_ids:
            participant = self.participants[participant_id]
            if not participant.trajectory.has_state(self.current_frame):
                self.retired_ids.add(participant_id)
                self.last_plans.pop(participant_id, None)
                continue
            cached = self.last_plans.get(participant_id)
            if cached is not None and cached.has_state(next_frame):
                next_state = cached.get_state(next_frame)
                _set_trajectory_state(participant.trajectory, next_state)
            else:
                if participant_id == self.ego_id:
                    return self._fail("cached ego plan did not cover the next simulation step")
                self.retired_ids.add(participant_id)
                self.last_plans.pop(participant_id, None)

        active_controlled_ids = self.owned_ids - self.retired_ids
        self.controlled_counts.append(len(active_controlled_ids))
        self.controlled_ids_by_frame[next_frame] = set(active_controlled_ids)

        for participant_id, recorded_states in self.recorded_snapshots.items():
            if participant_id in self.owned_ids:
                continue
            replay_state = self._interpolate_recorded_state(
                recorded_states, self.recorded_frames[participant_id], next_frame
            )
            if replay_state is not None:
                _set_trajectory_state(self.participants[participant_id].trajectory, replay_state)

        next_ego_state = self.participants[self.ego_id].trajectory.get_state(next_frame)
        local_ids = []
        for participant_id, participant in self.participants.items():
            if not isinstance(participant, Vehicle) or not participant.trajectory.has_state(
                next_frame
            ):
                continue
            state = participant.trajectory.get_state(next_frame)
            if (
                np.hypot(state.x - next_ego_state.x, state.y - next_ego_state.y)
                <= self.roi_outer_radius
            ):
                local_ids.append(participant_id)

        current_collisions = {
            pair
            for pair in collision_pairs_at_frame(self.participants, local_ids, next_frame)
            if not pair.isdisjoint(active_controlled_ids)
        }
        self.collision_events += len(current_collisions - self.previous_collisions)
        self.collision_steps += int(bool(current_collisions))
        self.previous_collisions = current_collisions

        for participant_id, participant in self.participants.items():
            if isinstance(participant, Vehicle) and participant.trajectory.has_state(next_frame):
                self.rollout_states.setdefault(participant_id, {})[next_frame] = (
                    participant.trajectory.get_state(next_frame)
                )

        self.simulation_frames.append(next_frame)
        self.current_frame = next_frame
        return True

    def result(self, expected_cycles: Optional[int] = None) -> InteractiveReplayResult:
        """Return a snapshot of the current run output."""

        expected = len(self.simulation_frames) if expected_cycles is None else expected_cycles
        return InteractiveReplayResult(
            ego_plans=dict(self.ego_plans),
            simulation_frames=list(self.simulation_frames),
            rollout_states={
                participant_id: dict(states)
                for participant_id, states in self.rollout_states.items()
            },
            controlled_ids_by_frame={
                frame: set(participant_ids)
                for frame, participant_ids in self.controlled_ids_by_frame.items()
            },
            plan_times=list(self.plan_times),
            num_cycles=len(self.simulation_frames),
            expected_cycles=expected,
            owned_vehicle_count=len(self.owned_ids),
            retired_vehicle_count=len(self.retired_ids),
            avg_roi_vehicles=statistics.mean(self.roi_counts) if self.roi_counts else 0.0,
            avg_background_vehicles=(
                statistics.mean(self.background_counts) if self.background_counts else 0.0
            ),
            avg_planned_vehicles=(
                statistics.mean(self.planned_counts) if self.planned_counts else 0.0
            ),
            avg_controlled_vehicles=(
                statistics.mean(self.controlled_counts) if self.controlled_counts else 0.0
            ),
            avg_new_takeovers=(
                statistics.mean(self.new_takeover_counts) if self.new_takeover_counts else 0.0
            ),
            avg_replayed_candidates=(
                statistics.mean(self.replayed_candidate_counts)
                if self.replayed_candidate_counts
                else 0.0
            ),
            avg_interaction_groups=(
                statistics.mean(self.group_counts) if self.group_counts else 0.0
            ),
            multi_agent_cycles=self.multi_agent_cycles,
            collision_events=self.collision_events,
            collision_steps=self.collision_steps,
            status=self.status,
            error_message=self.error_message,
        )

    def _fail(self, message: str) -> bool:
        self.status = "incomplete"
        self.error_message = message
        return False

    def is_involved(self, participant_id, planning_result, frame: int) -> bool:
        """Return whether a planned vehicle should switch from replay to control."""

        if participant_id == self.ego_id or participant_id in self.owned_ids:
            return True
        if self._recorded_track_is_short(participant_id, frame):
            return True

        for aoi_id in (self.owned_ids - self.retired_ids) | {self.ego_id}:
            aoi_trajectory = planning_result.trajectories.get(aoi_id)
            if aoi_trajectory is None:
                aoi_trajectory = self.last_plans.get(aoi_id)
            if aoi_trajectory is not None and self._recorded_trajectory_collides(
                participant_id, aoi_id, aoi_trajectory, frame
            ):
                return True
        return False

    def _recorded_track_is_short(self, participant_id, frame: int) -> bool:
        recorded_frames = self.recorded_frames.get(participant_id, ())
        if not recorded_frames:
            return True

        current_index = bisect_left(recorded_frames, frame)
        return len(recorded_frames) - current_index <= self.short_replay_state_threshold

    def _recorded_trajectory_collides(
        self, participant_id, target_id, target_trajectory, frame: int
    ) -> bool:
        """Compare a candidate's database future with a controlled plan."""

        candidate = self.participants.get(participant_id)
        target = self.participants.get(target_id)
        recorded_states = self.recorded_snapshots.get(participant_id)
        recorded_frames = self.recorded_frames.get(participant_id)
        if not isinstance(candidate, Vehicle) or not isinstance(target, Vehicle):
            return False
        if not recorded_states or not recorded_frames:
            return False

        future_frames = [planned for planned in target_trajectory.frames if planned > frame]
        for future_frame in future_frames[:: self.involvement_check_stride]:
            candidate_state = self._interpolate_recorded_state(
                recorded_states, recorded_frames, future_frame
            )
            if candidate_state is None:
                break
            target_state = target_trajectory.get_state(future_frame)
            candidate_shape = spatial.oriented_box(
                candidate_state.x,
                candidate_state.y,
                candidate_state.heading,
                candidate.length,
                candidate.width,
            )
            target_shape = spatial.oriented_box(
                target_state.x, target_state.y, target_state.heading, target.length, target.width
            )
            if candidate_shape.intersects(target_shape):
                return True
        return False

    def _update_local_route_map(self):
        if self.route_updater is not None:
            self.route_updater(
                self.participants,
                self.map,
                self.current_frame,
                self.ego_id,
                self.roi_outer_radius,
                self.route_map,
            )

    def _truncate_trajectory_after(self, trajectory, frame):
        retained_states = {
            existing: state
            for existing, state in trajectory._history_states.items()
            if existing <= frame
        }
        _replace_trajectory_states(trajectory, retained_states)

    def _interpolate_recorded_state(self, recorded_states, recorded_frames, frame):
        if frame in recorded_states:
            return recorded_states[frame]
        right_index = bisect_left(recorded_frames, frame)
        if right_index == 0 or right_index == len(recorded_frames):
            return None
        left_frame = recorded_frames[right_index - 1]
        right_frame = recorded_frames[right_index]
        if right_frame - left_frame > self.replay_max_gap_ms:
            return None

        left = recorded_states[left_frame]
        right = recorded_states[right_frame]
        weight = (frame - left_frame) / (right_frame - left_frame)
        x = left.x + weight * (right.x - left.x)
        y = left.y + weight * (right.y - left.y)
        heading_delta = np.arctan2(
            np.sin(right.heading - left.heading), np.cos(right.heading - left.heading)
        )
        heading = left.heading + weight * heading_delta
        interval_s = (right_frame - left_frame) / 1000.0
        vx = (
            left.vx + weight * (right.vx - left.vx)
            if left.vx is not None and right.vx is not None
            else (right.x - left.x) / interval_s
        )
        vy = (
            left.vy + weight * (right.vy - left.vy)
            if left.vy is not None and right.vy is not None
            else (right.y - left.y) / interval_s
        )
        return type(left)(frame=frame, x=x, y=y, heading=heading, vx=vx, vy=vy)


__all__ = [
    "InteractiveReplayController",
    "InteractiveReplayResult",
    "apply_rollout_states",
    "collision_pairs_at_frame",
    "restore_recorded_snapshots",
    "snapshot_vehicle_trajectories",
]
