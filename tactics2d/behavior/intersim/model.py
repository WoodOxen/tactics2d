# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Public InterSim-style relation-driven behavior model."""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from tactics2d.behavior.base import BehaviorModelBase
from tactics2d.geometry import spatial
from tactics2d.map.element import Map
from tactics2d.participant.element import Vehicle
from tactics2d.participant.trajectory import State, Trajectory

from .config import InterSimConfig
from .planner import (
    ArcPath,
    arcs_from_speeds,
    cruise_with_corner_caps,
    deceleration_speeds,
    lane_chain_points,
    match_lane,
    polyline_nearest_s,
    straight_path,
    yield_speeds,
)
from .relation import AgentBody, Edge, check_body_collision, detect_relation_edges

# Adapted from InterSim (github.com/Tsinghua-MARS-Lab/InterSim), MIT,
# Copyright (c) 2022 Tsinghua MARS Lab.

# A reactor yields to any conflict inside the planning horizon (upstream scans
# the whole horizon for the earliest collision rather than gating on a
# relative-arrival window), but stops outright only when the conflict is close.
_MIN_DISTANCE_TO_TRAVEL = 4.0
# In "directed_tie" mode a near-simultaneous arrival is treated like upstream's
# relation no-edge (the learned model cannot pick a side): both agents yield.
# A clear time separation keeps the plain directed single-yield behaviour.
_NO_EDGE_TIE_FRAMES = 3
# Relation detection only needs close-in-time contacts; bounding the pair scan
# keeps large scenes tractable.
_COLLISION_GAP = 12


@dataclass
class InterSimPlanResult:
    """Output of one InterSim-style planning update.

    ``trajectories`` covers the requested agents only; ``relations`` holds the
    directed ``[influencer -> reactor]`` conflict graph detected on the
    baseline rollouts, and ``actions`` tags every planned scene agent as
    ``"follow"`` (keeps its plan) or ``"yield"`` (braked for a reactor).
    """

    trajectories: Dict[object, Trajectory] = field(default_factory=dict)
    relations: List[Edge] = field(default_factory=list)
    actions: Dict[object, str] = field(default_factory=dict)
    scene_agent_ids: List[object] = field(default_factory=list)


@dataclass
class _AgentRecord:
    """Per-agent state plus its forward reference path."""

    agent_id: object
    is_vehicle: bool
    length: float
    width: float
    x: float
    y: float
    heading: float
    v0: float
    path: Optional[ArcPath] = None
    s0: float = 0.0
    goal: Optional[np.ndarray] = None
    speed_limit: Optional[float] = None
    intent_speed: Optional[float] = None
    lane_id: Optional[object] = None
    # Braking/acceleration limits taken from the participant (Vehicle defaults:
    # max_decel 10.0, max_accel 3.0 m/s^2) instead of a hard-coded constant.
    max_accel: float = 3.0
    max_decel: float = 10.0


class InterSimBehaviorModel(BehaviorModelBase):
    """Plan interactions with explicit directed relations on Tactics2D data.

    Each requested vehicle first rolls a lane-following (or constant-velocity)
    baseline; overlapping baselines are turned into directed relations, and
    every reactor decelerates to a stop short of its conflict so the final
    forecast is collision-free. This reproduces InterSim's rule core without
    any learned predictor: the relation and marginal model options are closed.
    """

    def __init__(self, config: Optional[InterSimConfig] = None):
        self.config = config or InterSimConfig()
        if self.config.use_relation_model or self.config.use_marginal_model:
            raise NotImplementedError(
                "InterSim learning-based predictors are not ported; all weight "
                "options must stay disabled."
            )
        if self.config.relation_mode not in ("directed", "directed_tie", "yield_all", "nn"):
            raise ValueError(
                "relation_mode must be 'directed', 'directed_tie', 'yield_all', or 'nn', "
                f"got {self.config.relation_mode!r}."
            )
        self._relations_override = None
        # Optional learned direction arbiter: callable(reactor, influencer) ->
        # True/False/None, used by relation_mode="nn" (see rolling runner).
        self._relation_decider = None

    # -- public interface -------------------------------------------------

    def plan(
        self,
        participants: Dict[object, object],
        map_: Optional[Map],
        frame: int,
        agent_ids: Optional[Iterable[object]] = None,
    ) -> InterSimPlanResult:
        """Plan collision-free future trajectories for the requested agents."""

        records = self._build_scene_records(participants, map_, frame, agent_ids)
        result = InterSimPlanResult(
            scene_agent_ids=[record.agent_id for record in records.values()]
        )
        if not records:
            return result

        for record in records.values():
            record.path, record.s0 = self._reference_path(record, map_)

        baseline = {agent_id: self._baseline_speeds(record) for agent_id, record in records.items()}
        poses = {agent_id: self._poses(record, baseline[agent_id]) for agent_id, record in records.items()}
        shapes = {agent_id: (record.length, record.width) for agent_id, record in records.items()}
        is_vehicle = {agent_id: record.is_vehicle for agent_id, record in records.items()}

        if self._relations_override is not None:
            # The override may have been computed over a wider agent set than the
            # records of this call (ego-only vs env planning), so keep only edges
            # whose endpoints are actually modeled here.
            edges = [
                edge
                for edge in self._relations_override
                if edge.influencer in records and edge.reactor in records
            ]
        else:
            edges = detect_relation_edges(poses, shapes, is_vehicle, max_gap=_COLLISION_GAP)
        result.relations = list(edges)

        final_speeds = self._resolve_conflicts(records, baseline, edges)
        self._apply_traffic_lights(records, final_speeds, frame, map_)
        final_poses = {
            agent_id: self._poses(records[agent_id], final_speeds[agent_id])
            for agent_id in records
        }

        for agent_id in records:
            result.actions[agent_id] = "yield" if agent_id in self._yielded else "follow"

        for agent_id, record in records.items():
            if not record.is_vehicle:
                continue
            if agent_id in participants and agent_id in final_poses:
                result.trajectories[agent_id] = self._to_trajectory(
                    record, final_speeds[agent_id], final_poses[agent_id], frame
                )
        result.trajectories = {
            agent_id: trajectory
            for agent_id, trajectory in result.trajectories.items()
            if agent_ids is None or agent_id in set(agent_ids)
        }
        return result

    def predict(
        self,
        participants: Dict[object, object],
        map_: Optional[Map],
        frame: int,
        agent_ids: Optional[Iterable[object]] = None,
    ) -> Dict[object, Trajectory]:
        """Plan future trajectories for selected agents.

        This method provides the shared behavior-model interface. Use
        :meth:`plan` when the directed relations and per-agent actions are
        needed.
        """

        return self.plan(participants, map_, frame, agent_ids=agent_ids).trajectories

    # -- scene construction ----------------------------------------------

    def _build_scene_records(self, participants, map_, frame, agent_ids):
        requested_ids = self._select_vehicle_ids(participants, agent_ids)
        records: Dict[object, _AgentRecord] = {}
        for agent_id in requested_ids:
            record = self._make_record(participants.get(agent_id), frame)
            if record is not None:
                records[agent_id] = record

        requested_positions = [(record.x, record.y) for record in records.values()]
        for participant_id, participant in participants.items():
            if participant_id in records:
                continue
            state = self._current_state(participant, frame)
            if state is None:
                continue
            if not any(
                np.hypot(state.x - x, state.y - y) <= self.config.interaction_distance
                for x, y in requested_positions
            ):
                continue
            record = self._make_record(participant, frame, state=state)
            if record is not None:
                records[participant_id] = record
        return records

    def _select_vehicle_ids(self, participants, agent_ids) -> List[object]:
        if agent_ids is None:
            return [agent_id for agent_id, participant in participants.items() if isinstance(participant, Vehicle)]
        selected = []
        for agent_id in agent_ids:
            participant = participants.get(agent_id)
            if participant is not None and isinstance(participant, Vehicle):
                selected.append(agent_id)
        return selected

    def _make_record(self, participant, frame, state=None) -> Optional[_AgentRecord]:
        if participant is None:
            return None
        state = state if state is not None else self._current_state(participant, frame)
        if state is None:
            return None
        is_vehicle = isinstance(participant, Vehicle)
        length = participant.length
        width = participant.width
        if length is None or length <= 0.0:
            length = self.config.default_vehicle_length if is_vehicle else 0.6
        if width is None or width <= 0.0:
            width = self.config.default_vehicle_width if is_vehicle else 0.5
        speed = state.speed
        if speed is None:
            if state.vx is not None and state.vy is not None:
                speed = float(np.hypot(state.vx, state.vy))
            else:
                speed = 0.0
        goal_xy = getattr(participant, "goal_xy", None)
        goal = None
        if goal_xy is not None:
            goal_array = np.asarray(goal_xy, dtype=float)
            if goal_array.shape == (2,):
                goal = goal_array
        return _AgentRecord(
            agent_id=participant.id_,
            is_vehicle=is_vehicle,
            length=float(length),
            width=float(width),
            x=float(state.x),
            y=float(state.y),
            heading=spatial.normalize_angle(float(state.heading)),
            v0=float(max(0.0, speed)),
            goal=goal,
            intent_speed=float(getattr(participant, "intent_speed", 0.0) or 0.0),
            max_accel=float(getattr(participant, "max_accel", 0.0) or 3.0),
            max_decel=float(
                self.config.vehicle_decel_limit
                or getattr(participant, "max_decel", 0.0)
                or 10.0
            ),
        )

    def _current_state(self, participant, frame):
        trajectory = participant.trajectory
        observed = [frame_ms for frame_ms in trajectory.frames if frame_ms <= frame]
        if not observed:
            return None
        return trajectory.get_state(observed[-1])

    def _reference_path(self, record: _AgentRecord, map_: Optional[Map]) -> Tuple[ArcPath, float]:
        config = self.config
        lookahead = record.v0 * config.horizon_steps * config.dt + record.length + config.stop_margin + 10.0
        path = None
        s0 = 0.0
        if map_ is not None:
            matched = match_lane(
                map_,
                record.x,
                record.y,
                record.heading,
                config.lane_match_radius,
                config.lane_heading_tolerance_deg,
                lookahead,
            )
            if matched is not None:
                lane_id, s0 = matched
                lane = map_.lanes[lane_id]
                record.lane_id = lane_id
                record.speed_limit = lane.speed_limit
                chain = lane_chain_points(map_, lane_id, lookahead, goal=record.goal)
                if chain is not None:
                    path = ArcPath(chain)
        if path is None:
            path = straight_path(record.x, record.y, record.heading, max(lookahead, 20.0))
            s0 = 0.0
        return path, s0

    # -- rollout and conflict resolution ---------------------------------

    def _baseline_speeds(self, record: _AgentRecord) -> np.ndarray:
        """Return a cruise profile accelerating toward the target speed.

        A free vehicle cruises toward ``max(v0, cruise_speed)`` instead of
        holding a frozen speed, so an agent that once braked to a stop can
        resume once its conflict clears (as upstream env planners do).
        """

        config = self.config
        if not record.is_vehicle:
            return np.full(config.horizon_steps + 1, record.v0, dtype=float)
        target = max(record.v0, config.cruise_speed)
        if record.intent_speed:
            target = max(target, float(record.intent_speed))
        if record.speed_limit:
            target = max(target, min(float(record.speed_limit), config.max_target_speed))
        target = min(target, config.max_target_speed)
        accel = min(config.cruise_accel, record.max_accel)
        if record.path is not None and len(record.path.points) >= 2:
            return cruise_with_corner_caps(
                record.path.points,
                record.s0,
                record.v0,
                target,
                accel,
                config.max_target_speed,
                config.dt,
                config.horizon_steps,
            )
        return cruise_with_corner_caps(
            np.array(
                [[0.0, 0.0], [record.v0 * config.horizon_steps * config.dt + 10.0, 0.0]]
            ),
            record.s0,
            record.v0,
            target,
            accel,
            config.max_target_speed,
            config.dt,
            config.horizon_steps,
        )

    def _poses(self, record: _AgentRecord, speeds: np.ndarray) -> np.ndarray:
        arcs = record.s0 + arcs_from_speeds(speeds, self.config.dt)
        return record.path.sample_poses(arcs)

    def _resolve_conflicts(self, records, baseline, edges):
        config = self.config
        horizon = config.horizon_steps
        final_speeds = dict(baseline)
        self._yielded = set()

        # Map each potential reactor to the influencers it must brake for.
        # "directed" only brakes the later arrival; "yield_all" (all-yield
        # reference) brakes both sides of every imminent conflict. In "nn" mode
        # an optional ``_relation_decider`` (the learned predictor plus upstream
        # rule prefilter) overrides the direction per vehicle-vehicle pair, with
        # the geometric edge kept whenever the predictor is not confident.
        yield_edges = {(edge.influencer, edge.reactor) for edge in edges}
        reactor_influencers: Dict[object, List[object]] = {}
        handled_pairs = set()
        for edge in edges:
            if config.relation_mode in ("directed", "directed_tie", "nn"):
                a, b = edge.influencer, edge.reactor
                if not (records[a].is_vehicle and records[b].is_vehicle):
                    if records[edge.reactor].is_vehicle:
                        reactor_influencers.setdefault(edge.reactor, []).append(edge.influencer)
                    continue
                pair = tuple(sorted((a, b), key=str))
                if pair in handled_pairs:
                    continue
                handled_pairs.add(pair)
                first, second = pair
                actor_yields = (second, first) in yield_edges
                partner_yields = (first, second) in yield_edges
                if config.relation_mode == "directed_tie" and abs(edge.frame_diff) <= _NO_EDGE_TIE_FRAMES:
                    actor_yields = partner_yields = True
                elif config.relation_mode == "nn" and self._relation_decider is not None:
                    first_yields = self._relation_decider(first, second)
                    second_yields = self._relation_decider(second, first)
                    if first_yields is not None and second_yields is not None:
                        actor_yields, partner_yields = first_yields, second_yields
                if actor_yields:
                    reactor_influencers.setdefault(first, []).append(second)
                if partner_yields:
                    reactor_influencers.setdefault(second, []).append(first)
            else:
                for agent_id, counter_id in ((edge.influencer, edge.reactor), (edge.reactor, edge.influencer)):
                    if records[agent_id].is_vehicle and records[counter_id].is_vehicle:
                        reactor_influencers.setdefault(agent_id, []).append(counter_id)

        # Escalation state per reactor: slow toward a reduced speed first and
        # only fall back to a full stop when the conflict is unavoidable.
        end_targets: Dict[object, float] = {
            reactor_id: max(records[reactor_id].v0 * config.yield_speed_ratio, 0.6)
            for reactor_id in reactor_influencers
        }
        stopped: set = set()
        for _ in range(config.max_resolution_iters):
            changed = False
            poses = {
                agent_id: self._poses(records[agent_id], final_speeds[agent_id])
                for agent_id in records
            }
            for reactor_id, influencers in reactor_influencers.items():
                if reactor_id in stopped:
                    continue
                record = records[reactor_id]
                overlap_step = None
                for influencer_id in influencers:
                    if influencer_id not in records:
                        continue
                    step = self._same_time_overlap(
                        poses[reactor_id],
                        poses[influencer_id],
                        record,
                        records[influencer_id],
                    )
                    if step is not None and (overlap_step is None or step < overlap_step):
                        overlap_step = step
                if overlap_step is None:
                    continue
                push = record.length / 2.0 + config.stop_margin
                arcs = arcs_from_speeds(final_speeds[reactor_id], config.dt)
                distance = max(0.0, arcs[overlap_step] - push)
                self._yielded.add(reactor_id)
                if (
                    end_targets[reactor_id] <= 0.5
                    or distance <= _MIN_DISTANCE_TO_TRAVEL
                    or record.v0 < 0.1
                ):
                    stopped.add(reactor_id)
                    final_speeds[reactor_id] = deceleration_speeds(
                        record.v0, distance, config.dt, horizon, a_emergency=record.max_decel
                    )
                else:
                    final_speeds[reactor_id] = yield_speeds(
                        record.v0,
                        end_targets[reactor_id],
                        distance,
                        config.dt,
                        horizon,
                        a_emergency=record.max_decel,
                    )
                    end_targets[reactor_id] *= 0.5
                changed = True
            if not changed:
                break
        return final_speeds

    def _apply_traffic_lights(self, records, final_speeds, frame, map_):
        """Stop free-running vehicles before a red traffic light stop point."""

        config = self.config
        if not config.respect_traffic_light or map_ is None:
            return
        lights = [reg for reg in map_.regulations.values() if reg.is_traffic_light()]
        if not lights:
            return
        red_states = {"stop", "arrow_stop", "flashing_stop"}
        for agent_id, record in records.items():
            if not record.is_vehicle or record.lane_id is None or agent_id in self._yielded:
                continue
            for reg in lights:
                lane_ref = (reg.custom_tags or {}).get("lane_id")
                if lane_ref != record.lane_id:
                    continue
                state_record = reg.state_at(frame)
                if not state_record or state_record.get("state") not in red_states:
                    break
                stop_point = state_record.get("stop_point")
                if stop_point is None and reg.position is not None:
                    stop_point = [reg.position.x, reg.position.y]
                if stop_point is None:
                    break
                s_light, _ = polyline_nearest_s(
                    record.path.points, np.asarray(stop_point, dtype=float)
                )
                distance = s_light - record.s0
                if 0.0 < distance < record.v0 * config.horizon_steps * config.dt:
                    final_speeds[agent_id] = deceleration_speeds(
                        record.v0, distance, config.dt, config.horizon_steps
                    )
                break

    def _same_time_overlap(
        self,
        poses_a: np.ndarray,
        poses_b: np.ndarray,
        record_a: _AgentRecord,
        record_b: _AgentRecord,
        margin: float = 0.7,
    ) -> Optional[int]:
        for step in range(len(poses_a)):
            if step >= len(poses_b):
                break
            pose_a = poses_a[step]
            pose_b = poses_b[step]
            if pose_a[0] == -1 or pose_b[0] == -1:
                continue
            body_a = AgentBody(
                float(pose_a[0]), float(pose_a[1]), float(pose_a[3]), record_a.length, record_a.width
            )
            body_b = AgentBody(
                float(pose_b[0]), float(pose_b[1]), float(pose_b[3]), record_b.length, record_b.width
            )
            if check_body_collision(body_a, body_b, margin):
                return step
        return None

    # -- serialization ----------------------------------------------------

    def _to_trajectory(
        self,
        record: _AgentRecord,
        speeds: np.ndarray,
        poses: np.ndarray,
        frame: int,
    ) -> Trajectory:
        config = self.config
        trajectory = Trajectory(id_=record.agent_id, fps=round(1.0 / config.dt, 3), stable_freq=True)
        for step in range(1, config.horizon_steps + 1):
            pose = poses[step]
            speed = float(speeds[step - 1])
            heading = spatial.normalize_angle(float(pose[3]))
            trajectory.add_state(
                State(
                    frame=int(round(frame + step * config.step_ms)),
                    x=float(pose[0]),
                    y=float(pose[1]),
                    heading=heading,
                    vx=speed * np.cos(heading),
                    vy=speed * np.sin(heading),
                )
            )
        return trajectory
