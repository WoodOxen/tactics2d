# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Public InterSim-style relation-driven behavior model."""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from tactics2d.behavior.base import BehaviorModelBase
from tactics2d.geometry import spatial
from tactics2d.map.element import Map
from tactics2d.participant.trajectory import State, Trajectory
from tactics2d.routing.utils import augment_lane_successors

from . import relation_decider, replay, scene
from .config import InterSimConfig
from .planner import (
    arcs_from_speeds,
    cruise_with_corner_caps,
    deceleration_speeds,
    polyline_nearest_s,
    yield_speeds,
)
from .relation_geometry import AgentBody, Edge, check_body_collision, detect_relation_edges
from .replay import RollingSimulationResult
from .scene import AgentRecord

# Adapted from InterSim (github.com/Tsinghua-MARS-Lab/InterSim), MIT,
# Copyright (c) 2022 Tsinghua MARS Lab.

__all__ = [
    "AgentRecord",
    "InterSimBehaviorModel",
    "InterSimPlanResult",
    "RollingSimulationResult",
]

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

    # -- public interface -------------------------------------------------

    def plan(
        self,
        participants: Dict[object, object],
        map_: Optional[Map],
        frame: int,
        agent_ids: Optional[Iterable[object]] = None,
        decider=None,
    ) -> InterSimPlanResult:
        """Plan collision-free future trajectories for the requested agents.

        Args:
            participants (Dict): All participants in the scenario.
            map_ (Optional[Map]): The map. None falls back to straight paths.
            frame (int): The current frame number. The unit is millisecond (ms).
            agent_ids (Iterable, optional): The agents to plan; all vehicles when
                None. Defaults to None.
            decider (Callable, optional): Direction arbiter for
                ``relation_mode="nn"``, used by the closed-loop runner. Defaults
                to None.

        Returns:
            The planned trajectories together with the detected relations and
            per-agent follow/yield actions.
        """

        records = scene.build_scene_records(self.config, participants, frame, agent_ids)
        result = InterSimPlanResult(
            scene_agent_ids=[record.agent_id for record in records.values()]
        )
        if not records:
            return result

        for record in records.values():
            record.path, record.s0 = scene.reference_path(self.config, record, map_)

        baseline = {agent_id: self._baseline_speeds(record) for agent_id, record in records.items()}
        poses = {
            agent_id: self._poses(record, baseline[agent_id])
            for agent_id, record in records.items()
        }
        shapes = {agent_id: (record.length, record.width) for agent_id, record in records.items()}
        is_vehicle = {agent_id: record.is_vehicle for agent_id, record in records.items()}

        edges = detect_relation_edges(poses, shapes, is_vehicle, max_gap=_COLLISION_GAP)
        result.relations = list(edges)

        final_speeds, yielded = self._resolve_conflicts(records, baseline, edges, decider=decider)
        self._apply_traffic_lights(records, final_speeds, yielded, frame, map_)
        final_poses = {
            agent_id: self._poses(records[agent_id], final_speeds[agent_id]) for agent_id in records
        }

        for agent_id in records:
            result.actions[agent_id] = "yield" if agent_id in yielded else "follow"

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
        decider=None,
    ) -> Dict[object, Trajectory]:
        """Plan future trajectories for selected agents.

        This method provides the shared behavior-model interface. Use
        :meth:`plan` when the directed relations and per-agent actions are
        needed.
        """

        return self.plan(
            participants, map_, frame, agent_ids=agent_ids, decider=decider
        ).trajectories

    # -- rollout and conflict resolution ---------------------------------

    def _baseline_speeds(self, record: AgentRecord) -> np.ndarray:
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

    def _poses(self, record: AgentRecord, speeds: np.ndarray) -> np.ndarray:
        arcs = record.s0 + arcs_from_speeds(speeds, self.config.dt)
        return record.path.sample_poses(arcs)

    def _resolve_conflicts(self, records, baseline, edges, decider=None):
        """Return the braked speed profiles and the set of yielding agents.

        Args:
            records (Dict): The scene records keyed by agent id.
            baseline (Dict): The unbraked speed profiles.
            edges (List): The directed relations detected on the baselines.
            decider (Callable, optional): Direction arbiter used when
                ``relation_mode="nn"``. Defaults to None.

        Returns:
            A tuple of the final speed profiles and the ids of the agents that
            braked for a conflict.
        """

        config = self.config
        horizon = config.horizon_steps
        final_speeds = dict(baseline)
        yielded: set = set()

        # Map each potential reactor to the influencers it must brake for.
        # "directed" only brakes the later arrival; "yield_all" (all-yield
        # reference) brakes both sides of every imminent conflict. In "nn" mode
        # an optional ``decider`` (the learned predictor plus upstream rule
        # prefilter) overrides the direction per vehicle-vehicle pair, with the
        # geometric edge kept whenever the predictor is not confident.
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
                if (
                    config.relation_mode == "directed_tie"
                    and abs(edge.frame_diff) <= _NO_EDGE_TIE_FRAMES
                ):
                    actor_yields = partner_yields = True
                elif config.relation_mode == "nn" and decider is not None:
                    first_yields = decider(first, second)
                    second_yields = decider(second, first)
                    if first_yields is not None and second_yields is not None:
                        actor_yields, partner_yields = first_yields, second_yields
                if actor_yields:
                    reactor_influencers.setdefault(first, []).append(second)
                if partner_yields:
                    reactor_influencers.setdefault(second, []).append(first)
            else:
                for agent_id, counter_id in (
                    (edge.influencer, edge.reactor),
                    (edge.reactor, edge.influencer),
                ):
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
                yielded.add(reactor_id)
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
        return final_speeds, yielded

    def _apply_traffic_lights(self, records, final_speeds, yielded, frame, map_):
        """Stop free-running vehicles before a red traffic light stop point.

        Args:
            records (Dict): The scene records keyed by agent id.
            final_speeds (Dict): The speed profiles, updated in place.
            yielded (set): The ids of agents already braking for a conflict.
            frame (int): The current frame number. The unit is millisecond (ms).
            map_ (Optional[Map]): The map carrying the traffic light regulations.
        """

        config = self.config
        if not config.respect_traffic_light or map_ is None:
            return
        lights = [reg for reg in map_.regulations.values() if reg.is_traffic_light()]
        if not lights:
            return
        red_states = {"stop", "arrow_stop", "flashing_stop"}
        for agent_id, record in records.items():
            if not record.is_vehicle or record.lane_id is None or agent_id in yielded:
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
        record_a: AgentRecord,
        record_b: AgentRecord,
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
                float(pose_a[0]),
                float(pose_a[1]),
                float(pose_a[3]),
                record_a.length,
                record_a.width,
            )
            body_b = AgentBody(
                float(pose_b[0]),
                float(pose_b[1]),
                float(pose_b[3]),
                record_b.length,
                record_b.width,
            )
            if check_body_collision(body_a, body_b, margin):
                return step
        return None

    # -- serialization ----------------------------------------------------

    def _to_trajectory(
        self,
        record: AgentRecord,
        speeds: np.ndarray,
        poses: np.ndarray,
        frame: int,
    ) -> Trajectory:
        config = self.config
        trajectory = Trajectory(
            id_=record.agent_id, fps=round(1.0 / config.dt, 3), stable_freq=True
        )
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

    # -- closed-loop replay ------------------------------------------------

    def run_closed_loop(
        self,
        participants: Dict[object, object],
        map_: Optional[Map],
        ego_id: object,
        frame_ms0: int = 0,
    ) -> RollingSimulationResult:
        """Replay the scenario closed-loop and return its outcome.

        The runner mirrors the upstream InterSim cadence: the scenario is carried
        by per-agent pose arrays that start from the parsed ground truth and are
        overwritten with committed plans at planning frames. Only an ego-centric
        relevant set (ego plus vehicles whose future conflicts with it) is
        re-planned; the remaining vehicles keep their ground-truth motion, so the
        ego can collide with exactly the same kind of imperfectly predicted agents
        the original closed loop collided with.

        Args:
            participants (Dict): All participants in the scenario.
            map_ (Optional[Map]): The map. None falls back to straight paths.
            ego_id (object): The agent the loop is centred on.
            frame_ms0 (int, optional): Time stamp of index 0. Defaults to 0.

        Returns:
            The closed-loop outcome, carrying the upstream-aligned metrics and
            the final per-index poses.
        """

        step_ms = self.config.step_ms
        poses, dims, types = replay.extract_arrays(self.config, participants, step_ms, frame_ms0)
        goals: Dict[object, Optional[Tuple[float, float]]] = {
            agent_id: (
                (participant.trajectory.last_state.x, participant.trajectory.last_state.y)
                if participant.trajectory.last_state is not None
                else None
            )
            for agent_id, participant in participants.items()
        }
        if self.config.augment_lane_graph and map_ is not None:
            augment_lane_successors(map_)
        steps = self.config.scenario_steps
        warmup = self.config.planning_warmup_steps
        interval = self.config.planning_interval

        relevant_union: List[object] = []
        collided = False
        collision_kind = [0, 0, 0]  # front, side, rear
        end_index = min(89, steps - 1)

        current = 1
        while current <= end_index:
            if (current - warmup) >= 0 and (current - warmup) % interval == 0:
                relevant = self._plan_once(
                    poses, dims, types, participants, map_, ego_id, current, frame_ms0, goals
                )
                for agent_id in relevant:
                    if agent_id not in relevant_union:
                        relevant_union.append(agent_id)
            kind = replay.ego_collision_at(poses, dims, ego_id, current)
            if kind is not None:
                collision_kind[kind] += 1
                collided = True
                end_index = current
                break
            current += 1

        progress, controlled = replay.progress(
            self.config, poses, ego_id, relevant_union, end_index
        )
        return RollingSimulationResult(
            front_collisions=collision_kind[0],
            side_collisions=collision_kind[1],
            rear_collisions=collision_kind[2],
            progress=progress,
            total_agents_controlled=controlled,
            collided=collided,
            end_index=end_index,
            ego_id=ego_id,
            relevant_ids=list(relevant_union),
            poses={agent_id: poses[agent_id].copy() for agent_id in poses},
        )

    def _plan_once(self, poses, dims, types, participants, map_, ego_id, current, frame_ms0, goals):
        """Plan ego first, then its relevant environment, and commit both.

        Mirroring the upstream ordering (ego base planner commits before the env
        relevant detection), the ego is planned alone, its newly committed
        future is used to grow the relevant set, and only then are the relevant
        environment vehicles re-planned so they brake for the ego.
        """

        horizon = self.config.horizon_steps
        steps = self.config.scenario_steps
        frame_ms = frame_ms0 + current * self.config.step_ms
        snapshot, _ = replay.snapshot(
            self.config, poses, dims, types, participants, ego_id, current, frame_ms0, goals
        )
        if not snapshot:
            return []

        relevant = replay.relevant_ids(poses, dims, ego_id, current, horizon, steps)
        decider = None
        if self.config.relation_mode == "nn":
            decider = relation_decider.make_decider(
                self.config, poses, dims, types, map_, ego_id, current
            )

        planned = []
        ego_trajectories = self.predict(
            snapshot, map_, frame_ms, agent_ids=[ego_id], decider=decider
        )
        if ego_id in ego_trajectories:
            replay.commit(
                self.config, poses, ego_id, ego_trajectories[ego_id], current, steps, frame_ms0
            )
            planned.append(ego_id)

        env_ids = [agent_id for agent_id in relevant if agent_id != ego_id and agent_id in snapshot]
        if env_ids:
            env_trajectories = self.predict(
                snapshot, map_, frame_ms, agent_ids=env_ids, decider=decider
            )
            for agent_id, trajectory in env_trajectories.items():
                replay.commit(self.config, poses, agent_id, trajectory, current, steps, frame_ms0)
                planned.append(agent_id)
        return planned
