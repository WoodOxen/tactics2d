# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Receding-horizon closed-loop runner for the InterSim-style model."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from tactics2d.geometry import spatial
from tactics2d.map.element import Map
from tactics2d.participant.element import Cyclist, Pedestrian, Vehicle
from tactics2d.participant.trajectory import State, Trajectory
from tactics2d.routing.utils import augment_lane_successors

from .config import InterSimConfig
from .model import InterSimBehaviorModel
from .relation import AgentBody, Edge, check_body_collision

# Adapted from InterSim (github.com/Tsinghua-MARS-Lab/InterSim), MIT,
# Copyright (c) 2022 Tsinghua MARS Lab.

# Collision classification thresholds by relative heading (mirroring upstream).
_REAR_TOL = 30.0 * np.pi / 180.0
_SIDE_TOL = 150.0 * np.pi / 180.0

_SNAP_CLASSES = (Vehicle, Pedestrian, Cyclist)
_SNAP_RADIUS = 150.0


@dataclass
class RollingSimulationResult:
    """Per-scenario closed-loop outcome with upstream-aligned metric fields."""

    front_collisions: int = 0
    side_collisions: int = 0
    rear_collisions: int = 0
    offroad_scenarios: int = 0
    progress: float = 0.0
    total_agents_controlled: int = 0
    collided: bool = False
    end_index: int = 0
    ego_id: object = None
    relevant_ids: List[object] = field(default_factory=list)
    # Final per-index poses (x, y, z, yaw; -1 marks invalid) per agent.
    poses: Dict[object, np.ndarray] = field(default_factory=dict)


class InterSimRollingRunner:
    """Run the relation-driven model over a WOMD-style scenario.

    The runner mirrors the upstream InterSim cadence: the scenario is carried
    by per-agent pose arrays that start from the parsed ground truth and are
    overwritten with committed plans at planning frames. Only an ego-centric
    relevant set (ego plus vehicles whose future conflicts with it) is
    re-planned; the remaining vehicles keep their ground-truth motion, so the
    ego can collide with exactly the same kind of imperfectly predicted agents
    the original closed loop collided with.
    """

    def __init__(self, config: Optional[InterSimConfig] = None):
        self.config = config or InterSimConfig()
        self.behavior_model = InterSimBehaviorModel(self.config)
        self._goals: Dict[object, Optional[Tuple[float, float]]] = {}
        self._nn_model = None

    # -- public entry ------------------------------------------------------

    def run(
        self,
        participants: Dict[object, object],
        map_: Optional[Map],
        ego_id: object,
        frame_ms0: int = 0,
    ) -> RollingSimulationResult:
        """Simulate one scenario and return its closed-loop outcome."""

        step_ms = self.config.step_ms
        poses, dims, types = self._extract_arrays(participants, step_ms, frame_ms0)
        self._goals = {
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
                relevant = self._plan_once(poses, dims, types, participants, map_, ego_id, current, frame_ms0)
                for agent_id in relevant:
                    if agent_id not in relevant_union:
                        relevant_union.append(agent_id)
            kind = self._ego_collision_at(poses, dims, ego_id, current)
            if kind is not None:
                collision_kind[kind] += 1
                collided = True
                end_index = current
                break
            current += 1

        progress, controlled = self._progress(poses, ego_id, relevant_union, end_index)
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

    # -- data preparation --------------------------------------------------

    def _extract_arrays(self, participants, step_ms, frame_ms0):
        steps = self.config.scenario_steps
        poses: Dict[object, np.ndarray] = {}
        dims: Dict[object, Tuple[float, float]] = {}
        types: Dict[object, type] = {}
        for agent_id, participant in participants.items():
            array = np.full((steps, 4), -1.0, dtype=float)
            for frame_ms in participant.trajectory.frames:
                index = int(round((frame_ms - frame_ms0) / step_ms))
                if index < 0 or index >= steps:
                    continue
                state = participant.trajectory.get_state(frame_ms)
                array[index, :] = (state.x, state.y, 0.0, spatial.normalize_angle(float(state.heading)))
            poses[agent_id] = array
            length = participant.length
            width = participant.width
            if isinstance(participant, Vehicle):
                dims[agent_id] = (
                    float(length if length and length > 0 else self.config.default_vehicle_length),
                    float(width if width and width > 0 else self.config.default_vehicle_width),
                )
            else:
                dims[agent_id] = (
                    float(length if length and length > 0 else 0.6),
                    float(width if width and width > 0 else 0.5),
                )
            types[agent_id] = type(participant)
        return poses, dims, types

    def _snapshot(
        self,
        poses,
        dims,
        types,
        participants,
        ego_id,
        current,
        frame_ms0,
    ) -> Tuple[Dict[object, object], List[object]]:
        """Build a one-state participant snapshot around the ego at a frame."""

        step_ms = self.config.step_ms
        frame_ms = frame_ms0 + current * step_ms
        ego_pose = poses[ego_id][current]
        if ego_pose[0] == -1:
            return {}, []
        snapshot = {}
        order = []
        for agent_id in participants:
            if types[agent_id] not in _SNAP_CLASSES:
                continue
            pose = poses[agent_id][current]
            if pose[0] == -1:
                continue
            if agent_id != ego_id:
                if np.hypot(pose[0] - ego_pose[0], pose[1] - ego_pose[1]) > _SNAP_RADIUS:
                    continue
            speed = self._speed_at(poses, agent_id, current)
            intent_speed = self._recent_intent_speed(poses, agent_id, current)
            participant = self._wrap_at(
                participants[agent_id],
                agent_id,
                dims[agent_id],
                pose,
                frame_ms,
                speed,
                intent_speed,
            )
            snapshot[agent_id] = participant
            order.append(agent_id)
        return snapshot, order

    def _recent_intent_speed(self, poses, agent_id, current) -> float:
        """Return the agent's own recent cruising speed (no future used)."""

        best = 0.0
        for index in range(max(0, current - 10), current):
            if index + 1 >= self.config.scenario_steps:
                break
            pose_i = poses[agent_id][index]
            pose_j = poses[agent_id][index + 1]
            if pose_i[0] == -1 or pose_j[0] == -1:
                continue
            speed = float(np.hypot(pose_j[0] - pose_i[0], pose_j[1] - pose_i[1])) / self.config.dt
            best = max(best, speed)
        return best

    def _speed_at(self, poses, agent_id, current) -> float:
        """Estimate current speed from a recent finite difference."""

        for back in (5, 4, 3, 2, 1):
            index = current - back
            if index < 0:
                continue
            pose_past = poses[agent_id][index]
            if pose_past[0] == -1:
                continue
            pose_now = poses[agent_id][current]
            distance = float(np.hypot(pose_now[0] - pose_past[0], pose_now[1] - pose_past[1]))
            return distance / (back * self.config.dt)
        return 0.0

    def _wrap_at(self, participant, agent_id, dims, pose, frame_ms, speed, intent_speed):
        length, width = dims
        heading = spatial.normalize_angle(float(pose[3]))
        trajectory = Trajectory(id_=agent_id, fps=round(1.0 / self.config.dt, 3), stable_freq=True)
        trajectory.add_state(
            State(
                frame=int(frame_ms),
                x=float(pose[0]),
                y=float(pose[1]),
                heading=heading,
                vx=speed * np.cos(heading),
                vy=speed * np.sin(heading),
            )
        )
        cls = type(participant)
        kwargs = {"length": length, "width": width}
        wrapper = cls(agent_id, participant.type_, trajectory=trajectory, **kwargs)
        goal = self._goals.get(agent_id) if hasattr(self, "_goals") else None
        if goal is not None:
            wrapper.goal_xy = goal
        if intent_speed > 0.0:
            wrapper.intent_speed = intent_speed
        return wrapper

    def _plan_once(self, poses, dims, types, participants, map_, ego_id, current, frame_ms0):
        """Plan ego first, then its relevant environment, and commit both.

        Mirroring the upstream ordering (ego base planner commits before the env
        relevant detection), the ego is planned alone, its newly committed
        future is used to grow the relevant set, and only then are the relevant
        environment vehicles re-planned so they brake for the ego.
        """

        horizon = self.config.horizon_steps
        steps = self.config.scenario_steps
        frame_ms = frame_ms0 + current * self.config.step_ms
        snapshot, _ = self._snapshot(poses, dims, types, participants, ego_id, current, frame_ms0)
        if not snapshot:
            return []

        relevant = self._relevant_ids(poses, dims, ego_id, current, horizon, steps)
        self.behavior_model._relations_override = None
        if self.config.relation_mode == "nn":
            self.behavior_model._relation_decider = self._make_nn_decider(
                poses, dims, types, map_, ego_id, current
            )
        else:
            self.behavior_model._relation_decider = None

        planned = []
        ego_trajectories = self.behavior_model.predict(snapshot, map_, frame_ms, agent_ids=[ego_id])
        if ego_id in ego_trajectories:
            self._commit(poses, ego_id, ego_trajectories[ego_id], current, steps, frame_ms0)
            planned.append(ego_id)

        env_ids = [agent_id for agent_id in relevant if agent_id != ego_id and agent_id in snapshot]
        if env_ids:
            env_trajectories = self.behavior_model.predict(snapshot, map_, frame_ms, agent_ids=env_ids)
            for agent_id, trajectory in env_trajectories.items():
                self._commit(poses, agent_id, trajectory, current, steps, frame_ms0)
                planned.append(agent_id)
        self.behavior_model._relations_override = None
        return planned

    def _road_graph(self, map_, cx: float, cy: float):
        """Sample nearby map lanes into roadgraph point/type/id arrays."""

        import tactics2d.behavior.intersim.m2i_features as features

        type_map = {"road": 2, "highway": 1, "bicycle_lane": 3}
        points = []
        types = []
        ids = []
        lane_id = 0
        for lane in map_.lanes.values():
            centerline = lane.centerline()
            if centerline is None or len(centerline.coords) < 2:
                continue
            coords = np.asarray(centerline.coords, dtype=float)
            lane_type = type_map.get(lane.subtype, 2) if lane.subtype else 2
            for index in range(0, len(coords), 2):
                x, y = coords[index]
                if abs(x - cx) <= 150.0 and -60.0 <= y - cy <= 170.0:
                    points.append([x, y, 0.0])
                    types.append(lane_type)
                    ids.append(lane_id)
            lane_id += 1
        if not points:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
            )
        return (
            np.asarray(points, dtype=np.float32),
            np.asarray(types, dtype=np.int32).reshape(-1),
            np.asarray(ids, dtype=np.int32).reshape(-1),
        )

    def _load_nn_model(self):
        if self._nn_model is None:
            from .relation_nn import RelationVectorNet

            if not self.config.relation_model_path:
                raise ValueError("relation_mode='nn' requires config.relation_model_path.")
            self._nn_model = RelationVectorNet.from_checkpoint(self.config.relation_model_path)
        return self._nn_model

    def _make_nn_decider(self, poses, dims, types, map_, ego_id, current):
        """Build the per-frame learned direction arbiter for relation_mode="nn".

        The model's own geometric detector still builds the conflict graph; this
        closure only decides the direction of each vehicle-vehicle pair, holding
        the .bin predictor (with the upstream rule prefilter) and returning
        ``None`` when it is not confident so the geometric edge is kept.
        """

        import torch

        ego_pose = poses[ego_id][current]
        road_points, road_types, road_ids = self._road_graph(
            map_, float(ego_pose[0]), float(ego_pose[1])
        )
        is_vehicle = {agent_id: types[agent_id] is Vehicle for agent_id in poses}
        model = self._load_nn_model()
        device = torch.device("cpu")
        cache = {}

        def decide(reactor_id, influencer_id):
            if not (is_vehicle.get(reactor_id, True) and is_vehicle.get(influencer_id, True)):
                return None
            key = (reactor_id, influencer_id)
            if key not in cache:
                cache[key] = self._nn_edge_yields(
                    reactor_id,
                    influencer_id,
                    poses,
                    current,
                    is_vehicle,
                    road_points,
                    road_types,
                    road_ids,
                    model,
                    device,
                )
            return cache[key]

        return decide
        return False

    def _nn_edge_yields(
        self,
        reactor_id,
        influencer_id,
        poses,
        current,
        is_vehicle,
        road_points,
        road_types,
        road_ids,
        model,
        device,
    ) -> Optional[bool]:
        """Whether the reactor must yield, using the upstream rule prefilter.

        Returns ``True``/``False`` when a rule or a confident predictor decides,
        and ``None`` when the predictor is not confident (the caller then keeps
        the geometric edge). Mirrors upstream: the same-direction (< 30 deg) rule
        yields for whoever faces the other's body, a non-vehicle partner is never
        forced to yield, and the .bin predictor only arbitrates the rest.
        """

        import tactics2d.behavior.intersim.m2i_features as features

        reactor_pose = poses[reactor_id][current]
        influencer_pose = poses[influencer_id][current]
        if reactor_pose[0] == -1 or influencer_pose[0] == -1:
            return None

        yaw = float(reactor_pose[3])
        target_yaw = float(influencer_pose[3])
        yaw_diff = abs(spatial.normalize_angle(yaw - target_yaw))
        if yaw_diff < np.pi / 6.0:
            heading = np.array([np.cos(yaw), np.sin(yaw)])
            offset = influencer_pose[:2] - reactor_pose[:2]
            return bool(float(np.dot(offset, heading)) > 0)
        if not is_vehicle.get(influencer_id, True):
            return True

        def window_7(agent_id):
            out = np.zeros((11, 7), dtype=np.float32)
            for j in range(11):
                index = current + j
                if index >= self.config.scenario_steps or poses[agent_id][index, 0] == -1:
                    continue
                out[j, 0] = poses[agent_id][index, 0]
                out[j, 1] = poses[agent_id][index, 1]
                out[j, 4] = poses[agent_id][index, 3]
            return out

        reactor = window_7(reactor_id)
        influencer = window_7(influencer_id)
        angle = -float(reactor[0, 4]) + np.pi / 2
        x0, y0 = float(reactor[0, 0]), float(reactor[0, 1])
        stacked = np.stack([reactor, influencer], axis=0)
        mapping = features.build_mapping(
            stacked,
            np.ones(2, dtype=np.int32),
            road_points,
            road_types,
            road_ids,
            (x0, y0, angle),
            [reactor_id, influencer_id],
            str(current),
        )
        scores = model.forward(
            mapping["matrix"], mapping["polyline_spans"], mapping["map_start_polyline_idx"], device
        )[0]
        if float(np.max(scores)) <= 0.5:
            return None
        return bool(np.argmax(scores) == 1)

    def _commit(self, poses, agent_id, trajectory, current, steps, frame_ms0):
        """Write one planned trajectory into the pose arrays from ``current`` on."""

        for frame_ms_future in trajectory.frames:
            index = int(round((frame_ms_future - frame_ms0) / self.config.step_ms))
            if index <= current or index >= steps:
                continue
            state = trajectory.get_state(frame_ms_future)
            poses[agent_id][index] = (state.x, state.y, 0.0, float(state.heading))

    # -- relevance and collision detection ---------------------------------

    def _relevant_ids(self, poses, dims, ego_id, current, horizon, steps) -> List[object]:
        """Grow a relevant set from the ego over future body collisions."""

        seen = {ego_id}
        queue = [ego_id]
        result = [ego_id]
        while queue:
            agent_id = queue.pop()
            for index in range(current + 1, min(steps, current + horizon + 1)):
                pose_a = poses[agent_id][index]
                if pose_a[0] == -1:
                    continue
                body_a = AgentBody(float(pose_a[0]), float(pose_a[1]), float(pose_a[3]), *dims[agent_id])
                for other_id in poses:
                    if other_id in seen:
                        continue
                    pose_b = poses[other_id][index]
                    if pose_b[0] == -1:
                        continue
                    body_b = AgentBody(float(pose_b[0]), float(pose_b[1]), float(pose_b[3]), *dims[other_id])
                    if not check_body_collision(body_a, body_b):
                        continue
                    seen.add(other_id)
                    queue.append(other_id)
                    result.append(other_id)
        return result

    def _ego_collision_at(self, poses, dims, ego_id, index) -> Optional[int]:
        """Classify a same-frame ego collision at one index, if any."""

        pose_ego = poses[ego_id][index]
        if pose_ego[0] == -1:
            return None
        body_ego = AgentBody(float(pose_ego[0]), float(pose_ego[1]), float(pose_ego[3]), *dims[ego_id])
        for other_id in poses:
            if other_id == ego_id:
                continue
            pose_other = poses[other_id][index]
            if pose_other[0] == -1:
                continue
            body_other = AgentBody(float(pose_other[0]), float(pose_other[1]), float(pose_other[3]), *dims[other_id])
            if not check_body_collision(body_ego, body_other):
                continue
            diff = abs(spatial.normalize_angle(float(pose_ego[3]) - float(pose_other[3])))
            if diff < _REAR_TOL:
                return 2  # rear
            if diff > _SIDE_TOL:
                return 1  # side
            return 0  # front
        return None

    # -- metrics -----------------------------------------------------------

    def _progress(self, poses, ego_id, relevant_ids, end_index) -> Tuple[float, int]:
        """Sum per-step displacements over frames 12..79 for controlled agents."""

        controlled = [agent_id for agent_id in relevant_ids]
        if ego_id not in controlled:
            controlled.append(ego_id)
        progress = 0.0
        count = 0
        for agent_id in controlled:
            total = 0.0
            for index in range(12, 80):
                if index >= end_index:
                    break
                if index + 1 >= self.config.scenario_steps:
                    break
                pose_i = poses[agent_id][index]
                pose_j = poses[agent_id][index + 1]
                if pose_i[0] == -1 or pose_j[0] == -1:
                    break
                dist = float(np.hypot(pose_i[0] - pose_j[0], pose_i[1] - pose_j[1]))
                if dist >= 20.0:
                    continue
                total += dist
            progress += total
            count += 1
        return progress, count
