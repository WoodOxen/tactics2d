# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Receding-horizon closed-loop replay for the InterSim behavior model."""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from tactics2d.geometry import spatial
from tactics2d.map.element import Map
from tactics2d.participant.element import Cyclist, Pedestrian, Vehicle
from tactics2d.participant.trajectory import State, Trajectory
from tactics2d.routing.graph_builder import augment_lane_successors

from ..results import ClosedLoopMetrics
from ..rollout_metrics import collision_kind, progress_window
from ..trajectory_processing import resample_participants
from . import relation_decider
from .config import InterSimConfig
from .relation_geometry import AgentBody, check_body_collision

# Adapted from InterSim (github.com/Tsinghua-MARS-Lab/InterSim), MIT,
# Copyright (c) 2022 Tsinghua MARS Lab.

_SNAP_CLASSES = (Vehicle, Pedestrian, Cyclist)
_SNAP_RADIUS = 150.0


@dataclass
class InterSimRollingResult:
    """Per-scenario closed-loop outcome and its collision metrics."""

    metrics: ClosedLoopMetrics = field(default_factory=ClosedLoopMetrics)
    offroad_scenarios: int = 0
    end_index: int = 0
    ego_id: object = None
    relevant_ids: List[object] = field(default_factory=list)
    # Final per-index poses (x, y, z, yaw; -1 marks invalid) per agent.
    poses: Dict[object, np.ndarray] = field(default_factory=dict)


@dataclass
class ReplayState:
    """The indexed pose arrays one closed-loop replay steps over.

    Ground truth is laid out per agent on a common frame grid; ``poses`` is
    overwritten in place as plans are committed.
    """

    config: InterSimConfig
    participants: Dict[object, object]
    poses: Dict[object, np.ndarray]
    dims: Dict[object, Tuple[float, float]]
    types: Dict[object, type]
    base_frame_ms: int = 0

    @classmethod
    def from_participants(
        cls, config: InterSimConfig, participants, base_frame_ms: int
    ) -> "ReplayState":
        """Lay every participant's ground truth into indexed pose arrays."""

        steps = config.scenario_steps
        step_ms = config.step_ms
        poses: Dict[object, np.ndarray] = {}
        dims: Dict[object, Tuple[float, float]] = {}
        types: Dict[object, type] = {}
        for agent_id, participant in participants.items():
            array = np.full((steps, 4), -1.0, dtype=float)
            for frame_ms in participant.trajectory.frames:
                index = int(round((frame_ms - base_frame_ms) / step_ms))
                if index < 0 or index >= steps:
                    continue
                state = participant.trajectory.get_state(frame_ms)
                array[index, :] = (
                    state.x,
                    state.y,
                    0.0,
                    spatial.normalize_angle(float(state.heading)),
                )
            poses[agent_id] = array
            length = participant.length
            width = participant.width
            if isinstance(participant, Vehicle):
                dims[agent_id] = (
                    float(length if length and length > 0 else config.default_vehicle_length),
                    float(width if width and width > 0 else config.default_vehicle_width),
                )
            else:
                dims[agent_id] = (
                    float(length if length and length > 0 else 0.6),
                    float(width if width and width > 0 else 0.5),
                )
            types[agent_id] = type(participant)
        return cls(
            config=config,
            participants=participants,
            poses=poses,
            dims=dims,
            types=types,
            base_frame_ms=base_frame_ms,
        )


def snapshot(
    state: ReplayState, ego_id, current: int, goals: Dict[object, Optional[Tuple[float, float]]]
) -> Dict[object, object]:
    """Build a one-state participant snapshot around the ego at a frame."""

    config = state.config
    poses = state.poses
    step_ms = config.step_ms
    frame_ms = state.base_frame_ms + current * step_ms
    ego_pose = poses[ego_id][current]
    if ego_pose[0] == -1:
        return {}
    result = {}
    for agent_id in state.participants:
        if state.types[agent_id] not in _SNAP_CLASSES:
            continue
        pose = poses[agent_id][current]
        if pose[0] == -1:
            continue
        if agent_id != ego_id:
            if np.hypot(pose[0] - ego_pose[0], pose[1] - ego_pose[1]) > _SNAP_RADIUS:
                continue
        speed = _speed_at(state, agent_id, current)
        intent_speed = _recent_intent_speed(state, agent_id, current)
        participant = _wrap_at(
            state,
            state.participants[agent_id],
            agent_id,
            pose,
            frame_ms,
            speed,
            intent_speed,
            goals.get(agent_id),
        )
        result[agent_id] = participant
    return result


def relevant_ids(state: ReplayState, ego_id, current: int, horizon: int) -> List[object]:
    """Grow a relevant set from the ego over future body collisions."""

    poses = state.poses
    dims = state.dims
    steps = state.config.scenario_steps
    seen = {ego_id}
    queue = [ego_id]
    result = [ego_id]
    while queue:
        agent_id = queue.pop()
        for index in range(current + 1, min(steps, current + horizon + 1)):
            pose_a = poses[agent_id][index]
            if pose_a[0] == -1:
                continue
            body_a = AgentBody(
                float(pose_a[0]), float(pose_a[1]), float(pose_a[3]), *dims[agent_id]
            )
            for other_id in poses:
                if other_id in seen:
                    continue
                pose_b = poses[other_id][index]
                if pose_b[0] == -1:
                    continue
                body_b = AgentBody(
                    float(pose_b[0]), float(pose_b[1]), float(pose_b[3]), *dims[other_id]
                )
                if not check_body_collision(body_a, body_b):
                    continue
                seen.add(other_id)
                queue.append(other_id)
                result.append(other_id)
    return result


def commit(state: ReplayState, agent_id, trajectory, current: int) -> None:
    """Write one planned trajectory into the pose arrays from ``current`` on."""

    steps = state.config.scenario_steps
    step_ms = state.config.step_ms
    base_frame_ms = state.base_frame_ms
    poses = state.poses
    for frame_ms_future in trajectory.frames:
        index = int(round((frame_ms_future - base_frame_ms) / step_ms))
        if index <= current or index >= steps:
            continue
        trajectory_state = trajectory.get_state(frame_ms_future)
        poses[agent_id][index] = (
            trajectory_state.x,
            trajectory_state.y,
            0.0,
            float(trajectory_state.heading),
        )


def ego_collision_at(state: ReplayState, ego_id, index: int) -> Optional[int]:
    """Classify a same-frame ego collision at one index, if any.

    Returns:
        ``0`` for front, ``1`` for side, ``2`` for rear, or ``None`` when the
        ego does not overlap any other body at ``index``.
    """

    return collision_kind(state.poses, state.dims, ego_id, index, "factor")


def progress(
    state: ReplayState, ego_id, relevant_ids: List[object], end_index: int
) -> Tuple[float, int]:
    """Sum per-step displacements over the frames up to ``end_index``.

    Returns:
        A tuple ``(progress, count)`` of the summed displacement in meters and
        the number of controlled agents it was summed over.
    """

    steps = state.config.scenario_steps
    controlled = list(relevant_ids)
    if ego_id not in controlled:
        controlled.append(ego_id)
    total_progress = 0.0
    for agent_id in controlled:
        total, _ = progress_window(state.poses, agent_id, end_index, steps)
        total_progress += total
    return total_progress, len(controlled)


def _recent_intent_speed(state: ReplayState, agent_id, current: int) -> float:
    """Return the agent's own recent cruising speed (no future used)."""

    config = state.config
    poses = state.poses
    best = 0.0
    for index in range(max(0, current - 10), current):
        if index + 1 >= config.scenario_steps:
            break
        pose_i = poses[agent_id][index]
        pose_j = poses[agent_id][index + 1]
        if pose_i[0] == -1 or pose_j[0] == -1:
            continue
        speed = float(np.hypot(pose_j[0] - pose_i[0], pose_j[1] - pose_i[1])) / config.dt
        best = max(best, speed)
    return best


def _speed_at(state: ReplayState, agent_id, current: int) -> float:
    """Estimate current speed from a recent finite difference."""

    config = state.config
    poses = state.poses
    for back in (5, 4, 3, 2, 1):
        index = current - back
        if index < 0:
            continue
        pose_past = poses[agent_id][index]
        if pose_past[0] == -1:
            continue
        pose_now = poses[agent_id][current]
        distance = float(np.hypot(pose_now[0] - pose_past[0], pose_now[1] - pose_past[1]))
        return distance / (back * config.dt)
    return 0.0


def _wrap_at(
    state: ReplayState,
    participant,
    agent_id,
    pose,
    frame_ms: int,
    speed: float,
    intent_speed: float,
    goal,
):
    """Rebuild one participant as a single-state wrapper at ``frame_ms``."""

    length, width = state.dims[agent_id]
    heading = spatial.normalize_angle(float(pose[3]))
    trajectory = Trajectory(id_=agent_id, fps=round(1.0 / state.config.dt, 3), stable_freq=True)
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
    wrapper = cls(agent_id, participant.type_, trajectory=trajectory, length=length, width=width)
    if goal is not None:
        wrapper.goal_xy = goal
    if intent_speed > 0.0:
        wrapper.intent_speed = intent_speed
    return wrapper


class InterSimRollingRunner:
    """Replay a scenario closed-loop around one ego.

    Attributes:
        model (InterSimBehaviorModel): The model being replayed.
        config (InterSimConfig): The model's configuration.
    """

    def __init__(self, model, config: Optional[InterSimConfig] = None):
        """Initialize the runner.

        Args:
            model (InterSimBehaviorModel): The model to replay, exposing ``predict``.
            config (Optional[InterSimConfig], optional): Configuration to replay with.
                Defaults to None, which uses the model's own.
        """

        self.model = model
        self.config = config if config is not None else model.config

    def run(
        self,
        participants: Dict[object, object],
        map_: Optional[Map],
        ego_id: object,
        base_frame_ms: Optional[int] = None,
        controlled_ids: Optional[Iterable[object]] = None,
    ) -> InterSimRollingResult:
        """Replay the scenario closed-loop and return its outcome.

        Args:
            participants (Dict): All participants in the scenario.
            map_ (Optional[Map]): The map. None falls back to straight paths.
            ego_id (object): The agent the loop is centred on.
            base_frame_ms (Optional[int], optional): Time stamp of index 0, in
                milliseconds. Defaults to None, which uses the scenario's own
                first observed frame - pass this only for a scenario whose frames
                are stamped relative to another origin.
            controlled_ids (Optional[Iterable], optional): The exact set of
                agents to plan and commit each cycle. Defaults to None, which
                grows the set from the ego over future body collisions. When
                given, *ego_id* must be a member; the ego keeps its meaning as
                the loop's centre, the snapshot anchor and the agent collisions
                are attributed to.

        Returns:
            The closed-loop outcome with its metrics and final per-index poses.

        Raises:
            ValueError: If *controlled_ids* is given without *ego_id* in it.
        """

        if controlled_ids is not None:
            controlled_ids = list(controlled_ids)
            if ego_id not in controlled_ids:
                raise ValueError(
                    "controlled_ids must contain ego_id {!r}; got {!r}".format(
                        ego_id, controlled_ids
                    )
                )

        participants = resample_participants(participants, self.config.step_ms)
        if base_frame_ms is None:
            base_frame_ms = int(participants[ego_id].trajectory.first_frame)
        state = ReplayState.from_participants(self.config, participants, base_frame_ms)
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
        kinds = [0, 0, 0]  # front, side, rear
        end_index = min(89, steps - 1)

        current = 1
        while current <= end_index:
            if (current - warmup) >= 0 and (current - warmup) % interval == 0:
                relevant = self._plan_once(state, map_, ego_id, current, goals, controlled_ids)
                for agent_id in relevant:
                    if agent_id not in relevant_union:
                        relevant_union.append(agent_id)
            kind = ego_collision_at(state, ego_id, current)
            if kind is not None:
                kinds[kind] += 1
                collided = True
                end_index = current
                break
            current += 1

        progress_total, controlled = progress(state, ego_id, relevant_union, end_index)
        return InterSimRollingResult(
            metrics=ClosedLoopMetrics(
                front_collisions=kinds[0],
                side_collisions=kinds[1],
                rear_collisions=kinds[2],
                progress=progress_total,
                total_agents_controlled=controlled,
                collided=collided,
            ),
            end_index=end_index,
            ego_id=ego_id,
            relevant_ids=list(relevant_union),
            poses={agent_id: array.copy() for agent_id, array in state.poses.items()},
        )

    def _plan_once(self, state, map_, ego_id, current, goals, controlled_ids=None):
        """Plan ego first, then its relevant environment, and commit both.

        When *controlled_ids* is given it replaces the collision-grown
        relevant set entirely.
        """

        horizon = self.config.horizon_steps
        frame_ms = state.base_frame_ms + current * self.config.step_ms
        nearby = snapshot(state, ego_id, current, goals)
        if not nearby:
            return []

        if controlled_ids is None:
            relevant = relevant_ids(state, ego_id, current, horizon)
        else:
            relevant = list(controlled_ids)
        decider = None
        if self.config.relation_mode == "nn":
            decider = relation_decider.make_decider(
                self.config, state.poses, state.types, map_, ego_id, current
            )

        planned = []
        ego_trajectories = self.model.predict(
            nearby, map_, frame_ms, agent_ids=[ego_id], decider=decider
        )
        if ego_id in ego_trajectories:
            commit(state, ego_id, ego_trajectories[ego_id], current)
            planned.append(ego_id)

        env_ids = [agent_id for agent_id in relevant if agent_id != ego_id and agent_id in nearby]
        if env_ids:
            env_trajectories = self.model.predict(
                nearby, map_, frame_ms, agent_ids=env_ids, decider=decider
            )
            for agent_id, trajectory in env_trajectories.items():
                commit(state, agent_id, trajectory, current)
                planned.append(agent_id)
        return planned
