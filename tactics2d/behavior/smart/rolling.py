# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Adapted from SMART (github.com/rainmaker22/SMART), Apache-2.0.

"""Closed-loop joint rollout for the SMART port."""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from tactics2d.geometry import spatial
from tactics2d.map.element import Map
from tactics2d.participant.element import Cyclist, Pedestrian, Vehicle
from tactics2d.participant.trajectory import State, Trajectory

from ..results import ClosedLoopMetrics
from ..rollout_metrics import collision_kind, progress_window
from ..trajectory_processing import resample_participants
from .config import SmartConfig


@dataclass
class SmartRollingResult:
    """SMART-specific rollout state plus shared closed-loop metrics."""

    metrics: ClosedLoopMetrics = field(default_factory=ClosedLoopMetrics)
    ego_id: object = None
    modelled_ids: List[object] = field(default_factory=list)
    poses: Dict[object, np.ndarray] = field(default_factory=dict)


# Fallback extents for a participant that carries none of its own.
_DEFAULT_VEHICLE_LENGTH = 4.8
_DEFAULT_VEHICLE_WIDTH = 1.9
_DEFAULT_VRU_LENGTH = 0.6
_DEFAULT_VRU_WIDTH = 0.5

# Progress is summed over this frame window, skipping steps that teleport.
_PROGRESS_START = 12
_PROGRESS_END = 80
_MAX_PROGRESS_STEP = 20.0

_SNAP_CLASSES = (Vehicle, Pedestrian, Cyclist)


def _body_extent(participant) -> Tuple[float, float]:
    """Return a participant's collision extent as length and width.

    Args:
        participant: A tactics2d participant.

    Returns:
        The extent in metres, falling back to a per-class default.
    """

    length = getattr(participant, "length", None)
    width = getattr(participant, "width", None)
    if isinstance(participant, Vehicle):
        fallback = (_DEFAULT_VEHICLE_LENGTH, _DEFAULT_VEHICLE_WIDTH)
    else:
        fallback = (_DEFAULT_VRU_LENGTH, _DEFAULT_VRU_WIDTH)
    return (
        float(length) if length and length > 0 else fallback[0],
        float(width) if width and width > 0 else fallback[1],
    )


def extract_poses(
    participants: Dict[object, object], step_ms: int, frame_ms0: int, steps: int
) -> Tuple[Dict[object, np.ndarray], Dict[object, Tuple[float, float]], Dict[object, type]]:
    """Lay every participant's ground truth into indexed pose arrays.

    Each array has one row of ``(x, y, z, yaw)`` per scenario step, with a row
    of ``-1`` marking a step the participant does not occupy.

    Args:
        participants (Dict[object, object]): All participants in the scenario.
        step_ms (int): Interval between scenario steps, in milliseconds.
        frame_ms0 (int): Timestamp of index 0, in milliseconds.
        steps (int): Number of scenario steps.

    Returns:
        Per-agent pose arrays, ``(length, width)`` extents and classes, by id.
    """

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
        dims[agent_id] = _body_extent(participant)
        types[agent_id] = type(participant)
    return poses, dims, types


def _speed_at(poses, agent_id, index: int, step_ms: int) -> float:
    """Estimate a speed from the pose array's most recent usable pair.

    Args:
        poses (Dict[object, np.ndarray]): The rolling pose arrays.
        agent_id (object): The agent to measure.
        index (int): Step the speed is wanted at.
        step_ms (int): Interval between scenario steps, in milliseconds.

    Returns:
        The speed in metres per second, zero when no recent pair exists.
    """

    for back in (1, 2, 3, 4, 5):
        past = index - back
        if past < 0:
            continue
        pose_past = poses[agent_id][past]
        if pose_past[0] == -1:
            continue
        pose_now = poses[agent_id][index]
        distance = float(np.hypot(pose_now[0] - pose_past[0], pose_now[1] - pose_past[1]))
        return distance / (back * step_ms / 1000.0)
    return 0.0


def _history_trajectory(
    config: SmartConfig, poses, agent_id, current: int, frame_ms0: int
) -> Trajectory:
    """Wrap an agent's rolling history window as a ``Trajectory``.

    The window is read back out of the pose arrays, so the model conditions on
    its own past commitments rather than the log's ground truth.

    Args:
        config (SmartConfig): The port configuration.
        poses (Dict[object, np.ndarray]): The rolling pose arrays.
        agent_id (object): The agent to wrap.
        current (int): Index of the newest history frame.
        frame_ms0 (int): Timestamp of index 0, in milliseconds.

    Returns:
        A trajectory holding one state per occupied frame of the window.
    """

    step_ms = config.step_ms
    trajectory = Trajectory(id_=agent_id, fps=round(1000.0 / step_ms, 3), stable_freq=True)
    for index in range(max(0, current - config.history_steps + 1), current + 1):
        pose = poses[agent_id][index]
        if pose[0] == -1:
            continue
        heading = float(pose[3])
        speed = _speed_at(poses, agent_id, index, step_ms)
        trajectory.add_state(
            State(
                frame=int(frame_ms0 + index * step_ms),
                x=float(pose[0]),
                y=float(pose[1]),
                heading=heading,
                vx=speed * np.cos(heading),
                vy=speed * np.sin(heading),
            )
        )
    return trajectory


def snapshot(
    config: SmartConfig, poses, dims, types, participants, ego_id, current: int, frame_ms0: int
) -> Tuple[Dict[object, object], List[object]]:
    """Build a history-window participant snapshot around the ego at a frame.

    Args:
        config (SmartConfig): The port configuration.
        poses (Dict[object, np.ndarray]): The rolling pose arrays.
        dims (Dict[object, Tuple[float, float]]): Per-agent collision extents.
        types (Dict[object, type]): Per-agent participant classes.
        participants (Dict[object, object]): All participants.
        ego_id (object): The agent the loop is centred on.
        current (int): Step the snapshot is taken at.
        frame_ms0 (int): Timestamp of index 0, in milliseconds.

    Returns:
        The snapshot participants keyed by id, and the order they were added in.
    """

    ego_pose = poses[ego_id][current]
    if ego_pose[0] == -1:
        return {}, []

    result: Dict[object, object] = {}
    order: List[object] = []
    for agent_id in participants:
        if types[agent_id] not in _SNAP_CLASSES:
            continue
        pose = poses[agent_id][current]
        if pose[0] == -1:
            continue
        if agent_id != ego_id:
            reach = np.hypot(pose[0] - ego_pose[0], pose[1] - ego_pose[1])
            if reach > config.valid_radius:
                continue
        trajectory = _history_trajectory(config, poses, agent_id, current, frame_ms0)
        length, width = dims[agent_id]
        result[agent_id] = types[agent_id](
            agent_id,
            participants[agent_id].type_,
            trajectory=trajectory,
            length=length,
            width=width,
        )
        order.append(agent_id)
    return result, order


def commit(
    config: SmartConfig,
    poses,
    agent_id,
    trajectory: Trajectory,
    current: int,
    steps: int,
    frame_ms0: int,
) -> int:
    """Write a planned trajectory into the pose arrays from ``current`` on.

    Args:
        config (SmartConfig): The port configuration.
        poses (Dict[object, np.ndarray]): The rolling pose arrays.
        agent_id (object): The agent whose plan is being committed.
        trajectory (Trajectory): The planned trajectory, in the world frame.
        current (int): Step the plan was made at; the pose there is kept.
        steps (int): Number of scenario steps.
        frame_ms0 (int): Timestamp of index 0, in milliseconds.

    Returns:
        How many poses were written.
    """

    written = 0
    for frame_ms_future in trajectory.frames:
        index = int(round((frame_ms_future - frame_ms0) / config.step_ms))
        if index <= current or index >= steps:
            continue
        state = trajectory.get_state(frame_ms_future)
        poses[agent_id][index] = (state.x, state.y, 0.0, float(state.heading))
        written += 1
    return written


class SmartRollingRunner:
    """Replay a scenario closed-loop with the joint decoder.

    Every ``planning_interval`` steps after a ``planning_warmup_steps`` warmup
    the whole modelled set is decoded once and committed together.

    Attributes:
        model (SmartBehaviorModel): The port model being replayed.
        config (SmartConfig): The port configuration.
        warmup_steps (int): Steps of ground truth before the first replan.
        planning_interval (int): Steps between replans.
        scenario_steps (int): Number of steps in a scenario.
    """

    def __init__(
        self,
        model,
        config: Optional[SmartConfig] = None,
        warmup_steps: Optional[int] = None,
        planning_interval: int = 10,
        scenario_steps: Optional[int] = None,
    ):
        """Initialize the runner.

        Args:
            model (SmartBehaviorModel): The model to replay, exposing
                ``predict_scene``.
            config (Optional[SmartConfig], optional): Port configuration.
                Defaults to None, which uses the model's own.
            warmup_steps (Optional[int], optional): Steps before the first
                replan. Defaults to ``config.history_steps``.
            planning_interval (int, optional): Steps between replans. Defaults
                to 10, i.e. one replan per second at 10 Hz.
            scenario_steps (Optional[int], optional): Steps in a scenario.
                Defaults to ``history_steps + future_steps``.
        """

        self.model = model
        self.config = config if config is not None else model.config
        self.warmup_steps = self.config.history_steps if warmup_steps is None else int(warmup_steps)
        self.planning_interval = int(planning_interval)
        self.scenario_steps = (
            self.config.history_steps + self.config.future_steps
            if scenario_steps is None
            else int(scenario_steps)
        )

    def run(
        self,
        participants: Dict[object, object],
        map_: Optional[Map],
        ego_id: object,
        frame_ms0: Optional[int] = None,
        controlled_ids: Optional[Iterable[object]] = None,
    ) -> SmartRollingResult:
        """Replay the scenario closed-loop and return its outcome.

        Args:
            participants (Dict[object, object]): All participants.
            map_ (Optional[Map]): The map, or None to fail on the first replan.
            ego_id (object): The agent the loop is centred on.
            frame_ms0 (Optional[int], optional): Timestamp of index 0, in
                milliseconds. Defaults to None, which uses the scenario's own
                first observed frame - pass this only for a scenario whose frames
                are stamped relative to another origin.
            controlled_ids (Optional[Iterable], optional): The set of agents to
                commit each cycle. Defaults to None, which commits every agent
                the joint decoder returns. When given, *ego_id* must be a member
                and only ids in the set are committed.

                SMART may commit only a subset of the requested ids; read
                ``total_agents_controlled`` for the count actually controlled.

        Returns:
            The closed-loop outcome, with the metrics and the final poses.

        Raises:
            KeyError: If *ego_id* is not a participant.
            ValueError: If *controlled_ids* is given without *ego_id* in it.
        """

        if ego_id not in participants:
            raise KeyError(f"ego_id {ego_id!r} is not a participant of this scenario.")
        if controlled_ids is not None:
            controlled_ids = list(controlled_ids)
            if ego_id not in controlled_ids:
                raise ValueError(
                    "controlled_ids must contain ego_id {!r}; got {!r}".format(
                        ego_id, controlled_ids
                    )
                )
        participants = resample_participants(participants, self.config.step_ms)
        if frame_ms0 is None:
            # Default the window origin to the ego's first frame.
            frame_ms0 = int(participants[ego_id].trajectory.first_frame)
        step_ms = self.config.step_ms
        poses, dims, types = extract_poses(participants, step_ms, frame_ms0, self.scenario_steps)

        modelled_union: List[object] = []
        kinds = [0, 0, 0]  # front, side, rear
        collided = False
        # Index 0 is the last history frame, not a simulated step, so the loop
        # starts one step in and ends after one whole modelled future.
        end_index = self.scenario_steps - 2

        current = 1
        while current <= end_index:
            if (current - self.warmup_steps) >= 0 and (current - self.warmup_steps) % (
                self.planning_interval
            ) == 0:
                frame_ms = frame_ms0 + current * step_ms
                scene, _ = snapshot(
                    self.config, poses, dims, types, participants, ego_id, current, frame_ms0
                )
                if scene:
                    requested = [ego_id] if controlled_ids is None else controlled_ids
                    prediction = self.model.predict_scene(
                        scene, map_, frame_ms, agent_ids=requested
                    )
                    for agent_id in prediction.agent_ids:
                        if controlled_ids is not None and agent_id not in controlled_ids:
                            continue
                        commit(
                            self.config,
                            poses,
                            agent_id,
                            prediction.trajectory(agent_id),
                            current,
                            self.scenario_steps,
                            frame_ms0,
                        )
                        if agent_id not in modelled_union:
                            modelled_union.append(agent_id)
            kind = collision_kind(poses, dims, ego_id, current, "linear")
            if kind is not None:
                kinds[kind] += 1
                collided = True
                end_index = current
                break
            current += 1

        total_progress = 0.0
        for agent_id in modelled_union:
            total, _ = progress_window(poses, agent_id, end_index, self.scenario_steps)
            total_progress += total
        controlled = len(modelled_union)
        return SmartRollingResult(
            metrics=ClosedLoopMetrics(
                front_collisions=kinds[0],
                side_collisions=kinds[1],
                rear_collisions=kinds[2],
                progress=total_progress,
                total_agents_controlled=controlled,
                collided=collided,
            ),
            ego_id=ego_id,
            modelled_ids=list(modelled_union),
            poses={agent_id: poses[agent_id].copy() for agent_id in poses},
        )
