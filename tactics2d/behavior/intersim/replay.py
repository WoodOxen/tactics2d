# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Closed-loop frame stepping and scenario metrics."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from tactics2d.geometry import spatial
from tactics2d.participant.element import Cyclist, Pedestrian, Vehicle
from tactics2d.participant.trajectory import State, Trajectory

from .config import InterSimConfig
from .relation_geometry import AgentBody, check_body_collision

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


def extract_arrays(config: InterSimConfig, participants, step_ms: int, frame_ms0: int):
    """Lay every participant's ground truth into indexed pose arrays.

    Returns:
        A tuple ``(poses, dims, types)`` of per-agent ``(steps, 4)`` pose arrays,
        ``(length, width)`` pairs, and participant classes.
    """

    steps = config.scenario_steps
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
    return poses, dims, types


def snapshot(
    config: InterSimConfig,
    poses,
    dims,
    types,
    participants,
    ego_id,
    current: int,
    frame_ms0: int,
    goals: Dict[object, Optional[Tuple[float, float]]],
) -> Tuple[Dict[object, object], List[object]]:
    """Build a one-state participant snapshot around the ego at a frame."""

    step_ms = config.step_ms
    frame_ms = frame_ms0 + current * step_ms
    ego_pose = poses[ego_id][current]
    if ego_pose[0] == -1:
        return {}, []
    result = {}
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
        speed = speed_at(config, poses, agent_id, current)
        intent_speed = recent_intent_speed(config, poses, agent_id, current)
        participant = wrap_at(
            config,
            participants[agent_id],
            agent_id,
            dims[agent_id],
            pose,
            frame_ms,
            speed,
            intent_speed,
            goals.get(agent_id),
        )
        result[agent_id] = participant
        order.append(agent_id)
    return result, order


def recent_intent_speed(config: InterSimConfig, poses, agent_id, current: int) -> float:
    """Return the agent's own recent cruising speed (no future used)."""

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


def speed_at(config: InterSimConfig, poses, agent_id, current: int) -> float:
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
        return distance / (back * config.dt)
    return 0.0


def wrap_at(
    config: InterSimConfig,
    participant,
    agent_id,
    dims,
    pose,
    frame_ms: int,
    speed: float,
    intent_speed: float,
    goal,
):
    """Rebuild one participant as a single-state wrapper at ``frame_ms``."""

    length, width = dims
    heading = spatial.normalize_angle(float(pose[3]))
    trajectory = Trajectory(id_=agent_id, fps=round(1.0 / config.dt, 3), stable_freq=True)
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


def commit(
    config: InterSimConfig,
    poses,
    agent_id,
    trajectory,
    current: int,
    steps: int,
    frame_ms0: int,
) -> None:
    """Write one planned trajectory into the pose arrays from ``current`` on."""

    for frame_ms_future in trajectory.frames:
        index = int(round((frame_ms_future - frame_ms0) / config.step_ms))
        if index <= current or index >= steps:
            continue
        state = trajectory.get_state(frame_ms_future)
        poses[agent_id][index] = (state.x, state.y, 0.0, float(state.heading))


def relevant_ids(poses, dims, ego_id, current: int, horizon: int, steps: int) -> List[object]:
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


def ego_collision_at(poses, dims, ego_id, index: int) -> Optional[int]:
    """Classify a same-frame ego collision at one index, if any.

    Returns:
        ``0`` for front, ``1`` for side, ``2`` for rear, or ``None`` when the
        ego does not overlap any other body at ``index``.
    """

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
        body_other = AgentBody(
            float(pose_other[0]), float(pose_other[1]), float(pose_other[3]), *dims[other_id]
        )
        if not check_body_collision(body_ego, body_other):
            continue
        diff = abs(spatial.normalize_angle(float(pose_ego[3]) - float(pose_other[3])))
        if diff < _REAR_TOL:
            return 2  # rear
        if diff > _SIDE_TOL:
            return 1  # side
        return 0  # front
    return None


def progress(
    config: InterSimConfig, poses, ego_id, relevant_ids: List[object], end_index: int
) -> Tuple[float, int]:
    """Sum per-step displacements over frames 12..79 for controlled agents.

    Returns:
        A tuple ``(progress, count)`` of the summed displacement in meters and
        the number of controlled agents it was summed over.
    """

    controlled = [agent_id for agent_id in relevant_ids]
    if ego_id not in controlled:
        controlled.append(ego_id)
    total_progress = 0.0
    count = 0
    for agent_id in controlled:
        total = 0.0
        for index in range(12, 80):
            if index >= end_index:
                break
            if index + 1 >= config.scenario_steps:
                break
            pose_i = poses[agent_id][index]
            pose_j = poses[agent_id][index + 1]
            if pose_i[0] == -1 or pose_j[0] == -1:
                break
            dist = float(np.hypot(pose_i[0] - pose_j[0], pose_i[1] - pose_j[1]))
            if dist >= 20.0:
                continue
            total += dist
        total_progress += total
        count += 1
    return total_progress, count
