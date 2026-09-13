# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Scene construction and reference paths."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from tactics2d.geometry import spatial
from tactics2d.map.element import Map
from tactics2d.participant.element import Vehicle

from .config import InterSimConfig
from .planner import ArcPath, lane_chain_points, match_lane, straight_path

# Adapted from InterSim (github.com/Tsinghua-MARS-Lab/InterSim), MIT,
# Copyright (c) 2022 Tsinghua MARS Lab.


@dataclass
class AgentRecord:
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


def build_scene_records(
    config: InterSimConfig,
    participants: Dict[object, object],
    frame: int,
    agent_ids: Optional[Iterable[object]] = None,
) -> Dict[object, AgentRecord]:
    """Collect the requested vehicles plus every agent near them.

    Args:
        config (InterSimConfig): The model configuration.
        participants (Dict): All participants in the scenario.
        frame (int): The current frame number. The unit is millisecond (ms).
        agent_ids (Iterable, optional): The agents to plan; all vehicles when
            None. Defaults to None.

    Returns:
        The scene records keyed by agent id.
    """

    requested_ids = select_vehicle_ids(participants, agent_ids)
    records: Dict[object, AgentRecord] = {}
    for agent_id in requested_ids:
        record = make_record(config, participants.get(agent_id), frame)
        if record is not None:
            records[agent_id] = record

    requested_positions = [(record.x, record.y) for record in records.values()]
    for participant_id, participant in participants.items():
        if participant_id in records:
            continue
        state = current_state(participant, frame)
        if state is None:
            continue
        if not any(
            np.hypot(state.x - x, state.y - y) <= config.interaction_distance
            for x, y in requested_positions
        ):
            continue
        record = make_record(config, participant, frame, state=state)
        if record is not None:
            records[participant_id] = record
    return records


def select_vehicle_ids(participants, agent_ids) -> List[object]:
    """Return the requested ids that are vehicles (all of them when None)."""

    if agent_ids is None:
        return [
            agent_id
            for agent_id, participant in participants.items()
            if isinstance(participant, Vehicle)
        ]
    selected = []
    for agent_id in agent_ids:
        participant = participants.get(agent_id)
        if participant is not None and isinstance(participant, Vehicle):
            selected.append(agent_id)
    return selected


def make_record(
    config: InterSimConfig, participant, frame: int, state=None
) -> Optional[AgentRecord]:
    """Build one agent record from its participant at ``frame``."""

    if participant is None:
        return None
    state = state if state is not None else current_state(participant, frame)
    if state is None:
        return None
    is_vehicle = isinstance(participant, Vehicle)
    length = participant.length
    width = participant.width
    if length is None or length <= 0.0:
        length = config.default_vehicle_length if is_vehicle else 0.6
    if width is None or width <= 0.0:
        width = config.default_vehicle_width if is_vehicle else 0.5
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
    return AgentRecord(
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
            config.vehicle_decel_limit
            or getattr(participant, "max_decel", 0.0)
            or 10.0
        ),
    )


def current_state(participant, frame: int):
    """Return the participant's latest state at or before ``frame``."""

    trajectory = participant.trajectory
    observed = [frame_ms for frame_ms in trajectory.frames if frame_ms <= frame]
    if not observed:
        return None
    return trajectory.get_state(observed[-1])


def reference_path(
    config: InterSimConfig, record: AgentRecord, map_: Optional[Map]
) -> Tuple[ArcPath, float]:
    """Build the lane-following path (or a straight fallback) for an agent.

    Returns:
        A tuple ``(path, s0)`` of the reference path and the agent's arc-length
        offset on it.
    """

    lookahead = (
        record.v0 * config.horizon_steps * config.dt + record.length + config.stop_margin + 10.0
    )
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
