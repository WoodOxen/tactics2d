# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared helpers for behavior-model rolling simulations."""

from copy import copy
from typing import Dict, Iterable, Optional, Sequence

from tactics2d.geometry import normalize_angle
from tactics2d.participant.element import Vehicle
from tactics2d.participant.trajectory import State, Trajectory


def copy_state(state: State, frame: Optional[int] = None) -> State:
    """Copy a trajectory state, optionally assigning a new frame."""

    velocity = state.velocity or (0.0, 0.0)
    return State(
        frame=state.frame if frame is None else frame,
        x=float(state.x),
        y=float(state.y),
        heading=normalize_angle(state.heading),
        vx=float(velocity[0]),
        vy=float(velocity[1]),
    )


def clone_vehicle_participants_at_frames(
    participants: Dict[object, object],
    frames: Sequence[int],
    agent_ids: Optional[Iterable[object]] = None,
    required_frame: Optional[int] = None,
    fps: Optional[float] = None,
) -> Dict[object, object]:
    """Clone vehicles and keep only the requested trajectory frames.

    Args:
        participants: Source participants keyed by agent id.
        frames: Frames to copy into each cloned trajectory, in chronological order.
        agent_ids: Optional subset of participants to clone.
        required_frame: If set, participants missing this frame are skipped.
        fps: Optional frequency assigned to cloned trajectories.

    Returns:
        A dictionary of shallow-copied vehicles with fresh trajectories.
    """

    selected_ids = list(participants.keys()) if agent_ids is None else list(agent_ids)
    clones = {}
    for agent_id in selected_ids:
        participant = participants.get(agent_id)
        if not isinstance(participant, Vehicle):
            continue
        if required_frame is not None and not participant.trajectory.has_state(required_frame):
            continue

        copied_states = [
            copy_state(participant.trajectory.get_state(state_frame), state_frame)
            for state_frame in frames
            if participant.trajectory.has_state(state_frame)
        ]
        if not copied_states:
            continue

        clone = copy(participant)
        clone.trajectory = Trajectory(
            id_=participant.trajectory.id_,
            fps=fps if fps is not None else participant.trajectory.fps,
            stable_freq=True,
        )
        for state in copied_states:
            clone.trajectory.add_state(state)
        clones[agent_id] = clone

    return clones
