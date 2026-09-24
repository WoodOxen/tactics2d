# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Prepare and convert trajectories at behavior-model boundaries."""

from copy import copy
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from tactics2d.geometry import spatial
from tactics2d.participant.trajectory.state import State
from tactics2d.participant.trajectory.trajectory import Trajectory


def first_observed_frame(participants: Mapping[object, object]) -> int:
    """Return the earliest timestamp present in a participant collection."""

    frames = [
        participant.trajectory.first_frame
        for participant in participants.values()
        if participant.trajectory.first_frame is not None
    ]
    if not frames:
        raise ValueError("the scenario has no observed frames to anchor the rollout on.")
    return int(min(frames))


def observed_step_ms(participants: Mapping[object, object]) -> float:
    """Return the median observed trajectory interval in milliseconds."""

    gaps = []
    for participant in participants.values():
        frames = sorted(participant.trajectory.frames)
        gaps.extend(frames[index + 1] - frames[index] for index in range(len(frames) - 1))
    return float(np.median(gaps)) if gaps else float("nan")


def resample_trajectory(
    trajectory: Trajectory, step_ms: int, origin_ms: Optional[int] = None
) -> Trajectory:
    """Interpolate a trajectory onto a fixed timestamp lattice."""

    if step_ms <= 0:
        raise ValueError("step_ms must be positive.")
    frames = sorted(trajectory.frames)
    if len(frames) < 2:
        return trajectory
    origin = frames[0] if origin_ms is None else int(origin_ms)
    start = origin + int(np.ceil((frames[0] - origin) / step_ms)) * step_ms
    grid = list(range(start, frames[-1] + 1, step_ms))
    if len(grid) < 2:
        return trajectory

    states = [trajectory.get_state(frame) for frame in frames]
    times = np.asarray(frames, dtype=float)
    xs = np.interp(grid, times, [state.x for state in states])
    ys = np.interp(grid, times, [state.y for state in states])
    headings = np.interp(grid, times, np.unwrap([state.heading for state in states]))
    result = Trajectory(id_=trajectory.id_, fps=round(1000.0 / step_ms, 3), stable_freq=True)
    for index, frame in enumerate(grid):
        heading = float(spatial.normalize_angle(headings[index]))
        ahead = min(index + 1, len(grid) - 1)
        distance = float(np.hypot(xs[ahead] - xs[index], ys[ahead] - ys[index]))
        speed = distance / (step_ms / 1000.0) if ahead > index else 0.0
        result.add_state(
            State(
                frame=int(frame),
                x=float(xs[index]),
                y=float(ys[index]),
                heading=heading,
                vx=speed * np.cos(heading),
                vy=speed * np.sin(heading),
            )
        )
    return result


def resample_participants(
    participants: Dict[object, object],
    step_ms: int,
    origin_ms: Optional[int] = None,
    *,
    tolerance: float = 0.1,
) -> Dict[object, object]:
    """Return shallow participant copies whose trajectories use one time lattice.

    Participant subclasses and custom attributes are preserved because only
    the participant object and its trajectory are copied.
    """

    observed = observed_step_ms(participants)
    if not np.isfinite(observed) or abs(observed - step_ms) <= tolerance * step_ms:
        return participants
    origin = first_observed_frame(participants) if origin_ms is None else int(origin_ms)
    result = {}
    for agent_id, participant in participants.items():
        trajectory = resample_trajectory(participant.trajectory, step_ms, origin)
        if trajectory is participant.trajectory:
            result[agent_id] = participant
            continue
        cloned = copy(participant)
        cloned.trajectory = trajectory
        result[agent_id] = cloned
    return result


def trajectory_from_poses(
    agent_id: object,
    positions: np.ndarray,
    headings: np.ndarray,
    frames: Sequence[int],
    *,
    availabilities: Optional[Iterable[bool]] = None,
    speeds: Optional[Sequence[float]] = None,
    step_ms: Optional[int] = None,
) -> Trajectory:
    """Build a trajectory from aligned world-frame positions and headings."""

    count = len(frames)
    if len(positions) != count or len(headings) != count:
        raise ValueError("positions, headings, and frames must have equal lengths.")
    available = np.ones(count, dtype=bool) if availabilities is None else np.asarray(availabilities)
    if len(available) != count:
        raise ValueError("availabilities must match frames.")
    if speeds is not None and len(speeds) != count:
        raise ValueError("speeds must match frames.")
    if step_ms is None and count > 1:
        step_ms = int(round(float(np.median(np.diff(frames)))))
    fps = None if not step_ms else round(1000.0 / step_ms, 3)
    result = Trajectory(id_=agent_id, fps=fps, stable_freq=True)
    previous = None
    previous_frame = None
    for index, frame in enumerate(frames):
        if not bool(available[index]):
            continue
        heading = float(headings[index])
        if speeds is not None:
            speed = float(speeds[index])
        elif previous is not None and previous_frame is not None:
            elapsed = (int(frame) - previous_frame) / 1000.0
            speed = (
                float(np.linalg.norm(positions[index] - previous)) / elapsed if elapsed > 0 else 0.0
            )
        else:
            speed = 0.0
        result.add_state(
            State(
                frame=int(frame),
                x=float(positions[index][0]),
                y=float(positions[index][1]),
                heading=heading,
                vx=speed * np.cos(heading),
                vy=speed * np.sin(heading),
            )
        )
        previous = positions[index]
        previous_frame = int(frame)
    return result
