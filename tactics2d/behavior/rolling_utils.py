# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared closed-loop collision and progress primitives."""

from typing import Dict, Optional, Tuple

import numpy as np

from tactics2d.geometry import spatial
from tactics2d.participant.trajectory import State, Trajectory

# Collision classification thresholds by relative heading.
REAR_TOL = 30.0 * np.pi / 180.0
SIDE_TOL = 150.0 * np.pi / 180.0

# Bodies are shrunken by this much before the overlap test.
COLLISION_MARGIN = 0.7

# Progress is summed over this frame window, dropping implausible steps.
PROGRESS_START = 12
PROGRESS_END = 80
MAX_PROGRESS_STEP = 20.0


def first_observed_frame(participants: Dict[object, object]) -> int:
    """Return the earliest frame any participant is observed at.

    Raises:
        ValueError: If no participant carries a trajectory.
    """

    first_frames = [
        participant.trajectory.first_frame
        for participant in participants.values()
        if participant.trajectory.first_frame is not None
    ]
    if not first_frames:
        raise ValueError("the scenario has no observed frames to anchor the rollout on.")
    return int(min(first_frames))


def classify_collision(ego_yaw: float, other_yaw: float) -> int:
    """Classify a contact by the relative heading of the two bodies.

    Args:
        ego_yaw (float): Heading of the agent the collision is judged from, in rad.
        other_yaw (float): Heading of the other body, in rad.

    Returns:
        ``2`` for rear, ``1`` for side, ``0`` for front.
    """

    diff = abs(spatial.normalize_angle(ego_yaw - other_yaw))
    if diff < REAR_TOL:
        return 2
    if diff > SIDE_TOL:
        return 1
    return 0


def bodies_overlap(
    pose_a: np.ndarray,
    dims_a: Tuple[float, float],
    pose_b: np.ndarray,
    dims_b: Tuple[float, float],
    shrink: str = "linear",
) -> bool:
    """Test two posed bodies for overlap under a model's shrink convention.

    - ``"factor"``: half-extents are ``0.5 * extent * COLLISION_MARGIN``.
    - ``"linear"``: ``COLLISION_MARGIN`` is subtracted from each extent, floored at ``0.1`` m.

    Args:
        pose_a (np.ndarray): ``(4,)`` pose of x, y, z and yaw.
        dims_a (Tuple[float, float]): Length and width in m.
        pose_b (np.ndarray): The other ``(4,)`` pose.
        dims_b (Tuple[float, float]): The other length and width in m.
        shrink (str, optional): ``"linear"`` or ``"factor"``. Defaults to
            "linear".

    Returns:
        True when the two shrunken bodies overlap.
    """

    if shrink == "factor":
        box_a = (float(pose_a[0]), float(pose_a[1]), float(pose_a[3]), dims_a[0], dims_a[1])
        box_b = (float(pose_b[0]), float(pose_b[1]), float(pose_b[3]), dims_b[0], dims_b[1])
        return spatial.boxes_overlap(box_a, box_b, COLLISION_MARGIN)

    body_a = spatial.oriented_box(
        float(pose_a[0]),
        float(pose_a[1]),
        float(pose_a[3]),
        dims_a[0] - COLLISION_MARGIN,
        dims_a[1] - COLLISION_MARGIN,
    )
    body_b = spatial.oriented_box(
        float(pose_b[0]),
        float(pose_b[1]),
        float(pose_b[3]),
        dims_b[0] - COLLISION_MARGIN,
        dims_b[1] - COLLISION_MARGIN,
    )
    return bool(body_a.intersects(body_b))


def collision_kind(
    poses: Dict[object, np.ndarray],
    dims: Dict[object, Tuple[float, float]],
    ego_id: object,
    index: int,
    shrink: str = "linear",
) -> Optional[int]:
    """Classify a same-frame ego collision at one index, if any.

    Agents are scanned in ``poses`` insertion order, so when the ego overlaps
    several bodies the earliest-inserted one decides the class.

    Args:
        poses (Dict[object, np.ndarray]): Per-agent ``(S, 4)`` pose arrays.
        dims (Dict[object, Tuple[float, float]]): Per-agent length and width in m.
        ego_id (object): The agent the collision is judged from.
        index (int): Step to test.
        shrink (str, optional): Shrink convention passed to
            :func:`bodies_overlap`. Defaults to "linear".

    Returns:
        ``2`` for rear, ``1`` for side, ``0`` for front, or ``None`` when the
        ego is clear at ``index``.
    """

    pose_ego = poses[ego_id][index]
    if pose_ego[0] == -1:
        return None
    for other_id in poses:
        if other_id == ego_id:
            continue
        pose_other = poses[other_id][index]
        if pose_other[0] == -1:
            continue
        if not bodies_overlap(pose_ego, dims[ego_id], pose_other, dims[other_id], shrink):
            continue
        return classify_collision(float(pose_ego[3]), float(pose_other[3]))
    return None


def progress_window(
    poses: Dict[object, np.ndarray],
    agent_id: object,
    end_index: int,
    steps: int,
    start: int = PROGRESS_START,
    stop: int = PROGRESS_END,
    max_step: float = MAX_PROGRESS_STEP,
) -> Tuple[float, int]:
    """Sum one agent's displacement over the progress window.

    The scan stops at the first invalid pose and at the horizon end; a step
    longer than *max_step* is dropped as implausible without ending the scan.

    Args:
        poses (Dict[object, np.ndarray]): Per-agent ``(S, 4)`` pose arrays.
        agent_id (object): The agent to sum over.
        end_index (int): Last simulated step.
        steps (int): Number of steps laid out in each pose array.
        start (int, optional): First window index. Defaults to 12.
        stop (int, optional): One past the last window index. Defaults to 80.
        max_step (float, optional): Largest plausible per-step displacement in
            m. Defaults to 20.0.

    Returns:
        A tuple of:
            - progress (float): The summed displacement in meters.
            - counted (int): How many steps were summed.
    """

    array = poses[agent_id]
    total = 0.0
    counted = 0
    for index in range(start, stop):
        if index >= end_index:
            break
        if index + 1 >= steps:
            break
        pose_i = array[index]
        pose_j = array[index + 1]
        if pose_i[0] == -1 or pose_j[0] == -1:
            break
        distance = float(np.hypot(pose_i[0] - pose_j[0], pose_i[1] - pose_j[1]))
        if distance >= max_step:
            continue
        total += distance
        counted += 1
    return total, counted


# A scenario whose frames are further apart than this fraction of a model's step
# is treated as recorded at another rate and resampled onto the lattice.
RATE_TOLERANCE = 0.1


def observed_step_ms(participants: Dict[object, object]) -> float:
    """Return the median gap between the frames a scenario is observed at.

    Args:
        participants (Dict[object, object]): Participants keyed by agent id.

    Returns:
        The gap in milliseconds, or ``nan`` when no participant has two frames.
    """

    gaps = []
    for participant in participants.values():
        frames = sorted(participant.trajectory.frames)
        gaps.extend(frames[i + 1] - frames[i] for i in range(len(frames) - 1))
    return float(np.median(gaps)) if gaps else float("nan")


def to_lattice(participants: Dict[object, object], step_ms: int, frame_ms0: Optional[int] = None):
    """Lay a scenario's recorded motion onto a fixed ``step_ms`` lattice.

    Positions and headings are interpolated onto the lattice and the speed is
    re-derived from the resampled course; participants already on it are unchanged.

    Args:
        participants (Dict[object, object]): Participants to lay out, keyed by agent id.
        step_ms (int): Lattice interval in milliseconds.
        frame_ms0 (Optional[int], optional): Timestamp of lattice index 0.
            Defaults to None, which uses the scenario's own first frame.

    Returns:
        A participant dict on the lattice, or *participants* itself when it is
        already there.
    """

    observed = observed_step_ms(participants)
    if not np.isfinite(observed) or abs(observed - step_ms) <= RATE_TOLERANCE * step_ms:
        return participants

    first_frames = [
        participant.trajectory.first_frame
        for participant in participants.values()
        if participant.trajectory.first_frame is not None
    ]
    if not first_frames:
        return participants
    if frame_ms0 is None:
        frame_ms0 = int(min(first_frames))

    laid_out = {}
    for agent_id, participant in participants.items():
        trajectory = participant.trajectory
        frames = sorted(trajectory.frames)
        if len(frames) < 2:
            laid_out[agent_id] = participant
            continue

        times = np.asarray(frames, dtype=float)
        states = [trajectory.get_state(frame) for frame in frames]
        positions_x = np.asarray([state.x for state in states])
        positions_y = np.asarray([state.y for state in states])
        # Headings are unwrapped first, so a wrap through +/-pi interpolates as
        # the small turn it is rather than as a full revolution.
        headings = np.unwrap(np.asarray([state.heading for state in states]))

        start = frame_ms0 + int(np.ceil((frames[0] - frame_ms0) / step_ms)) * step_ms
        grid = list(range(start, frames[-1] + 1, step_ms))
        if len(grid) < 2:
            laid_out[agent_id] = participant
            continue
        grid_x = np.interp(grid, times, positions_x)
        grid_y = np.interp(grid, times, positions_y)
        grid_heading = np.interp(grid, times, headings)

        resampled = Trajectory(id_=agent_id, fps=round(1000.0 / step_ms, 3), stable_freq=True)
        for index, frame in enumerate(grid):
            heading = float(spatial.normalize_angle(grid_heading[index]))
            ahead = min(index + 1, len(grid) - 1)
            arc = float(np.hypot(grid_x[ahead] - grid_x[index], grid_y[ahead] - grid_y[index]))
            speed = arc / (step_ms / 1000.0) if ahead > index else 0.0
            resampled.add_state(
                State(
                    frame=frame,
                    x=float(grid_x[index]),
                    y=float(grid_y[index]),
                    heading=heading,
                    vx=speed * np.cos(heading),
                    vy=speed * np.sin(heading),
                )
            )
        laid_out[agent_id] = type(participant)(
            agent_id,
            participant.type_,
            trajectory=resampled,
            length=getattr(participant, "length", None),
            width=getattr(participant, "width", None),
        )
    return laid_out
