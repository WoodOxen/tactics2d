# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared closed-loop collision and progress primitives."""

from typing import Dict, Optional, Tuple

import numpy as np

from tactics2d.geometry import spatial

# Collision classification thresholds by relative heading.
REAR_TOL = 30.0 * np.pi / 180.0
SIDE_TOL = 150.0 * np.pi / 180.0

# Bodies are shrunken by this much before the overlap test.
COLLISION_MARGIN = 0.7

# Progress is summed over this frame window, dropping implausible steps.
PROGRESS_START = 12
PROGRESS_END = 80
MAX_PROGRESS_STEP = 20.0


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
