# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Directed relation detection over planned agent trajectories."""

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple

import numpy as np

from tactics2d.geometry import spatial

# Adapted from InterSim (github.com/Tsinghua-MARS-Lab/InterSim), MIT,
# Copyright (c) 2022 Tsinghua MARS Lab.


class Edge(NamedTuple):
    """Directed relation between one influencer and one reactor.

    The influencer passes the conflict point first; the reactor must yield.
    ``frame_diff`` is the signed reactor-minus-influencer frame offset.
    """

    influencer: object
    reactor: object
    frame_diff: int


@dataclass(frozen=True)
class AgentBody:
    """Oriented vehicle footprint used for collision checks."""

    x: float
    y: float
    yaw: float
    length: float
    width: float


def check_body_collision(a: AgentBody, b: AgentBody, margin: float = 0.7) -> bool:
    """Check whether two oriented boxes overlap, shrinking both by ``margin``.

    The test is symmetric in ``a`` and ``b``.
    """

    # This is the same four-axis separating-axis test as
    # ``spatial.boxes_overlap``.  InterSim calls it hundreds of thousands of
    # times per dense scene, so scalar arithmetic avoids allocating several
    # tiny NumPy arrays for every candidate pair.
    cos_a, sin_a = math.cos(a.yaw), math.sin(a.yaw)
    cos_b, sin_b = math.cos(b.yaw), math.sin(b.yaw)
    forward_a = (cos_a, sin_a)
    lateral_a = (-sin_a, cos_a)
    forward_b = (cos_b, sin_b)
    lateral_b = (-sin_b, cos_b)
    half_length_a = 0.5 * a.length * margin
    half_width_a = 0.5 * a.width * margin
    half_length_b = 0.5 * b.length * margin
    half_width_b = 0.5 * b.width * margin
    dx, dy = b.x - a.x, b.y - a.y

    for axis_x, axis_y in (forward_a, lateral_a, forward_b, lateral_b):
        projection_a = half_length_a * abs(
            forward_a[0] * axis_x + forward_a[1] * axis_y
        ) + half_width_a * abs(lateral_a[0] * axis_x + lateral_a[1] * axis_y)
        projection_b = half_length_b * abs(
            forward_b[0] * axis_x + forward_b[1] * axis_y
        ) + half_width_b * abs(lateral_b[0] * axis_x + lateral_b[1] * axis_y)
        if abs(axis_x * dx + axis_y * dy) > projection_a + projection_b:
            return False
    return True


def _rotate_point(x, y, origin_x, origin_y, angle):
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    relative_x = x - origin_x
    relative_y = y - origin_y
    return (
        relative_x * cos_angle - relative_y * sin_angle + origin_x,
        relative_x * sin_angle + relative_y * cos_angle + origin_y,
    )


def check_released_overlap(
    checking: AgentBody, target: AgentBody, yaw_sign: float, margin: float = 0.7
) -> bool:
    """Match the released simulator's asymmetric seven-point overlap test."""

    dx = target.x - checking.x
    dy = target.y - checking.y
    if abs(dx) > checking.length + target.length:
        return False
    if abs(dy) > checking.length + target.length:
        return False
    if math.hypot(dx, dy) <= (checking.width + target.width) / 2.0:
        return True

    half_width = target.width * margin / 2.0
    half_length = target.length * margin / 2.0
    target_points = [
        (dx - half_width, dy - half_length),
        (dx - half_width, dy),
        (dx - half_width, dy + half_length),
        (dx + half_width, dy + half_length),
        (dx + half_width, dy),
        (dx + half_width, dy - half_length),
    ]
    target_angle = yaw_sign * target.yaw + math.pi / 2.0
    target_points = [_rotate_point(x, y, dx, dy, target_angle) for x, y in target_points]
    target_points.insert(0, (dx, dy))
    checking_angle = yaw_sign * checking.yaw + math.pi / 2.0
    target_points = [_rotate_point(x, y, 0.0, 0.0, checking_angle) for x, y in target_points]
    return any(
        abs(x) < checking.width * margin / 2.0 and abs(y) < checking.length * margin / 2.0
        for x, y in target_points
    )


def detect_relation_edges(
    poses: Dict[object, np.ndarray],
    shapes: Dict[object, Tuple[float, float]],
    is_vehicle: Dict[object, bool],
    pair_pool: Optional[Iterable[Tuple[object, object]]] = None,
    margin: float = 0.7,
    same_direction_tolerance_deg: float = 30.0,
    max_gap: Optional[int] = None,
) -> List[Edge]:
    """Detect directed relations over planned trajectories.

    The later arrival is the reactor; a mixed pair makes the non-vehicle the
    influencer, and a rear-end catch-up makes the trailing vehicle the reactor.

    Args:
        poses: Agent id to planned pose array with shape ``(S, 4)`` storing
            ``[x, y, z, yaw]`` per step. Invalid slots use ``x == -1``.
        shapes: Agent id to ``(length, width)`` in meters.
        is_vehicle: Whether each agent is a motorized vehicle.
        pair_pool: Optional candidate ``(id_a, id_b)`` pairs. Defaults to all
            unordered pairs over the agents present in ``poses``.
        margin: Box shrink factor applied by the collision check.
        same_direction_tolerance_deg: Heading tolerance used to tell rear-end
            contacts from crossing contacts.
        max_gap: Optional frame window bounding the pair scan.

    Returns:
        A list of directed :class:`Edge` records.
    """

    if pair_pool is None:
        agent_ids = list(poses)
        pair_pool = [
            (agent_ids[i], agent_ids[j])
            for i in range(len(agent_ids))
            for j in range(i + 1, len(agent_ids))
        ]

    edges: List[Edge] = []
    tolerance_rad = same_direction_tolerance_deg * np.pi / 180.0
    for id_a, id_b in pair_pool:
        if id_a not in poses or id_b not in poses:
            continue
        vehicle_a = bool(is_vehicle.get(id_a, True))
        vehicle_b = bool(is_vehicle.get(id_b, True))
        if not vehicle_a and not vehicle_b:
            continue

        length_a, width_a = shapes[id_a]
        length_b, width_b = shapes[id_b]
        poses_a = poses[id_a]
        poses_b = poses[id_b]
        if _nearest_same_index_gap(poses_a, poses_b) > (length_a + length_b) / 2.0 + 2.0:
            continue
        pair = _collision_pairs(
            poses_a, poses_b, length_a, width_a, length_b, width_b, margin, max_gap=max_gap
        )
        if pair is None:
            continue
        idx_a, idx_b = pair
        yaw_a = float(poses[id_a][idx_a, 3])
        yaw_b = float(poses[id_b][idx_b, 3])

        if vehicle_a and vehicle_b:
            frame_diff = idx_a - idx_b
            if frame_diff > 0:
                edges.append(Edge(id_b, id_a, frame_diff))
            elif frame_diff < 0:
                edges.append(Edge(id_a, id_b, frame_diff))
            elif abs(spatial.normalize_angle(yaw_a - yaw_b)) < tolerance_rad:
                # Rear-end contact: the trailing vehicle is the reactor.
                heading = spatial.heading_unit(yaw_a)
                centre_a = poses[id_a][idx_a, :2]
                centre_b = poses[id_b][idx_b, :2]
                if float(np.dot(centre_b - centre_a, heading)) > 0:
                    edges.append(Edge(id_b, id_a, 0))
                else:
                    edges.append(Edge(id_a, id_b, 0))
            else:
                # Simultaneous crossing: the earlier corridor entrant passes first.
                entry_a, entry_b = _corridor_entry(
                    poses[id_a], poses[id_b], (length_a, width_a), (length_b, width_b), margin
                )
                if entry_a is not None and entry_b is not None and entry_a != entry_b:
                    if entry_a < entry_b:
                        edges.append(Edge(id_a, id_b, 0))
                    else:
                        edges.append(Edge(id_b, id_a, 0))
                else:
                    edges.append(Edge(id_b, id_a, 0))
                    edges.append(Edge(id_a, id_b, 0))
        elif vehicle_a:
            edges.append(Edge(id_b, id_a, idx_a - idx_b))
        else:
            edges.append(Edge(id_a, id_b, idx_b - idx_a))

    return edges


def _collision_pairs(
    poses_a: np.ndarray,
    poses_b: np.ndarray,
    length_a: float,
    width_a: float,
    length_b: float,
    width_b: float,
    margin: float,
    max_gap: Optional[int] = None,
) -> Optional[Tuple[int, int]]:
    """Return the closest-in-time colliding frame pair of two trajectories."""

    # An oriented rectangle is wholly contained in the circle centred on it
    # with half-diagonal radius.  Rejecting pairs whose circles do not meet is
    # therefore an exact broad phase: it avoids the much more expensive SAT
    # test without changing any collision result.  Keep the historical
    # ``margin <= 0`` edge case on the SAT path.
    circle_limit_sq = None
    if margin > 0.0:
        radius_a = 0.5 * margin * np.hypot(length_a, width_a)
        radius_b = 0.5 * margin * np.hypot(length_b, width_b)
        circle_limit_sq = float((radius_a + radius_b) ** 2)
    largest_gap = max(len(poses_a), len(poses_b)) - 1
    if max_gap is not None:
        largest_gap = min(largest_gap, max_gap)
    # Search by increasing time difference.  The first hit is therefore the
    # same minimum-gap pair the previous exhaustive scan retained.
    for diff in range(largest_gap + 1):
        pairs = []
        for idx_a in range(len(poses_a)):
            if diff:
                pairs.append((idx_a, idx_a - diff))
            pairs.append((idx_a, idx_a + diff))
        for idx_a, idx_b in pairs:
            if idx_b < 0 or idx_b >= len(poses_b):
                continue
            pose_a = poses_a[idx_a]
            if pose_a[0] == -1:
                continue
            body_a = AgentBody(
                float(pose_a[0]), float(pose_a[1]), float(pose_a[3]), length_a, width_a
            )
            pose_b = poses_b[idx_b]
            if pose_b[0] == -1:
                continue
            if circle_limit_sq is not None:
                dx = float(pose_b[0] - pose_a[0])
                dy = float(pose_b[1] - pose_a[1])
                if dx * dx + dy * dy > circle_limit_sq:
                    continue
            body_b = AgentBody(
                float(pose_b[0]), float(pose_b[1]), float(pose_b[3]), length_b, width_b
            )
            if not check_body_collision(body_a, body_b, margin):
                continue
            return idx_a, idx_b
    return None


def _corridor_entry(
    poses_a: np.ndarray,
    poses_b: np.ndarray,
    shape_a: Tuple[float, float],
    shape_b: Tuple[float, float],
    margin: float,
) -> Tuple[Optional[int], Optional[int]]:
    """Return the first step each trajectory enters the other's corridor.

    The corridor half-width is the summed lateral half-extent of the two bodies,
    shrunken by ``margin``.
    """

    half = 0.5 * (shape_a[1] + shape_b[1]) * margin + 1e-6
    distance_sq = np.sum((poses_a[:, None, :2] - poses_b[None, :, :2]) ** 2, axis=-1)
    hits_a = np.flatnonzero(np.any(distance_sq <= half * half, axis=1))
    hits_b = np.flatnonzero(np.any(distance_sq <= half * half, axis=0))
    entry_a = int(hits_a[0]) if hits_a.size else None
    entry_b = int(hits_b[0]) if hits_b.size else None
    return entry_a, entry_b


def _nearest_same_index_gap(poses_a: np.ndarray, poses_b: np.ndarray) -> float:
    """Return the minimum centre distance between two trajectories at equal time."""

    common = min(len(poses_a), len(poses_b))
    valid = (poses_a[:common, 0] != -1) & (poses_b[:common, 0] != -1)
    if not np.any(valid):
        return float("inf")
    delta = poses_a[:common, :2][valid] - poses_b[:common, :2][valid]
    return float(np.min(np.hypot(delta[:, 0], delta[:, 1])))
