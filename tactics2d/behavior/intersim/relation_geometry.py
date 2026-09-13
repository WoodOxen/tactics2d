# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Directed relation detection over planned agent trajectories."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple

import numpy as np

# Adapted from InterSim (github.com/Tsinghua-MARS-Lab/InterSim), MIT,
# Copyright (c) 2022 Tsinghua MARS Lab.


class Edge(NamedTuple):
    """Directed relation between one influencer and one reactor.

    The influencer passes the conflict point first; the reactor must yield.
    ``reactor_step`` is the frame index of the reactor pose involved in the
    closest-in-time conflict; ``frame_diff`` is the signed reactor-minus-
    influencer frame offset (positive means the reactor arrives later).
    """

    influencer: object
    reactor: object
    reactor_step: int
    frame_diff: int


@dataclass(frozen=True)
class AgentBody:
    """Axis-aligned vehicle footprint used for collision checks."""

    x: float
    y: float
    yaw: float
    length: float
    width: float


def _obb_axes(x, y, yaw, length, width, margin):
    """Return the four candidate separating axes of an oriented box."""
    forward = np.array([np.cos(yaw), np.sin(yaw)])
    lateral = np.array([-np.sin(yaw), np.cos(yaw)])
    half_length = 0.5 * length * margin
    half_width = 0.5 * width * margin
    return forward, lateral, half_length, half_width


def check_body_collision(a: AgentBody, b: AgentBody, margin: float = 0.7) -> bool:
    """Check whether two oriented boxes overlap.

    Both boxes are shrunk by ``margin`` before testing (mirroring the 0.7
    safety factor used by InterSim), so bodies that merely graze are treated
    as collision-free. The test is symmetric in ``a`` and ``b``.
    """

    f1, l1, hl1, hw1 = _obb_axes(a.x, a.y, a.yaw, a.length, a.width, margin)
    f2, l2, hl2, hw2 = _obb_axes(b.x, b.y, b.yaw, b.length, b.width, margin)
    delta = np.array([b.x - a.x, b.y - a.y])

    for axis in (f1, l1, f2, l2):
        half1 = hl1 * abs(float(np.dot(f1, axis))) + hw1 * abs(float(np.dot(l1, axis)))
        half2 = hl2 * abs(float(np.dot(f2, axis))) + hw2 * abs(float(np.dot(l2, axis)))
        if abs(float(np.dot(axis, delta))) > half1 + half2:
            return False
    return True


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
    """Return the closest-in-time colliding frame pair of two trajectories.

    When ``max_gap`` is given, only frame pairs within that time offset are
    considered, which bounds the scan cost on large scenes; the closest-in-time
    conflict is what relation resolution consumes anyway.
    """

    best_diff = None
    best_pair = None
    for idx_a in range(len(poses_a)):
        pose_a = poses_a[idx_a]
        if pose_a[0] == -1:
            continue
        body_a = AgentBody(
            float(pose_a[0]), float(pose_a[1]), float(pose_a[3]), length_a, width_a
        )
        idx_b_lo = max(0, idx_a - max_gap) if max_gap is not None else 0
        idx_b_hi = min(len(poses_b), idx_a + max_gap + 1) if max_gap is not None else len(poses_b)
        for idx_b in range(idx_b_lo, idx_b_hi):
            pose_b = poses_b[idx_b]
            if pose_b[0] == -1:
                continue
            body_b = AgentBody(
                float(pose_b[0]), float(pose_b[1]), float(pose_b[3]), length_b, width_b
            )
            if not check_body_collision(body_a, body_b, margin):
                continue
            diff = abs(idx_a - idx_b)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_pair = (idx_a, idx_b)
    if best_pair is None:
        return None
    return best_pair


def _nearest_same_index_gap(poses_a: np.ndarray, poses_b: np.ndarray) -> float:
    """Return the minimum centre distance between two trajectories at equal time."""

    common = min(len(poses_a), len(poses_b))
    best = None
    for idx in range(common):
        if poses_a[idx, 0] == -1 or poses_b[idx, 0] == -1:
            continue
        distance = float(np.hypot(poses_a[idx, 0] - poses_b[idx, 0], poses_a[idx, 1] - poses_b[idx, 1]))
        if best is None or distance < best:
            best = distance
    return float("inf") if best is None else best


def _same_direction(yaw_a: float, yaw_b: float, tolerance_rad: float) -> bool:
    diff = np.arctan2(np.sin(yaw_a - yaw_b), np.cos(yaw_a - yaw_b))
    return abs(diff) < tolerance_rad


def _corridor_entry(
    poses_a: np.ndarray,
    poses_b: np.ndarray,
    shape_a: Tuple[float, float],
    shape_b: Tuple[float, float],
    margin: float,
) -> Tuple[Optional[int], Optional[int]]:
    """Return the first step each trajectory enters the other's corridor.

    The corridor half-width is the summed lateral half-extent of the two
    bodies (shrunken by ``margin``); the entry step is when the agent's centre
    first comes within that distance of the other agent's reference polyline.
    """

    half = 0.5 * (shape_a[1] + shape_b[1]) * margin + 1e-6
    path_b = poses_b[:, :2]
    path_a = poses_a[:, :2]
    entry_a = None
    entry_b = None
    for idx in range(len(poses_a)):
        if float(np.min(np.linalg.norm(poses_a[idx, :2] - path_b, axis=1))) <= half:
            entry_a = idx
            break
    for idx in range(len(poses_b)):
        if float(np.min(np.linalg.norm(poses_b[idx, :2] - path_a, axis=1))) <= half:
            entry_b = idx
            break
    return entry_a, entry_b


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

    For every candidate pair sharing a spatio-temporal conflict, the agent
    reaching the conflict point later is the reactor and the earlier one the
    influencer. A mixed vehicle/non-vehicle pair always resolves to the
    non-vehicle being the influencer (a vehicle yields to it), mirroring the
    InterSim rule that non-motorized agents are never forced to yield. A
    same-frame, same-direction contact (a rear-end catch-up) resolves to the
    trailing vehicle being the reactor.

    Pairs whose same-index centres never come within reach of each other are
    skipped up front, so large scenes do not pay the full pair scan.

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
            poses_a,
            poses_b,
            length_a,
            width_a,
            length_b,
            width_b,
            margin,
            max_gap=max_gap,
        )
        if pair is None:
            continue
        idx_a, idx_b = pair
        yaw_a = float(poses[id_a][idx_a, 3])
        yaw_b = float(poses[id_b][idx_b, 3])

        if vehicle_a and vehicle_b:
            frame_diff = idx_a - idx_b
            if frame_diff > 0:
                edges.append(Edge(id_b, id_a, idx_a, frame_diff))
            elif frame_diff < 0:
                edges.append(Edge(id_a, id_b, idx_b, frame_diff))
            elif _same_direction(yaw_a, yaw_b, tolerance_rad):
                # Rear-end contact: the trailing vehicle is the reactor.
                heading = np.array([np.cos(yaw_a), np.sin(yaw_a)])
                centre_a = poses[id_a][idx_a, :2]
                centre_b = poses[id_b][idx_b, :2]
                if float(np.dot(centre_b - centre_a, heading)) > 0:
                    edges.append(Edge(id_b, id_a, idx_a, 0))
                else:
                    edges.append(Edge(id_a, id_b, idx_b, 0))
            else:
                # Simultaneous crossing: the agent entering the shared corridor
                # earlier passes first; the later one yields.
                entry_a, entry_b = _corridor_entry(
                    poses[id_a],
                    poses[id_b],
                    (length_a, width_a),
                    (length_b, width_b),
                    margin,
                )
                if entry_a is not None and entry_b is not None and entry_a != entry_b:
                    if entry_a < entry_b:
                        edges.append(Edge(id_a, id_b, idx_b, 0))
                    else:
                        edges.append(Edge(id_b, id_a, idx_a, 0))
                else:
                    edges.append(Edge(id_b, id_a, idx_a, 0))
                    edges.append(Edge(id_a, id_b, idx_b, 0))
        elif vehicle_a:
            edges.append(Edge(id_b, id_a, idx_a, idx_a - idx_b))
        else:
            edges.append(Edge(id_a, id_b, idx_b, idx_b - idx_a))

    return edges
