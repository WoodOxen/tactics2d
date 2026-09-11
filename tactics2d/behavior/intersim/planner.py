# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Lane-following rollout and deceleration profiles."""

import math
from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import LineString

from tactics2d.geometry import spatial
from tactics2d.map.element import Map

# Adapted from InterSim (github.com/Tsinghua-MARS-Lab/InterSim), MIT,
# Copyright (c) 2022 Tsinghua MARS Lab.

# Comfortable / emergency deceleration used by InterSim's speed adjustment.
_A_SLOWDOWN = 2.0
_A_EMERGENCY = 4.5
# Corner speed limiting mirrors upstream `proper_speed_minimal = max(5, pi/3 /
# yaw_change)`: the path is pre-scanned for the sharpest upcoming heading change
# and the cruise profile brakes so the corner is taken at a safe speed.
_MIN_TURN_SPEED = 5.0
_TURN_YAW_K = math.pi / 3.0
_TURN_YAW_MIN = 0.04
_TURN_YAW_MAX = math.pi / 2.0 * 0.9


class ArcPath:
    """Arc-length re-sampler over a 2-D polyline."""

    def __init__(self, points: np.ndarray):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        keep = np.ones(len(points), dtype=bool)
        if len(points) > 1:
            seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
            keep[1:] = seg > 1e-9
        points = points[keep]
        if len(points) < 2:
            raise ValueError("ArcPath needs at least two distinct points.")
        self.points = points
        seg_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        self.cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        self.total_length = float(self.cumulative[-1])

    def _clamped_arcs(self, arcs: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(arcs, dtype=float), 0.0, self.total_length)

    def sample_poses(self, arcs: np.ndarray) -> np.ndarray:
        """Return pose rows ``[x, y, 0, yaw]`` at the given arc lengths."""

        arcs = self._clamped_arcs(arcs)
        xs = np.interp(arcs, self.cumulative, self.points[:, 0])
        ys = np.interp(arcs, self.cumulative, self.points[:, 1])
        tangents = self.points[1:] - self.points[:-1]
        segment_idx = np.clip(
            np.searchsorted(self.cumulative, arcs, side="right") - 1,
            0,
            len(tangents) - 1,
        )
        tangents = tangents[segment_idx]
        yaws = np.arctan2(tangents[:, 1], tangents[:, 0])
        poses = np.zeros((len(arcs), 4), dtype=float)
        poses[:, 0] = xs
        poses[:, 1] = ys
        poses[:, 3] = yaws
        return poses


def lane_chain_points(
    map_: Map,
    lane_id: object,
    lookahead: float,
    goal: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Concatenate a lane centerline with its successors along the map.

    Successors are followed until the chain covers ``lookahead`` meters or no
    successor remains. When ``goal`` is provided the successor whose initial
    heading best aims at the goal is chosen (destination routing); otherwise
    successors are followed greedily by heading continuity.

    Returns:
        A ``(M, 2)`` polyline oriented in the driving direction, or ``None``
        when the starting lane has no usable centerline.
    """

    seen = {lane_id}
    segments = []
    length = 0.0
    current = lane_id
    while current in map_.lanes:
        lane = map_.lanes[current]
        centerline = lane.centerline()
        if centerline is None or len(centerline.coords) < 2:
            break
        pts = np.asarray(centerline.coords, dtype=float)
        if segments:
            seg_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            length += float(np.sum(seg_lengths))
        segments.append(pts)
        if length >= lookahead:
            break
        successors = sorted(
            (
                next_id
                for next_id in lane.successors
                if next_id not in seen and next_id in map_.lanes
            ),
            key=str,
        )
        if not successors:
            break
        current = _best_continuation(segments[-1][-1], successors, map_, goal)
        seen.add(current)
    if not segments:
        return None
    chain = np.concatenate(segments, axis=0)
    chain = _deduplicate(chain)
    if len(chain) < 2:
        return None
    return chain


def _best_continuation(
    tail: np.ndarray,
    candidates: List[object],
    map_: Map,
    goal: Optional[np.ndarray] = None,
) -> object:
    """Pick the successor lane heading toward ``goal`` when given."""

    best_id = candidates[0]
    best_score = None
    goal_direction = None
    if goal is not None:
        delta = np.asarray(goal, dtype=float) - tail
        norm = np.linalg.norm(delta)
        if norm > 1e-9:
            goal_direction = delta / norm
    for candidate_id in candidates:
        lane = map_.lanes[candidate_id]
        centerline = lane.centerline()
        if centerline is None or len(centerline.coords) < 2:
            continue
        head = np.asarray(centerline.coords, dtype=float)[0]
        direction = head - tail
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            return candidate_id
        direction = direction / norm
        if goal_direction is not None:
            score = float(direction @ goal_direction)
        else:
            score = float(direction @ _end_direction(centerline))
        if best_score is None or score > best_score:
            best_score = score
            best_id = candidate_id
    return best_id


def _end_direction(centerline: LineString) -> np.ndarray:
    pts = np.asarray(centerline.coords, dtype=float)
    tangent = pts[-1] - pts[-2]
    norm = np.linalg.norm(tangent)
    if norm < 1e-9:
        return np.array([1.0, 0.0])
    return tangent / norm


def _deduplicate(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return points
    keep = np.ones(len(points), dtype=bool)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep[1:] = seg > 1e-9
    return points[keep]


def match_lane(
    map_: Optional[Map],
    x: float,
    y: float,
    heading: float,
    radius: float,
    heading_tolerance_deg: float,
    lookahead: float,
) -> Optional[Tuple[object, float]]:
    """Match a pose to a map lane.

    The closest lane whose centerline passes within ``radius`` of ``(x, y)``
    and whose local tangent agrees with ``heading`` within the given tolerance
    is selected.

    Returns:
        A tuple ``(lane_id, s0)`` where ``s0`` is the arc-length offset of the
        pose projected onto the lane centerline. ``None`` when no lane matches
        (the caller should fall back to constant velocity).
    """

    if map_ is None:
        return None
    tolerance_rad = heading_tolerance_deg * np.pi / 180.0
    best = None
    best_distance = radius
    for lane_id, lane in map_.lanes.items():
        centerline = lane.centerline()
        if centerline is None or len(centerline.coords) < 2:
            continue
        line = np.asarray(centerline.coords, dtype=float)
        s0, nearest = _nearest_along(line, np.array([x, y]))
        distance = float(np.hypot(nearest[0] - x, nearest[1] - y))
        if distance > best_distance:
            continue
        tangent = _tangent_at(line, s0)
        yaw_error = abs(spatial.normalize_angle(heading - np.arctan2(tangent[1], tangent[0])))
        if yaw_error > tolerance_rad:
            continue
        best_distance = distance
        best = (lane_id, s0, line)
    if best is None:
        return None
    lane_id, s0, _ = best
    return lane_id, s0


def _nearest_along(line: np.ndarray, point: np.ndarray) -> Tuple[float, np.ndarray]:
    segments = line[1:] - line[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(lengths)])
    to_point = point - line[:-1]
    dots = np.sum(to_point * segments, axis=1) / np.maximum(lengths * lengths, 1e-12)
    dots = np.clip(dots, 0.0, 1.0)
    projections = line[:-1] + dots[:, None] * segments
    distances = np.linalg.norm(projections - point, axis=1)
    index = int(np.argmin(distances))
    s = cum[index] + dots[index] * lengths[index]
    return float(s), projections[index]


def _tangent_at(line: np.ndarray, s: float) -> np.ndarray:
    segments = line[1:] - line[:-1]
    lengths = np.linalg.norm(segments, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(lengths)])
    index = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(segments) - 1))
    tangent = segments[index]
    norm = np.linalg.norm(tangent)
    if norm < 1e-9:
        return np.array([1.0, 0.0])
    return tangent / norm


def polyline_nearest_s(points: np.ndarray, point: np.ndarray) -> Tuple[float, np.ndarray]:
    """Project a point onto a polyline and return the arc offset."""

    line = _deduplicate(np.asarray(points, dtype=float).reshape(-1, 2))
    if len(line) < 2:
        return 0.0, np.asarray(point, dtype=float)
    s0, nearest = _nearest_along(line, np.asarray(point, dtype=float))
    return s0, nearest


def straight_path(x: float, y: float, heading: float, distance: float) -> ArcPath:
    """Build a straight reference path for the constant-velocity fallback."""

    direction = np.array([np.cos(heading), np.sin(heading)])
    points = np.stack(
        [np.array([x, y]) + direction * offset for offset in np.linspace(0.0, distance, 20)]
    )
    return ArcPath(points)


def baseline_speeds(v0: float, steps: int) -> np.ndarray:
    """Return a constant-velocity per-step speed profile."""

    return np.full(steps + 1, max(0.0, v0), dtype=float)


def deceleration_speeds(
    v0: float,
    stop_distance: float,
    dt: float,
    steps: int,
    a_slow: float = _A_SLOWDOWN,
    a_emergency: float = _A_EMERGENCY,
) -> np.ndarray:
    """Return a speed profile that stops before ``stop_distance`` meters.

    The profile cruises at ``v0`` until braking at ``a_slow`` brings the agent
    to rest exactly at ``stop_distance`` when the distance allows; otherwise
    braking starts immediately at the strongest feasible deceleration
    (capped at ``a_emergency``) and the speed holds at zero afterwards.
    """

    steps = int(steps)
    speeds = [max(0.0, v0)]
    if v0 <= 0.0 or stop_distance <= 0.0:
        for _ in range(steps):
            speeds.append(0.0)
        return np.asarray(speeds, dtype=float)

    distance_at_slow = v0 * v0 / (2.0 * a_slow)
    if stop_distance >= distance_at_slow:
        deceleration = a_slow
        cruise_distance = stop_distance - distance_at_slow
        cruise_steps = int(np.floor(cruise_distance / (v0 * dt)))
        cruise_steps = min(cruise_steps, steps)
        speed = v0
        speeds = []
        for index in range(steps):
            if index < cruise_steps:
                speeds.append(speed)
            else:
                speed = max(0.0, speed - deceleration * dt)
                speeds.append(speed)
        speeds.append(0.0)
    else:
        deceleration = min(a_emergency, v0 * v0 / (2.0 * stop_distance))
        speed = v0
        for _ in range(steps):
            speed = max(0.0, speed - deceleration * dt)
            speeds.append(speed)
    return np.asarray(speeds, dtype=float)


def yield_speeds(
    v0: float,
    end_v: float,
    distance: float,
    dt: float,
    steps: int,
    a_slow: float = _A_SLOWDOWN,
    a_emergency: float = _A_EMERGENCY,
) -> np.ndarray:
    """Return a speed profile that slows to ``end_v`` and then continues.

    Mirrors InterSim's ``adjust_speed_for_collision``: a yielding agent brakes
    toward ``end_v`` (rather than to a full stop) and keeps moving, so it stays
    off the road as a stationary target. When a near stop is unavoidable
    (``end_v`` near zero or the conflict distance too short), it falls back to
    the full-stop profile.
    """

    if end_v <= 0.0 or distance <= 0.0:
        return deceleration_speeds(v0, max(0.0, distance), dt, steps, a_slow, a_emergency)
    end_v = min(end_v, v0)
    steps = int(steps)
    needed = (v0 * v0 - end_v * end_v) / (2.0 * max(distance, 1e-6))
    deceleration = min(a_emergency, max(a_slow, needed))
    speeds = [v0]
    speed = v0
    for _ in range(steps):
        if speed > end_v:
            speed = max(end_v, speed - deceleration * dt)
        speeds.append(speed)
    return np.asarray(speeds, dtype=float)


def arcs_from_speeds(speeds: np.ndarray, dt: float) -> np.ndarray:
    """Return cumulative arc lengths travelled up to each step."""

    speeds = np.asarray(speeds, dtype=float)
    increments = speeds[:-1] * dt
    return np.concatenate([[0.0], np.cumsum(increments)])


def cruise_with_corner_caps(
    points: np.ndarray,
    s0: float,
    v0: float,
    target: float,
    accel: float,
    max_target: float,
    dt: float,
    horizon: int,
    a_slow: float = _A_SLOWDOWN,
) -> np.ndarray:
    """Return a cruise profile that brakes for upcoming turns.

    Mirrors upstream ``proper_speed_minimal = max(5, pi/3 / yaw_change)``: each
    path vertex whose tangent swings sharply gets a speed cap, and the speed is
    limited so the agent can brake down to that cap before reaching it. Straights
    keep the plain accelerate-toward-``target`` profile.
    """

    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) < 3:
        return _plain_cruise(v0, target, accel, max_target, dt, horizon)

    tangents = points[1:] - points[:-1]
    headings = np.arctan2(tangents[:, 1], tangents[:, 0])
    seg_lengths = np.linalg.norm(tangents, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])

    caps = np.full(len(headings), np.inf, dtype=float)
    for seg in range(1, len(headings)):
        yaw_change = abs(spatial.normalize_angle(headings[seg] - headings[seg - 1]))
        if _TURN_YAW_MIN < yaw_change < _TURN_YAW_MAX:
            caps[seg] = max(_MIN_TURN_SPEED, _TURN_YAW_K / yaw_change)

    speed = max(0.0, v0)
    travelled = s0
    profile = [speed]
    for _ in range(int(horizon)):
        limit = np.inf
        for seg in range(1, len(headings)):
            vertex_arc = cumulative[seg]
            if vertex_arc <= travelled:
                continue
            corner_cap = caps[seg]
            if not np.isfinite(corner_cap):
                continue
            bound = math.sqrt(corner_cap * corner_cap + 2.0 * a_slow * (vertex_arc - travelled))
            if bound < limit:
                limit = bound
        natural = speed
        if speed < target:
            natural = min(target, speed + accel * dt)
        natural = min(natural, max_target)
        speed = min(natural, limit)
        speed = max(0.0, speed)
        travelled += speed * dt
        profile.append(speed)
    return np.asarray(profile, dtype=float)


def _plain_cruise(v0, target, accel, max_target, dt, horizon) -> np.ndarray:
    """Accelerate toward ``target`` (capped at ``max_target``) from ``v0``."""

    speeds = [v0]
    speed = v0
    for _ in range(int(horizon)):
        if speed < target:
            speed = min(target, speed + accel * dt)
        speed = min(speed, max_target)
        speeds.append(speed)
    return np.asarray(speeds, dtype=float)


def poses_from_path(
    path: ArcPath,
    arcs: np.ndarray,
    fallback_x: float,
    fallback_y: float,
    fallback_yaw: float,
) -> np.ndarray:
    """Sample poses ``[x, y, 0, yaw]`` at arc lengths along a path."""

    return path.sample_poses(arcs)
