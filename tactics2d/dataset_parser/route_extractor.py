# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Extract lane-level route sequences from vehicle trajectories.

This module provides dataset-agnostic utilities that project a vehicle's
recorded trajectory onto the map's lane graph, producing an ordered lane
sequence that can be used as navigation guidance for behavior models such
as LimSim.
"""

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from shapely.geometry import LineString, Point

from tactics2d.geometry import normalize_angle
from tactics2d.map.element import Map
from tactics2d.participant.element import Vehicle
from tactics2d.routing.utils import find_nearest_lane, get_lane_centerline


def _lane_heading_at(lane, s: Optional[float]) -> Optional[float]:
    """Compute the heading of *lane* at longitudinal offset *s*."""
    centerline = get_lane_centerline(lane)
    if centerline is None or len(centerline) < 2:
        return None
    line = LineString(centerline)
    if line.length <= 1e-6:
        return None
    route_s = 0.0 if s is None else float(np.clip(s, 0.0, line.length))
    before = line.interpolate(max(0.0, route_s - 1.0))
    after = line.interpolate(min(line.length, route_s + 1.0))
    return float(np.arctan2(after.y - before.y, after.x - before.x))


def _candidate_lane_ids(map_: Map, lane_id: str, point_xy, lane_match_radius: float):
    """Collect candidate lane IDs near *point_xy*."""
    lane = map_.lanes.get(lane_id)
    point = Point(point_xy)
    lane_ids = set()
    for candidate_id, candidate_lane in map_.lanes.items():
        if candidate_lane.geometry is None:
            continue
        cl = get_lane_centerline(candidate_lane)
        distance = (
            LineString(cl).distance(point)
            if cl is not None
            else candidate_lane.geometry.distance(point)
        )
        if distance <= lane_match_radius:
            lane_ids.add(candidate_id)
    if lane is None:
        return list(lane_ids or {lane_id})
    lane_ids.update(
        {lane_id, *lane.left_neighbors, *lane.right_neighbors, *lane.predecessors, *lane.successors}
    )
    return list(lane_ids)


def match_lane_for_state(
    map_: Map,
    x: float,
    y: float,
    heading: float,
    lane_match_radius: float = 4.0,
    heading_weight: float = 2.0,
) -> Optional[str]:
    """Find the lane ID that best matches a single (x, y, heading) state.

    Uses the same distance + heading-consistency scoring as
    :class:`~tactics2d.behavior.limsim.scene.SceneBuilder`.

    Args:
        map_: Tactics2D Map with lanes.
        x, y: Position in meters.
        heading: Yaw angle in radians.
        lane_match_radius: Maximum lateral distance to a lane candidate (m).
        heading_weight: Weight of heading error relative to distance.

    Returns:
        Matching lane ID, or ``None`` if no lane is within range.
    """
    if map_ is None or len(map_.lanes) == 0:
        return None

    nearby_lane_id = find_nearest_lane(map_, (x, y))
    if nearby_lane_id is None:
        return None

    point = Point(x, y)
    best_lane_id = nearby_lane_id
    best_score = np.inf
    lane_ids = _candidate_lane_ids(map_, nearby_lane_id, (x, y), lane_match_radius)

    for lid in lane_ids:
        lane = map_.lanes.get(lid)
        if lane is None:
            continue
        proj = lane.project_point((x, y))
        distance = lane.geometry.distance(point) if proj is None else proj.distance
        if distance > lane_match_radius:
            continue
        lh = _lane_heading_at(lane, proj.s if proj is not None else None)
        heading_error = 0.0
        if lh is not None:
            heading_error = abs(normalize_angle(heading - lh))
            heading_error = min(heading_error, abs(np.pi - heading_error))
        score = distance + heading_weight * heading_error
        if score < best_score:
            best_score = score
            best_lane_id = lid

    if best_lane_id is None:
        return None

    # final distance check on the best match
    best_lane = map_.lanes[best_lane_id]
    proj = best_lane.project_point((x, y))
    distance = best_lane.geometry.distance(point) if proj is None else proj.distance
    if distance > lane_match_radius:
        return None
    return best_lane_id


def extract_lane_sequence(
    participant: Vehicle,
    map_: Map,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    lane_match_radius: float = 4.0,
    heading_weight: float = 2.0,
) -> List[str]:
    """Extract the ordered lane sequence a vehicle follows.

    Projects each frame in the vehicle's trajectory onto the map's lane
    graph using :func:`match_lane_for_state`.  Consecutive duplicates are
    collapsed so the returned list records only lane *transitions*.

    Args:
        participant: A :class:`Vehicle` whose trajectory holds history states.
        map_: Tactics2D Map with lanes.
        start_frame: First frame timestamp to consider (inclusive).
            Defaults to the earliest frame in the trajectory.
        end_frame: Last frame timestamp to consider (inclusive).
            Defaults to the latest frame in the trajectory.

    Returns:
        Ordered list of lane ID strings (no consecutive duplicates).
    """
    frames = sorted(participant.trajectory.history_states.keys())
    if start_frame is not None:
        frames = [f for f in frames if f >= start_frame]
    if end_frame is not None:
        frames = [f for f in frames if f <= end_frame]

    seq = []
    prev_lane = None
    for f in frames:
        state = participant.trajectory.get_state(f)
        if state is None:
            continue
        lid = match_lane_for_state(
            map_,
            state.x,
            state.y,
            state.heading,
            lane_match_radius=lane_match_radius,
            heading_weight=heading_weight,
        )
        if lid is not None and lid != prev_lane:
            seq.append(lid)
            prev_lane = lid

    return seq


def extract_all_lane_sequences(
    participants: Dict[object, object],
    map_: Map,
    reference_frame: int,
    agent_ids: Optional[Iterable[object]] = None,
    **kwargs,
) -> Dict[object, Tuple[str, ...]]:
    """Extract lane sequences for the selected vehicles.

    For each requested vehicle, the full trajectory is scanned to build a
    complete lane sequence.  The sequence is trimmed to start from the
    vehicle's lane at *reference_frame*.

    Args:
        participants: Traffic participants keyed by ID.
        map_: Tactics2D Map.
        reference_frame: The frame at which the route should be snapshotted.
        agent_ids: Optional subset of participant IDs to process.  Use this
            when only a few vehicles need lane sequences (e.g. ego-only).

    Returns:
        ``{agent_id: (lane_id_0, lane_id_1, ...)}``.
    """
    route_map = {}
    ids_to_process = agent_ids if agent_ids is not None else participants.keys()
    for pid in ids_to_process:
        p = participants.get(pid)
        if not isinstance(p, Vehicle):
            continue
        # only process vehicles that are active at takeover
        if reference_frame not in p.trajectory.history_states:
            continue
        frames = sorted(p.trajectory.history_states.keys())
        if not frames:
            continue

        # 1. extract full lane sequence from the entire trajectory
        full_seq = extract_lane_sequence(p, map_, **kwargs)
        if not full_seq:
            continue

        # 2. find the vehicle's lane at reference_frame
        state = p.trajectory.get_state(reference_frame)
        takeover_lane = match_lane_for_state(map_, state.x, state.y, state.heading, **kwargs)

        # 3. trim: keep suffix starting from takeover_lane
        if takeover_lane is not None and takeover_lane in full_seq:
            idx = full_seq.index(takeover_lane)
            trimmed = full_seq[idx:]
        else:
            # if we can't find the takeover lane, use the full sequence
            trimmed = full_seq

        if trimmed:
            route_map[pid] = tuple(trimmed)
    return route_map
