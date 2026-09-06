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

from tactics2d.geometry import spatial
from tactics2d.map.element import Map
from tactics2d.map.query._lane_semantics import is_vehicle_lane
from tactics2d.participant.element import Vehicle


DEFAULT_ENDPOINT_TOLERANCE = 2.0
DEFAULT_SUCCESSOR_HEADING_TOLERANCE_DEG = 45.0
DEFAULT_NEIGHBOR_HEADING_TOLERANCE_DEG = 30.0
MAX_ROUTE_CANDIDATES = 8
LANE_MATCH_BOUNDARY_SEARCH_MARGIN = 2.0
LANE_MATCH_BOUNDARY_TOLERANCE = 1.0


def _centerline_array(lane):
    centerline = lane.centerline()
    if centerline is None or len(centerline.coords) < 2:
        return None
    return np.asarray(centerline.coords, dtype=float)


def _heading_difference(first: float, second: float) -> float:
    return abs(spatial.normalize_angle(second - first))


def _lane_end_heading(points: np.ndarray) -> float:
    return float(np.arctan2(points[-1, 1] - points[-2, 1], points[-1, 0] - points[-2, 0]))


def _lane_start_heading(points: np.ndarray) -> float:
    return float(np.arctan2(points[1, 1] - points[0, 1], points[1, 0] - points[0, 0]))


def _route_connection_is_valid(
    first_id, second_id, map_: Map, max_gap: float = 4.0, max_heading_error_deg: float = 45.0
) -> bool:
    """Validate a recorded lane transition without mutating map topology.

    A lanelet exit may begin in the interior of the incoming lane.  Projecting
    the outgoing start onto that centerline keeps valid recorded turns usable
    while rejecting unsupported nearest-lane jumps.
    """

    first = map_.lanes.get(first_id)
    second = map_.lanes.get(second_id)
    if (
        first is None
        or second is None
        or not is_vehicle_lane(first)
        or not is_vehicle_lane(second)
    ):
        return False
    first_centerline = first.centerline()
    second_centerline = second.centerline()
    if first_centerline is None or second_centerline is None:
        return False
    first_points = np.asarray(first_centerline.coords, dtype=float)
    second_points = np.asarray(second_centerline.coords, dtype=float)
    if len(first_points) < 2 or len(second_points) < 2:
        return False
    first_line = LineString(first_points)
    second_start = Point(second_points[0])
    join_progress = float(first_line.project(second_start))
    join_point = first_line.interpolate(join_progress)
    gap = float(join_point.distance(second_start))

    tangent_before = first_line.interpolate(max(0.0, join_progress - 0.5))
    tangent_after = first_line.interpolate(min(first_line.length, join_progress + 0.5))
    first_heading = np.arctan2(
        tangent_after.y - tangent_before.y, tangent_after.x - tangent_before.x
    )
    second_heading = np.arctan2(
        second_points[1, 1] - second_points[0, 1],
        second_points[1, 0] - second_points[0, 0],
    )
    heading_error = abs(spatial.normalize_angle(second_heading - first_heading))
    if gap <= 0.5:
        return heading_error <= np.deg2rad(max_heading_error_deg)

    connector_heading = np.arctan2(
        second_points[0, 1] - join_point.y, second_points[0, 0] - join_point.x
    )
    connector_error = max(
        abs(spatial.normalize_angle(connector_heading - first_heading)),
        abs(spatial.normalize_angle(second_heading - connector_heading)),
    )
    return (
        gap <= max_gap
        and heading_error <= np.deg2rad(max_heading_error_deg)
        and connector_error <= np.deg2rad(max_heading_error_deg)
    )


def infer_lane_topology(
    map_: Map,
    endpoint_tolerance: float = DEFAULT_ENDPOINT_TOLERANCE,
    successor_heading_tolerance_deg: float = DEFAULT_SUCCESSOR_HEADING_TOLERANCE_DEG,
    neighbor_heading_tolerance_deg: float = DEFAULT_NEIGHBOR_HEADING_TOLERANCE_DEG,
) -> Dict[str, int]:
    """Infer missing lane relationships from Lanelet geometry and shared borders.

    Lanelet2 files used by the LevelX datasets contain lane geometry and shared
    boundary ids, but the generic OSM parser does not populate successor or
    neighbor sets.  LimSim needs those relationships for route-constrained
    matching and legal lane changes.  Inference is only applied when the map
    has no relationships at all; explicit topology from richer maps is left
    untouched.

    Successors are inferred when a lane end is close to another lane start and
    their directions agree.  Same-direction lanes sharing a boundary roadline
    become left/right neighbors.  The geometric thresholds intentionally reject
    the multi-metre jumps produced by an unconstrained nearest-lane sequence.
    """

    if map_ is None or not map_.lanes:
        return {"successors": 0, "neighbors": 0}
    marker = "_tactics2d_lane_topology_inferred"
    if getattr(map_, marker, False):
        return {"successors": 0, "neighbors": 0}
    if any(
        lane.predecessors
        or lane.successors
        or lane.left_neighbors
        or lane.right_neighbors
        for lane in map_.lanes.values()
    ):
        setattr(map_, marker, True)
        return {"successors": 0, "neighbors": 0}

    lane_data = {}
    for lane_id, lane in map_.lanes.items():
        if not is_vehicle_lane(lane):
            continue
        points = _centerline_array(lane)
        if points is None:
            continue
        lane_data[lane_id] = {
            "points": points,
            "start_heading": _lane_start_heading(points),
            "end_heading": _lane_end_heading(points),
        }

    successor_count = 0
    neighbor_count = 0
    successor_heading_tolerance = np.deg2rad(successor_heading_tolerance_deg)
    neighbor_heading_tolerance = np.deg2rad(neighbor_heading_tolerance_deg)

    left_boundary_lanes = {}
    right_boundary_lanes = {}
    for lane_id in lane_data:
        line_ids = map_.lanes[lane_id].line_ids or {}
        for line_id in line_ids.get("left", ()):
            left_boundary_lanes.setdefault(line_id, []).append(lane_id)
        for line_id in line_ids.get("right", ()):
            right_boundary_lanes.setdefault(line_id, []).append(lane_id)

    for first_id, first in lane_data.items():
        first_lane = map_.lanes[first_id]
        first_lines = first_lane.line_ids or {}
        for line_id in first_lines.get("left", ()):
            for second_id in right_boundary_lanes.get(line_id, ()):
                if second_id == first_id:
                    continue
                second = lane_data[second_id]
                if (
                    _heading_difference(first["start_heading"], second["start_heading"])
                    <= neighbor_heading_tolerance
                    and second_id not in first_lane.left_neighbors
                ):
                    first_lane.left_neighbors.add(second_id)
                    map_.lanes[second_id].right_neighbors.add(first_id)
                    neighbor_count += 1
        for line_id in first_lines.get("right", ()):
            for second_id in left_boundary_lanes.get(line_id, ()):
                if second_id == first_id:
                    continue
                second = lane_data[second_id]
                if (
                    _heading_difference(first["start_heading"], second["start_heading"])
                    <= neighbor_heading_tolerance
                    and second_id not in first_lane.right_neighbors
                ):
                    first_lane.right_neighbors.add(second_id)
                    map_.lanes[second_id].left_neighbors.add(first_id)
                    neighbor_count += 1

    bucket_size = max(float(endpoint_tolerance), 1e-6)
    start_buckets = {}
    for lane_id, lane in lane_data.items():
        start = lane["points"][0]
        bucket = tuple(np.floor(start / bucket_size).astype(int))
        start_buckets.setdefault(bucket, []).append(lane_id)

    for first_id, first in lane_data.items():
        first_lane = map_.lanes[first_id]
        end = first["points"][-1]
        end_bucket = np.floor(end / bucket_size).astype(int)
        for x_offset in (-1, 0, 1):
            for y_offset in (-1, 0, 1):
                bucket = (int(end_bucket[0] + x_offset), int(end_bucket[1] + y_offset))
                for second_id in start_buckets.get(bucket, ()):
                    if first_id == second_id:
                        continue
                    second = lane_data[second_id]
                    gap = float(np.linalg.norm(end - second["points"][0]))
                    if (
                        gap <= endpoint_tolerance
                        and _heading_difference(
                            first["end_heading"], second["start_heading"]
                        )
                        <= successor_heading_tolerance
                        and second_id not in first_lane.successors
                    ):
                        first_lane.successors.add(second_id)
                        map_.lanes[second_id].predecessors.add(first_id)
                        successor_count += 1

    setattr(map_, marker, True)
    return {"successors": successor_count, "neighbors": neighbor_count}


def _lane_heading_at(lane, s: Optional[float]) -> Optional[float]:
    """Compute the heading of *lane* at longitudinal offset *s*."""
    centerline = lane.centerline()
    centerline = np.asarray(centerline.coords, dtype=float) if centerline is not None else None
    if centerline is None or len(centerline) < 2:
        return None
    line = LineString(centerline)
    if line.length <= 1e-6:
        return None
    route_s = 0.0 if s is None else float(np.clip(s, 0.0, line.length))
    before = line.interpolate(max(0.0, route_s - 1.0))
    after = line.interpolate(min(line.length, route_s + 1.0))
    return float(np.arctan2(after.y - before.y, after.x - before.x))


def _rank_lane_candidates(
    map_: Map,
    x: float,
    y: float,
    heading: float,
    lane_match_radius: float,
    heading_weight: float,
    max_candidates: Optional[int] = None,
    lane_ids: Optional[Iterable[object]] = None,
):
    """Return nearby lanes sorted by geometric and heading consistency."""
    if map_ is None or not map_.lanes:
        return []

    point = Point(x, y)
    candidate_ids = list(map_.lanes if lane_ids is None else lane_ids)
    ranked = []
    for lane_id in candidate_ids:
        lane = map_.lanes.get(lane_id)
        if lane is None or not is_vehicle_lane(lane) or lane.geometry is None:
            continue
        centerline = lane.centerline()
        centerline = (
            np.asarray(centerline.coords, dtype=float)
            if centerline is not None
            else None
        )
        line = LineString(centerline) if centerline is not None else None
        projection = lane.project_point((x, y))
        distance = line.distance(point) if line is not None else lane.geometry.distance(point)
        if projection is not None:
            distance = projection.distance
        boundary_match = (
            distance <= lane_match_radius + LANE_MATCH_BOUNDARY_SEARCH_MARGIN
            and lane.geometry.distance(point) <= LANE_MATCH_BOUNDARY_TOLERANCE
        )
        if distance > lane_match_radius and not boundary_match:
            continue
        lane_heading = _lane_heading_at(
            lane, projection.s if projection is not None else None
        )
        heading_error = (
            _heading_difference(heading, lane_heading)
            if lane_heading is not None
            else 0.0
        )
        ranked.append((distance + heading_weight * heading_error, lane_id))

    ranked.sort(key=lambda item: (item[0], str(item[1])))
    return ranked if max_candidates is None else ranked[:max_candidates]


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

    ranked = _rank_lane_candidates(
        map_,
        x,
        y,
        heading,
        lane_match_radius=lane_match_radius,
        heading_weight=heading_weight,
        max_candidates=1,
    )
    return ranked[0][1] if ranked else None


def _transition_penalty(
    map_: Map,
    first_id,
    second_id,
    endpoint_tolerance: float = DEFAULT_ENDPOINT_TOLERANCE,
) -> float:
    """Score a lane transition, strongly penalizing disconnected jumps."""
    if first_id == second_id:
        return 0.0
    first = map_.lanes.get(first_id)
    second = map_.lanes.get(second_id)
    if first is None or second is None:
        return 30.0
    if second_id in first.successors:
        return 0.5
    if second_id in first.left_neighbors or second_id in first.right_neighbors:
        return 2.0
    if _route_connection_is_valid(first_id, second_id, map_):
        return 4.0
    first_points = _centerline_array(first)
    second_points = _centerline_array(second)
    if first_points is not None and second_points is not None:
        gap = float(np.linalg.norm(first_points[-1] - second_points[0]))
        if gap <= endpoint_tolerance:
            heading_error = _heading_difference(
                _lane_end_heading(first_points), _lane_start_heading(second_points)
            )
            if heading_error <= np.deg2rad(DEFAULT_SUCCESSOR_HEADING_TOLERANCE_DEG):
                return 4.0
        else:
            gap = min(gap, 20.0)
    else:
        gap = 20.0
    return 30.0 + gap


def extract_lane_sequence(
    participant: Vehicle,
    map_: Map,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    lane_match_radius: float = 4.0,
    heading_weight: float = 2.0,
) -> List[str]:
    """Extract the ordered lane sequence a vehicle follows.

    Ranks nearby lanes for each frame, then decodes the complete sequence with
    topology-aware transition costs.  Consecutive duplicates are collapsed so
    the returned list records only lane transitions.

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

    if not frames:
        return []
    infer_lane_topology(map_)

    frame_candidates = []
    for f in frames:
        state = participant.trajectory.get_state(f)
        if state is None:
            continue
        candidates = _rank_lane_candidates(
            map_,
            state.x,
            state.y,
            state.heading,
            lane_match_radius=lane_match_radius,
            heading_weight=heading_weight,
            max_candidates=MAX_ROUTE_CANDIDATES,
        )
        if candidates:
            frame_candidates.append(candidates)

    if not frame_candidates:
        return []

    costs = [score for score, _ in frame_candidates[0]]
    backpointers = []
    for candidates in frame_candidates[1:]:
        next_costs = []
        previous = []
        frame_index = len(backpointers) + 1
        for emission, lane_id in candidates:
            transition_costs = [
                costs[index] + _transition_penalty(map_, previous_lane, lane_id)
                for index, (_, previous_lane) in enumerate(
                    frame_candidates[frame_index - 1]
                )
            ]
            best_index = int(np.argmin(transition_costs))
            next_costs.append(emission + transition_costs[best_index])
            previous.append(best_index)
        backpointers.append(previous)
        costs = next_costs

    lane_index = int(np.argmin(costs))
    decoded_indices = [lane_index]
    for previous in reversed(backpointers):
        lane_index = previous[lane_index]
        decoded_indices.append(lane_index)
    decoded_indices.reverse()
    decoded = [
        frame_candidates[frame_index][candidate_index][1]
        for frame_index, candidate_index in enumerate(decoded_indices)
    ]

    seq = []
    for lane_id in decoded:
        if not seq or lane_id != seq[-1]:
            seq.append(lane_id)

    return seq


def extract_all_lane_sequences(
    participants: Dict[object, object],
    map_: Map,
    reference_frame: int,
    agent_ids: Optional[Iterable[object]] = None,
    **kwargs,
) -> Dict[object, Tuple[str, ...]]:
    """Extract lane sequences for the selected vehicles.

    For each requested vehicle, the trajectory suffix starting at
    *reference_frame* is decoded into an ordered lane sequence.  Starting the
    decode at the snapshot is important for routes that revisit a lane id.

    Args:
        participants: Traffic participants keyed by ID.
        map_: Tactics2D Map.
        reference_frame: The frame at which the route should be snapshotted.
        agent_ids: Optional subset of participant IDs to process.  Use this
            when only a few vehicles need lane sequences (e.g. ego-only).

    Returns:
        ``{agent_id: (lane_id_0, lane_id_1, ...)}``.
    """
    infer_lane_topology(map_)
    route_map = {}
    ids_to_process = agent_ids if agent_ids is not None else participants.keys()
    for pid in ids_to_process:
        p = participants.get(pid)
        if not isinstance(p, Vehicle):
            continue
        # only process vehicles that are active at takeover
        if reference_frame not in p.trajectory.history_states:
            continue
        # Decode from the snapshot onward.  Decoding only the suffix avoids
        # choosing the first occurrence when a lane id appears more than once
        # in a loop or roundabout route.
        route_kwargs = dict(kwargs)
        requested_start = route_kwargs.pop("start_frame", None)
        requested_end = route_kwargs.pop("end_frame", None)
        route_start = reference_frame if requested_start is None else max(
            reference_frame, requested_start
        )
        trimmed = extract_lane_sequence(
            p,
            map_,
            start_frame=route_start,
            end_frame=requested_end,
            **route_kwargs,
        )
        if not trimmed:
            continue
        route_map[pid] = tuple(trimmed)
    return route_map
