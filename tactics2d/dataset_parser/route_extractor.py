# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Extract lane-level route sequences from vehicle trajectories.

This module provides dataset-agnostic utilities that project a vehicle's
recorded trajectory onto the map's lane graph, producing an ordered lane
sequence that can be used as navigation guidance for behavior models such
as LimSim.
"""

from typing import Dict, Iterable, List, Optional, Tuple

from tactics2d.map.element import Map
from tactics2d.participant.element import Vehicle


def match_lane_for_state(
    map_: Map,
    x: float,
    y: float,
    heading: float,
    lane_match_radius: float = 4.0,
    heading_weight: float = 2.0,
) -> Optional[str]:
    """Find the lane ID that best matches a single (x, y, heading) state.

    Delegates distance and heading scoring to :meth:`Map.match_lane`.

    Args:
        map_: Tactics2D Map with lanes.
        x, y: Position in meters.
        heading: Yaw angle in radians.
        lane_match_radius: Maximum lateral distance to a lane candidate (m).
        heading_weight: Weight of heading error relative to distance.

    Returns:
        Matching lane ID, or ``None`` if no lane is within range.
    """
    if map_ is None:
        return None
    return map_.match_lane(x, y, heading, radius=lane_match_radius, heading_weight=heading_weight)


def extract_lane_sequence(
    participant: Vehicle,
    map_: Map,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    lane_match_radius: float = 4.0,
    heading_weight: float = 2.0,
    min_dwell_frames: int = 5,
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
        min_dwell_frames: Frames a lane has to be held for it to count as part
            of the route. Defaults to 5, i.e. half a second at 10 Hz.

    Returns:
        Ordered list of lane ID strings (no consecutive duplicates).
    """
    frames = sorted(participant.trajectory.history_states.keys())
    if start_frame is not None:
        frames = [f for f in frames if f >= start_frame]
    if end_frame is not None:
        frames = [f for f in frames if f <= end_frame]

    runs: List[Tuple[str, int]] = []
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
        if lid is None:
            continue
        if runs and runs[-1][0] == lid:
            runs[-1][1] += 1
        else:
            runs.append([lid, 1])

    seq = []
    for lid, dwell in runs:
        if dwell < min_dwell_frames:
            continue
        if not seq or seq[-1] != lid:
            seq.append(lid)
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
