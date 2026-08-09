# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Lane-following trajectory rollout for LimSim-style actions."""

from collections import deque
from typing import FrozenSet, List, Optional, Set, Tuple

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import substring

from tactics2d.geometry import polyline, spatial
from tactics2d.map.element import Map
from tactics2d.map.query import SemanticMapQuery

from .action import LimSimAction
from .config import LimSimConfig
from .schema import AgentDecisionState

_AVAILABLE_LANE_LOOK_FORWARD = 100.0
_AVAILABLE_LANE_NO_CHANGE = 5.0
_AVAILABLE_LANE_MAX_SUCCESSOR_DEPTH = 8


def _route_connection_is_valid(
    first_id, second_id, map_: Map, max_gap: float = 4.0, max_heading_error_deg: float = 45.0
) -> bool:
    """Validate a recorded route-local transition without mutating topology.

    Lanelet exits can start from the interior of an incoming lane.  In that
    case comparing only the incoming endpoint with the outgoing start rejects
    a valid recorded turn and makes the planner choose a different successor.
    """

    first = map_.lanes.get(first_id)
    second = map_.lanes.get(second_id)
    if first is None or second is None:
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
        second_points[1, 1] - second_points[0, 1], second_points[1, 0] - second_points[0, 0]
    )
    heading_error = abs(spatial.normalize_angle(second_heading - first_heading))
    # At an interior lanelet exit the projected centerline and outgoing start
    # can differ slightly because the two lane polygons were sampled
    # independently.  The tangent agreement is the meaningful direction test
    # for this sub-lane-width gap.
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


def _lane_is_internal(lane_id, map_: Map) -> bool:
    lane = map_.lanes.get(lane_id)
    if lane is None:
        return False
    tags = lane.custom_tags or {}
    sumo_id = str(tags.get("sumo_id", ""))
    subtype = (lane.subtype or "").lower()
    return (
        bool(tags.get("is_internal"))
        or sumo_id.startswith(":")
        or bool(tags.get("is_intersection"))
        or bool(tags.get("junction"))
        or "intersection" in subtype
        or "junction" in subtype
    )


def _sumo_edge_id(lane) -> Optional[str]:
    tags = lane.custom_tags or {}
    if bool(tags.get("is_internal")):
        return None
    sumo_id = str(tags.get("sumo_id", ""))
    if not sumo_id or sumo_id.startswith(":") or "_" not in sumo_id:
        return None
    edge_id, lane_index = sumo_id.rsplit("_", 1)
    return edge_id if lane_index.isdigit() else None


def _lateral_lane_section(lane_id, map_: Map) -> FrozenSet[str]:
    """Return the parallel lanes that represent one road section."""

    lane = map_.lanes.get(lane_id)
    if lane is None:
        return frozenset()
    if _lane_is_internal(lane_id, map_):
        return frozenset({lane_id})

    edge_id = _sumo_edge_id(lane)
    if edge_id is not None:
        return frozenset(
            candidate_id
            for candidate_id, candidate in map_.lanes.items()
            if _sumo_edge_id(candidate) == edge_id
        )

    section = {lane_id}
    pending = [lane_id]
    while pending:
        current_id = pending.pop()
        current = map_.lanes.get(current_id)
        if current is None:
            continue
        for neighbor_id in current.left_neighbors | current.right_neighbors:
            if neighbor_id in map_.lanes and neighbor_id not in section:
                section.add(neighbor_id)
                pending.append(neighbor_id)
    return frozenset(section)


def _successor_paths_to_section(
    source_id, target_section: FrozenSet[str], map_: Map, route_lane_ids=(), max_depth: int = 3
):
    """Find local topology paths from one entry lane to a downstream section."""

    route_successors = {}
    for first_id, second_id in zip(route_lane_ids, route_lane_ids[1:]):
        if _route_connection_is_valid(first_id, second_id, map_):
            route_successors.setdefault(first_id, set()).add(second_id)

    paths = []
    queue = deque([(source_id, (source_id,))])
    while queue:
        lane_id, path = queue.popleft()
        if len(path) > 1 and lane_id in target_section:
            paths.append(path)
            continue
        if len(path) - 1 >= max_depth:
            continue
        lane = map_.lanes.get(lane_id)
        if lane is None:
            continue
        successors = set(lane.successors)
        successors.update(route_successors.get(lane_id, ()))
        for successor_id in sorted(successors, key=str):
            if successor_id in map_.lanes and successor_id not in path:
                queue.append((successor_id, path + (successor_id,)))
    return paths


def available_lanes_from_route(agent: AgentDecisionState, map_: Optional[Map]) -> FrozenSet[str]:
    """Derive LimSim's local route-compatible lane set.

    SUMO maps are grouped by edge metadata. Generic lanelet maps use lateral
    neighbor components and a bounded successor corridor as the closest
    topology-preserving equivalent.
    """

    if map_ is None or agent.lane_id is None or agent.lane_id not in map_.lanes:
        return frozenset()

    current_id = agent.lane_id
    current_section = _lateral_lane_section(current_id, map_)
    route_lane_ids = tuple(lane_id for lane_id in agent.route_lane_ids if lane_id in map_.lanes)
    try:
        route_start = route_lane_ids.index(current_id)
    except ValueError:
        route_start = 0
    route_suffix = route_lane_ids[route_start:]

    outside_section = [lane_id for lane_id in route_suffix if lane_id not in current_section]
    if not outside_section:
        return current_section
    target_id = next(
        (lane_id for lane_id in outside_section if not _lane_is_internal(lane_id, map_)),
        outside_section[-1],
    )
    target_section = _lateral_lane_section(target_id, map_)

    paths = []
    for source_id in sorted(current_section, key=str):
        paths.extend(
            _successor_paths_to_section(
                source_id,
                target_section,
                map_,
                route_lane_ids=route_suffix,
                max_depth=_AVAILABLE_LANE_MAX_SUCCESSOR_DEPTH,
            )
        )

    if _lane_is_internal(current_id, map_):
        available = {current_id}
        available.update(target_section)
        for path in paths:
            available.update(path)
        return frozenset(available)

    lane = map_.lanes[current_id]
    centerline = lane.centerline()
    lane_length = float(centerline.length) if centerline is not None else 0.0
    lane_progress = float(np.clip(agent.route_progress, 0.0, lane_length))
    if lane_length < _AVAILABLE_LANE_LOOK_FORWARD:
        if lane_progress < _AVAILABLE_LANE_NO_CHANGE:
            return current_section
    elif lane_length - lane_progress > max(lane_length / 3.0, _AVAILABLE_LANE_LOOK_FORWARD):
        return current_section

    available: Set[str] = set()
    for path in paths:
        if len(path) == 2:
            # Lanelet maps may represent an edge-to-edge transition directly,
            # without a distinct junction connector lane.
            available.update(path)
        else:
            available.update(lane_id for lane_id in path if lane_id not in target_section)
    return frozenset(available or {current_id})


def _route_path_from_lanes(route_lanes, map_: Map, recorded_route_lanes=()):
    """Build one route path, splicing recorded interior lane exits in place."""

    centerlines = []
    for lane_id in route_lanes:
        centerline = map_.lanes[lane_id].centerline()
        centerlines.append(
            np.asarray(centerline.coords, dtype=float) if centerline is not None else None
        )

    recorded_pairs = set(zip(recorded_route_lanes, recorded_route_lanes[1:]))
    for index, (first_id, second_id) in enumerate(zip(route_lanes, route_lanes[1:])):
        if (first_id, second_id) not in recorded_pairs:
            continue
        first_points = centerlines[index]
        second_points = centerlines[index + 1]
        if first_points is None or second_points is None or len(first_points) < 2:
            continue
        if not _route_connection_is_valid(first_id, second_id, map_):
            continue

        first_line = LineString(first_points)
        second_start = Point(second_points[0])
        join_progress = float(first_line.project(second_start))
        # Endpoint connections already concatenate correctly.  Only an
        # interior exit needs the incoming lane to be shortened.
        if join_progress >= first_line.length - 0.5:
            continue
        trimmed = np.asarray(substring(first_line, 0.0, join_progress).coords, dtype=float)
        if len(trimmed) < 2:
            continue
        trimmed[-1] = second_points[0]
        centerlines[index] = trimmed

    return polyline.concatenate(centerlines)


def _continuous_route_lanes(route_lanes, map_: Map, recorded_route_lanes=(), max_gap: float = 2.0):
    """Keep the ordered prefix connected by topology or route-local evidence."""

    if not route_lanes:
        return []
    continuous = [route_lanes[0]]
    for lane_id in route_lanes[1:]:
        previous = map_.lanes.get(continuous[-1])
        current = map_.lanes.get(lane_id)
        if previous is None or current is None:
            break
        route_local = any(
            first_id == continuous[-1] and second_id == lane_id
            for first_id, second_id in zip(recorded_route_lanes, recorded_route_lanes[1:])
        )
        route_local_valid = route_local and _route_connection_is_valid(
            continuous[-1], lane_id, map_
        )
        if lane_id not in previous.successors and not route_local_valid:
            break
        if route_local_valid:
            continuous.append(lane_id)
            continue
        previous_centerline = previous.centerline()
        current_centerline = current.centerline()
        if previous_centerline is None or current_centerline is None:
            break
        previous_end = np.asarray(previous_centerline.coords[-1], dtype=float)
        current_start = np.asarray(current_centerline.coords[0], dtype=float)
        if np.linalg.norm(previous_end - current_start) > max_gap:
            break
        continuous.append(lane_id)
    return continuous


def _next_route_successor(
    current_lane_id, map_: Map, visited_lane_ids, route_lane_ids=(), available_lane_ids=()
) -> Optional[str]:
    """Choose an unambiguous successor without inventing a branch choice."""

    current_lane = map_.lanes.get(current_lane_id)
    if current_lane is None:
        return None
    available = set(available_lane_ids)

    def is_available(lane_id) -> bool:
        return not available or lane_id in available

    recorded_next = None
    for index, lane_id in enumerate(route_lane_ids[:-1]):
        if lane_id == current_lane_id and route_lane_ids[index + 1] not in visited_lane_ids:
            candidate = route_lane_ids[index + 1]
            if (
                candidate in map_.lanes
                and is_available(candidate)
                and (
                    candidate in current_lane.successors
                    or _route_connection_is_valid(current_lane_id, candidate, map_)
                )
            ):
                recorded_next = candidate
            break
    if recorded_next is not None:
        return recorded_next

    candidates = [
        lane_id
        for lane_id in current_lane.successors
        if lane_id in map_.lanes and lane_id not in visited_lane_ids and is_available(lane_id)
    ]
    route_candidates = [lane_id for lane_id in candidates if lane_id in route_lane_ids]
    if len(route_candidates) == 1:
        return route_candidates[0]
    return candidates[0] if len(candidates) == 1 else None


def route_lanes_from_agent(agent: AgentDecisionState, map_: Map, max_routes: int) -> List[str]:
    """Build a route prefix from known topology and local route evidence."""

    if agent.lane_id is None or agent.lane_id not in map_.lanes:
        return []
    route_lanes = [agent.lane_id]
    while len(route_lanes) < max_routes:
        next_lane_id = _next_route_successor(
            route_lanes[-1], map_, route_lanes, agent.route_lane_ids, agent.available_lane_ids
        )
        if next_lane_id is None:
            break
        route_lanes.append(next_lane_id)
    return route_lanes


def route_has_continuation(
    route_lanes, map_: Map, recorded_route_lanes=(), available_lane_ids=()
) -> bool:
    """Return whether the selected route can legally continue past its end."""

    if not route_lanes:
        return False
    next_lane_id = _next_route_successor(
        route_lanes[-1], map_, route_lanes, recorded_route_lanes, available_lane_ids
    )
    if next_lane_id is None:
        return False
    return len(
        _continuous_route_lanes(list(route_lanes) + [next_lane_id], map_, recorded_route_lanes)
    ) > len(route_lanes)


def _lane_at_route_progress(
    route_lanes,
    map_: Map,
    progress: float,
    recorded_route_lanes=(),
    keep_exact_lane_end: bool = False,
):
    """Map combined-path progress back to a lane ID and lane-local progress."""

    recorded_pairs = set(zip(recorded_route_lanes, recorded_route_lanes[1:]))
    remaining = max(float(progress), 0.0)
    for index, lane_id in enumerate(route_lanes):
        lane = map_.lanes.get(lane_id)
        centerline = lane.centerline() if lane is not None else None
        if centerline is None:
            continue
        line = LineString(centerline)
        exit_progress = float(line.length)
        gap = 0.0
        if index + 1 < len(route_lanes):
            next_id = route_lanes[index + 1]
            next_lane = map_.lanes.get(next_id)
            next_centerline = next_lane.centerline() if next_lane is not None else None
            if next_centerline is not None:
                next_start = Point(next_centerline.coords[0])
                if (lane_id, next_id) in recorded_pairs and _route_connection_is_valid(
                    lane_id, next_id, map_
                ):
                    exit_progress = min(exit_progress, float(line.project(next_start)))
                exit_point = line.interpolate(exit_progress)
                gap = float(exit_point.distance(next_start))

        span = exit_progress + gap
        at_exact_end = keep_exact_lane_end and remaining <= span + 1e-9
        if index + 1 == len(route_lanes) or remaining < span - 1e-9 or at_exact_end:
            return lane_id, min(remaining, exit_progress)
        remaining -= span

    if not route_lanes:
        return None, 0.0
    last_id = route_lanes[-1]
    last_lane = map_.lanes.get(last_id)
    last_centerline = last_lane.centerline() if last_lane is not None else None
    last_length = float(last_centerline.length) if last_centerline is not None else 0.0
    return last_id, last_length


def _route_point_and_tangent(
    route_path: LineString, progress: float, fallback_heading: float
) -> Tuple[Point, np.ndarray]:
    """Return a route point and tangent, clamped to the route endpoint."""

    progress = float(np.clip(progress, 0.0, route_path.length))
    point = route_path.interpolate(progress)
    lookahead = route_path.interpolate(min(progress + 0.5, route_path.length))
    if lookahead.distance(point) > 1e-6:
        tangent = np.asarray([lookahead.x - point.x, lookahead.y - point.y], dtype=float)
    else:
        behind = route_path.interpolate(max(progress - 0.5, 0.0))
        tangent = np.asarray([point.x - behind.x, point.y - behind.y], dtype=float)

    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-6:
        tangent = np.asarray([np.cos(fallback_heading), np.sin(fallback_heading)], dtype=float)
    else:
        tangent /= tangent_norm
    return point, tangent


class LaneFollower:
    """Roll out actions along lane centerlines when possible."""

    def __init__(self, config: LimSimConfig):
        self.config = config

    def flowstate_transition(
        self,
        agent: AgentDecisionState,
        action: LimSimAction,
        map_: Optional[Map],
    ) -> Optional[AgentDecisionState]:
        """Apply one official LimSim ``DECISION_RESOLUTION`` transition.

        FlowState stores one endpoint per MCTS layer.  It does not integrate
        ``dt``-sampled trajectory points and then select the final sample.  In
        particular, the original implementation updates velocity before the
        longitudinal displacement formula for ``AC`` and ``DC``; the formulas
        below intentionally preserve that ordering.

        Returns:
            The next decision endpoint, or ``None`` once the vehicle travels
            beyond the end of its route-local lane corridor.
        """

        duration = self.config.decision_resolution
        speed = agent.speed
        lateral_offset = agent.lateral_offset

        if action == LimSimAction.KS:
            travel = speed * duration
            if map_ is not None and agent.lane_id in map_.lanes:
                lane_width = self._lane_width(map_.lanes[agent.lane_id])
                if abs(lateral_offset) < lane_width / 4.0:
                    lateral_offset = 0.0
        elif action == LimSimAction.AC:
            acceleration = self.config.acceleration
            speed = min(speed + acceleration * duration, self.config.max_speed)
            travel = speed * duration + 0.5 * acceleration * duration**2
        elif action == LimSimAction.DC:
            deceleration = self.config.deceleration
            speed = max(speed + deceleration * duration, 0.0)
            travel = max(0.0, speed * duration + 0.5 * deceleration * duration**2)
        elif action == LimSimAction.LCL:
            lateral_offset += self.config.lateral_speed * duration
            travel = speed * duration
        elif action == LimSimAction.LCR:
            lateral_offset -= self.config.lateral_speed * duration
            travel = speed * duration
        else:  # pragma: no cover - LimSimAction is exhaustive
            raise ValueError(f"Unsupported LimSim action: {action!r}")

        endpoint = agent.with_updates(
            speed=float(speed),
            lateral_offset=float(lateral_offset),
            action=action,
        )
        endpoint = self._flowstate_lane_change(endpoint, action, map_)

        if map_ is None or endpoint.lane_id is None or endpoint.lane_id not in map_.lanes:
            return endpoint.with_updates(
                x=endpoint.x + travel * np.cos(endpoint.heading),
                y=endpoint.y + travel * np.sin(endpoint.heading),
                route_progress=endpoint.route_progress + travel,
            )
        return self._flowstate_route_endpoint(endpoint, action, map_, travel)

    def _flowstate_lane_change(
        self,
        agent: AgentDecisionState,
        action: LimSimAction,
        map_: Optional[Map],
    ) -> AgentDecisionState:
        """Convert an official lateral endpoint into the adjacent lane frame."""

        if action not in {LimSimAction.LCL, LimSimAction.LCR}:
            return agent
        if map_ is None or agent.lane_id is None or agent.lane_id not in map_.lanes:
            return agent

        current_lane = map_.lanes[agent.lane_id]
        current_width = self._lane_width(current_lane)
        crosses_left = action == LimSimAction.LCL and agent.lateral_offset > current_width / 2.0
        crosses_right = action == LimSimAction.LCR and agent.lateral_offset < -current_width / 2.0
        if not crosses_left and not crosses_right:
            return agent

        neighbor_ids = current_lane.left_neighbors if crosses_left else current_lane.right_neighbors
        next_lane_id = self._choose_neighbor_lane(agent, neighbor_ids, map_)
        if next_lane_id is None:
            boundary = current_width / 2.0
            clipped = boundary if crosses_left else -boundary
            return agent.with_updates(lateral_offset=clipped)

        next_width = self._lane_width(map_.lanes[next_lane_id])
        frame_shift = current_width / 2.0 + next_width / 2.0
        next_offset = (
            agent.lateral_offset - frame_shift
            if crosses_left
            else agent.lateral_offset + frame_shift
        )
        return agent.with_updates(lane_id=next_lane_id, lateral_offset=next_offset)

    def _flowstate_route_endpoint(
        self,
        agent: AgentDecisionState,
        action: LimSimAction,
        map_: Map,
        travel: float,
    ) -> Optional[AgentDecisionState]:
        """Advance one endpoint along the route and preserve lane-end removal."""

        route_limit = max(self.config.max_routes_per_agent, len(agent.route_lane_ids), 1)
        maximum_route_limit = max(route_limit, len(map_.lanes) + 1)
        while True:
            route_path, route_lanes, start_progress = self._select_route(
                agent, action, map_, max_routes=route_limit
            )
            if route_path is None or route_path.length <= 1e-6 or not route_lanes:
                return None

            target_progress = start_progress + travel
            if target_progress <= route_path.length + 1e-9:
                target_progress = min(target_progress, route_path.length)
                point, tangent = _route_point_and_tangent(
                    route_path, target_progress, agent.heading
                )
                heading = spatial.normalize_angle(np.arctan2(tangent[1], tangent[0]))
                x = float(point.x - agent.lateral_offset * np.sin(heading))
                y = float(point.y + agent.lateral_offset * np.cos(heading))
                lane_id, lane_progress = _lane_at_route_progress(
                    route_lanes,
                    map_,
                    target_progress,
                    agent.route_lane_ids,
                    keep_exact_lane_end=True,
                )
                return agent.with_updates(
                    x=x,
                    y=y,
                    heading=heading,
                    lane_id=lane_id,
                    route_progress=lane_progress,
                )

            if not route_has_continuation(
                route_lanes, map_, agent.route_lane_ids, agent.available_lane_ids
            ):
                return None
            if route_limit >= maximum_route_limit:
                return None
            route_limit = min(maximum_route_limit, max(route_limit + 1, route_limit * 2))

    def rollout(
        self,
        agent: AgentDecisionState,
        action: LimSimAction,
        map_: Optional[Map],
        steps: Optional[int] = None,
    ) -> List[AgentDecisionState]:
        """Generate future states for one agent under one high-level action."""

        horizon = self.config.horizon_steps if steps is None else steps
        route_path, route_lanes, start_progress = self._select_route(agent, action, map_)

        states = []
        speed = agent.speed
        progress = start_progress
        current = agent
        for _ in range(horizon):
            accel = 0.0
            if action == LimSimAction.AC:
                accel = self.config.acceleration
            elif action == LimSimAction.DC:
                accel = self.config.deceleration
            speed = float(
                np.clip(
                    speed + accel * self.config.dt, self.config.min_speed, self.config.max_speed
                )
            )
            travel = speed * self.config.dt
            transition = self._lane_transition(current, action, map_)
            if transition is not None:
                current = transition
                route_path, route_lanes, progress = self._select_route(current, action, map_)

            if route_path is not None and route_path.length > 1e-6:
                progress += travel
                terminal = bool(
                    route_lanes
                    and map_ is not None
                    and not route_has_continuation(
                        route_lanes, map_, current.route_lane_ids, current.available_lane_ids
                    )
                )
                if terminal and progress > route_path.length:
                    break
                point, tangent = _route_point_and_tangent(route_path, progress, current.heading)
                heading = spatial.normalize_angle(np.arctan2(tangent[1], tangent[0]))
                lateral_offset = self._next_lateral_offset(current, action)
                x = float(point.x - lateral_offset * np.sin(heading))
                y = float(point.y + lateral_offset * np.cos(heading))
                lane_id, lane_progress = _lane_at_route_progress(
                    route_lanes, map_, progress, current.route_lane_ids
                )
                current = current.with_updates(
                    x=x,
                    y=y,
                    heading=heading,
                    speed=speed,
                    action=action,
                    lane_id=lane_id,
                    route_progress=lane_progress,
                    lateral_offset=lateral_offset,
                )
            else:
                current = current.with_updates(
                    x=current.x + travel * np.cos(current.heading),
                    y=current.y + travel * np.sin(current.heading),
                    speed=speed,
                    action=action,
                )
            states.append(current)

        return states

    def _lane_transition(
        self, agent: AgentDecisionState, action: LimSimAction, map_: Optional[Map]
    ) -> Optional[AgentDecisionState]:
        """Smoothly transition lane_id when the lateral offset crosses a lane boundary.

        Instead of teleporting the lateral offset to the new lane's frame (which
        produces a discontinuous jump of ~one lane width), this method projects
        the vehicle's current Cartesian position onto the neighbor lane's
        centerline to obtain a **continuous** lateral offset in the new frame.
        """

        if action not in {LimSimAction.LCL, LimSimAction.LCR}:
            return None
        if map_ is None or agent.lane_id is None or agent.lane_id not in map_.lanes:
            return None
        direction = "left" if action == LimSimAction.LCL else "right"
        if not SemanticMapQuery(map_).get_lane_change_permission(
            agent.lane_id, direction, s=agent.route_progress
        ):
            return None

        current_lane = map_.lanes[agent.lane_id]
        current_width = self._lane_width(current_lane)
        if action == LimSimAction.LCL and agent.lateral_offset <= current_width / 2.0:
            return None
        if action == LimSimAction.LCR and agent.lateral_offset >= -current_width / 2.0:
            return None

        neighbor_ids = (
            current_lane.left_neighbors
            if action == LimSimAction.LCL
            else current_lane.right_neighbors
        )
        next_lane_id = self._choose_neighbor_lane(agent, neighbor_ids, map_)
        if next_lane_id is None:
            clipped_offset = np.clip(
                agent.lateral_offset, -current_width / 2.0, current_width / 2.0
            )
            return agent.with_updates(lateral_offset=float(clipped_offset))

        next_lane = map_.lanes[next_lane_id]
        next_centerline = next_lane.centerline()
        next_centerline = (
            np.asarray(next_centerline.coords, dtype=float) if next_centerline is not None else None
        )
        if next_centerline is None or len(next_centerline) < 2:
            return agent.with_updates(lane_id=next_lane_id)

        # Project the vehicle's current Cartesian position onto the neighbor
        # lane's centerline to get a continuous lateral offset in the new frame.
        line = LineString(next_centerline)
        next_progress = float(line.project(Point(agent.x, agent.y)))
        point = line.interpolate(next_progress)
        lookahead = line.interpolate(min(next_progress + 0.5, line.length))
        heading = spatial.normalize_angle(np.arctan2(lookahead.y - point.y, lookahead.x - point.x))

        # Signed distance from the neighbor centerline = continuous lateral offset.
        # This avoids the 3.6 m arithmetic jump that the old formula produced.
        dx = agent.x - point.x
        dy = agent.y - point.y
        next_offset = float(-dx * np.sin(heading) + dy * np.cos(heading))

        return agent.with_updates(
            lane_id=next_lane_id, route_progress=next_progress, lateral_offset=next_offset
        )

    def _choose_neighbor_lane(
        self, agent: AgentDecisionState, lane_ids, map_: Map
    ) -> Optional[str]:
        candidates = [lane_id for lane_id in lane_ids if lane_id in map_.lanes]
        if not candidates:
            return None
        point = Point(agent.x, agent.y)

        def _lane_centerline_distance(lid: str) -> float:
            centerline = map_.lanes[lid].centerline()
            centerline = (
                np.asarray(centerline.coords, dtype=float) if centerline is not None else None
            )
            return (
                LineString(centerline).distance(point) if centerline is not None else float("inf")
            )

        return min(candidates, key=_lane_centerline_distance)

    def _lane_width(self, lane) -> float:
        width = lane.get_width(default=self.config.default_lane_width)
        return self.config.default_lane_width if width is None else float(width)

    def _next_lateral_offset(self, agent: AgentDecisionState, action: LimSimAction) -> float:
        if action == LimSimAction.LCL:
            return agent.lateral_offset + self.config.lateral_speed * self.config.dt
        if action == LimSimAction.LCR:
            return agent.lateral_offset - self.config.lateral_speed * self.config.dt
        return agent.lateral_offset

    def _select_route(
        self,
        agent: AgentDecisionState,
        action: LimSimAction,
        map_: Optional[Map],
        max_routes: Optional[int] = None,
    ) -> Tuple[Optional[LineString], Tuple[str, ...], float]:
        if map_ is None or agent.lane_id is None or agent.lane_id not in map_.lanes:
            return None, agent.route_lane_ids, agent.route_progress
        if abs(agent.lateral_offset) > self.config.max_lateral_offset_for_lane_rollout:
            return None, agent.route_lane_ids, agent.route_progress

        route_lanes = route_lanes_from_agent(
            agent,
            map_,
            self.config.max_routes_per_agent if max_routes is None else max_routes,
        )

        route_lanes = _continuous_route_lanes(route_lanes, map_, agent.route_lane_ids)
        path_array = _route_path_from_lanes(route_lanes, map_, agent.route_lane_ids)
        if path_array is None or len(path_array) < 2:
            return None, tuple(route_lanes), 0.0

        route_path = LineString(path_array)
        location = Point(agent.x, agent.y)
        hinted_progress = float(agent.route_progress)
        if np.isfinite(hinted_progress) and hinted_progress >= 0.0:
            hinted_point, _ = _route_point_and_tangent(route_path, hinted_progress, agent.heading)
            if hinted_point.distance(location) <= self.config.lane_match_radius:
                return route_path, tuple(route_lanes), hinted_progress

        # Project on the current-lane prefix instead of the whole route.  A
        # roundabout or a nearby parallel lane can otherwise win the global
        # nearest-point query and jump the longitudinal progress forward.
        current_lane = map_.lanes.get(agent.lane_id)
        current_centerline = current_lane.centerline() if current_lane is not None else None
        if current_centerline is not None and current_centerline.length > 1e-6:
            prefix_limit = min(route_path.length, float(current_centerline.length) + 4.0)
            prefix = (
                route_path
                if prefix_limit >= route_path.length
                else substring(route_path, 0.0, prefix_limit)
            )
            start_progress = prefix.project(location)
        else:
            start_progress = route_path.project(location)
        return route_path, tuple(route_lanes), float(start_progress)


def is_action_valid(agent: AgentDecisionState, action: LimSimAction, map_: Optional[Map]) -> bool:
    """Check map-topology feasibility for a high-level action."""

    if action not in {LimSimAction.LCL, LimSimAction.LCR}:
        return True
    if map_ is None or agent.lane_id is None or agent.lane_id not in map_.lanes:
        return False
    lane = map_.lanes[agent.lane_id]
    if action == LimSimAction.LCL:
        return len(lane.left_neighbors) > 0 and SemanticMapQuery(map_).get_lane_change_permission(
            agent.lane_id, "left", s=agent.route_progress
        )
    if action == LimSimAction.LCR:
        return len(lane.right_neighbors) > 0 and SemanticMapQuery(map_).get_lane_change_permission(
            agent.lane_id, "right", s=agent.route_progress
        )
    return False
