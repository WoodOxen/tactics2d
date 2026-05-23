# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Socket-driven intersection generator for T-junction and cross-junction road elements."""

from __future__ import annotations

from typing import Any

import numpy as np
from shapely.geometry import LineString, Polygon

from tactics2d.interpolator import Bezier
from tactics2d.map.element import Junction, Lane, LaneRelationship, RoadLine

from ..geometry.geometry_utils import offset_polyline, sample_centerline
from ..geometry.module_geometry import curvature_stats, has_self_intersection, polyline_length
from ..rules.lane_marking_rules import intersection_mark_kwargs
from ..rules.module_types import RoadModuleResult, RoadPort, make_port


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _unit(heading: float) -> np.ndarray:
    return np.array([np.cos(heading), np.sin(heading)], dtype=float)


def _left_normal(heading: float) -> np.ndarray:
    return np.array([-np.sin(heading), np.cos(heading)], dtype=float)


def _bezier_connection(
    p0: np.ndarray, h0: float, p3: np.ndarray, h3: float, step_size: float = 0.1
) -> np.ndarray:
    """Generate a C1-continuous cubic Bezier curve between two poses."""
    p0 = np.asarray(p0, dtype=float)
    p3 = np.asarray(p3, dtype=float)

    chord = float(np.linalg.norm(p3 - p0))
    if chord < 1e-6:
        return np.vstack([p0, p3])

    delta = abs(_wrap_angle(h3 - h0))
    tension = 1.0 / 3.0 + (delta / np.pi) * (1.0 / 6.0)
    handle_length = chord * tension

    p1 = p0 + handle_length * _unit(h0)
    p2 = p3 - handle_length * _unit(h3)

    n = max(2, int(chord / step_size) + 1)
    return Bezier.get_curve(np.array([p0, p1, p2, p3]), n, order=3)


def _arm_boundary_point(
    center: np.ndarray, heading_outward: float, radius: float, curvature: float, step_size: float
) -> tuple[np.ndarray, float]:
    """Compute an arm boundary point and its inward heading."""
    pts = sample_centerline(center, heading_outward, radius, curvature, step_size)

    if len(pts) < 2:
        boundary_pt = center + radius * _unit(heading_outward)
        return boundary_pt, heading_outward + np.pi

    boundary_pt = pts[-1]
    tangent = pts[-1] - pts[-2]
    heading_out = float(np.arctan2(tangent[1], tangent[0]))

    return boundary_pt, heading_out + np.pi


def _normalize_arm(
    center: np.ndarray,
    arm: dict[str, Any] | RoadPort,
    default_radius: float,
    default_lane_width: float,
    default_speed_limit: float,
    step_size: float,
) -> dict[str, Any]:
    """Convert an arm descriptor into an internal arm record."""
    if isinstance(arm, RoadPort):
        point = np.asarray(arm.point, dtype=float)
        heading_outward = float(arm.heading)

        return {
            "point": point,
            "heading_outward": heading_outward,
            "heading_inward": heading_outward + np.pi,
            "lane_num": int(arm.lane_num),
            "lane_width": float(arm.lane_width),
            "speed_limit": float(arm.speed_limit),
            "curvature": 0.0,
            "radius": float(np.linalg.norm(point - center)),
        }

    if "heading" not in arm or "lane_num" not in arm:
        raise ValueError("Each arm dict must contain 'heading' and 'lane_num'.")

    heading_outward = float(arm["heading"])
    arm_radius = float(arm.get("radius", default_radius))
    curvature = float(arm.get("curvature", 0.0))

    point, heading_inward = _arm_boundary_point(
        center=center,
        heading_outward=heading_outward,
        radius=arm_radius,
        curvature=curvature,
        step_size=step_size,
    )

    return {
        "point": point,
        "heading_outward": heading_inward + np.pi,
        "heading_inward": heading_inward,
        "lane_num": int(arm["lane_num"]),
        "lane_width": float(arm.get("lane_width", default_lane_width)),
        "speed_limit": float(arm.get("speed_limit", default_speed_limit)),
        "curvature": curvature,
        "radius": arm_radius,
    }


def _arm_boundary_lines(
    boundary_pt: np.ndarray, heading_inward: float, lane_num: int, lane_width: float
) -> list[np.ndarray]:
    """Compute boundary points across an intersection arm throat."""
    normal = _left_normal(heading_inward)
    points: list[np.ndarray] = []

    for i in range(lane_num, 0, -1):
        points.append(boundary_pt + i * lane_width * normal)

    points.append(boundary_pt)

    for i in range(1, lane_num + 1):
        points.append(boundary_pt - i * lane_width * normal)

    return points


def _arm_lane_centers(
    boundary_pt: np.ndarray, heading_inward: float, lane_num: int, lane_width: float
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Compute incoming and outgoing lane centre points at an arm boundary."""
    normal = _left_normal(heading_inward)

    incoming_pts = [boundary_pt - (i + 0.5) * lane_width * normal for i in range(lane_num)]
    outgoing_pts = [boundary_pt + (i + 0.5) * lane_width * normal for i in range(lane_num)]

    return incoming_pts, outgoing_pts


def _turn_direction(inc_heading: float, out_heading: float) -> str:
    """Classify a turn as right, straight, or left."""
    inc_outward = inc_heading + np.pi
    angle = _wrap_angle(out_heading - inc_outward)

    if angle > np.pi / 4.0:
        return "left"
    if angle < -np.pi / 4.0:
        return "right"

    return "straight"


def _lane_pairs_for_turn(
    turn: str, incoming_lane_num: int, outgoing_lane_num: int
) -> list[tuple[int, int]]:
    """Select incoming/outgoing lane index pairs for a turn type."""
    n = min(incoming_lane_num, outgoing_lane_num)

    if n <= 0:
        return []

    if turn == "right":
        return [(0, 0)]

    if turn == "straight":
        return [(i, i) for i in range(n)]

    return [(n - 1, n - 1)]


def _junction_shape_from_arm_boundaries(
    normalized_arms: list[dict[str, Any]],
    arm_boundary_lines: list[list[np.ndarray]],
    step_size: float,
) -> list[tuple[float, float]]:
    """Build a stable junction pavement shape from arm throat boundaries."""
    arm_data = list(zip(normalized_arms, arm_boundary_lines))
    arm_data.sort(key=lambda item: item[0]["heading_inward"])

    shape_pts: list[tuple[float, float]] = []
    arm_num = len(arm_data)

    for i in range(arm_num):
        curr_arm, curr_boundary = arm_data[i]

        for pt in curr_boundary:
            shape_pts.append((float(pt[0]), float(pt[1])))

        next_arm, next_boundary = arm_data[(i + 1) % arm_num]

        p0 = curr_boundary[-1]
        h0 = float(curr_arm["heading_inward"])
        p3 = next_boundary[0]
        h3 = float(next_arm["heading_inward"] + np.pi)

        corner_pts = _bezier_connection(p0, h0, p3, h3, step_size)

        if len(corner_pts) > 2:
            for pt in corner_pts[1:-1]:
                shape_pts.append((float(pt[0]), float(pt[1])))

    if shape_pts:
        shape_pts.append(shape_pts[0])

    return shape_pts


def _corner_boundary_roadlines(
    normalized_arms: list[dict[str, Any]],
    arm_boundary_lines: list[list[np.ndarray]],
    step_size: float,
    id_counter: int,
) -> tuple[list[RoadLine], int]:
    """Generate boundary RoadLines for junction corner curves only.

    Each corner is the Bezier arc connecting adjacent arm throats. These are
    the curved edges that show as raw polygon boundaries without this fix.
    Arm throat edges are not included because road segment edge lines cover them.
    """
    arm_data = list(zip(normalized_arms, arm_boundary_lines))
    arm_data.sort(key=lambda item: item[0]["heading_inward"])

    corner_roadlines: list[RoadLine] = []
    arm_num = len(arm_data)

    for i in range(arm_num):
        curr_arm, curr_boundary = arm_data[i]
        next_arm, next_boundary = arm_data[(i + 1) % arm_num]

        p0 = curr_boundary[-1]
        h0 = float(curr_arm["heading_inward"])
        p3 = next_boundary[0]
        h3 = float(next_arm["heading_inward"] + np.pi)

        corner_pts = _bezier_connection(p0, h0, p3, h3, step_size)

        if len(corner_pts) < 2:
            continue

        corner_roadlines.append(
            RoadLine(
                id_=str(id_counter),
                geometry=LineString(corner_pts),
                type_="road_border",
                subtype="solid",
                color="white",
                custom_tags={
                    "module": "intersection",
                    "role": "corner_boundary",
                    "from_arm": i,
                    "to_arm": (i + 1) % arm_num,
                },
            )
        )
        id_counter += 1

    return corner_roadlines, id_counter


def _intersection_mark_kwargs(tags: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return RoadLine kwargs for an intersection connection marking."""
    custom_tags = {"marking_role": "intersection_connection"}

    if tags is not None:
        custom_tags.update(tags)

    return intersection_mark_kwargs("connection", custom_tags=custom_tags)


def intersection(
    center: np.ndarray,
    arms: list[dict[str, Any] | RoadPort],
    lane_width: float = 3.5,
    radius: float = 8.0,
    speed_limit: float = 30.0,
    step_size: float = 0.1,
    id_offset: int = 0,
) -> RoadModuleResult:
    """Generate a T-junction or cross-junction.

    Args:
        center: Junction centre point.
        arms: Arm descriptors.
        lane_width: Default lane width.
        radius: Default arm boundary radius for dict arms.
        speed_limit: Default speed limit.
        step_size: Sampling interval.
        id_offset: Starting id.

    Returns:
        RoadModuleResult with lanes, roadlines, junctions, ports, interfaces and quality.
    """
    center = np.asarray(center, dtype=float)

    if len(arms) not in (3, 4):
        raise ValueError("intersection requires 3 or 4 arms.")

    if step_size <= 0.0:
        raise ValueError("step_size must be positive.")

    normalized_arms = [
        _normalize_arm(
            center=center,
            arm=arm,
            default_radius=radius,
            default_lane_width=lane_width,
            default_speed_limit=speed_limit,
            step_size=step_size,
        )
        for arm in arms
    ]

    for i, arm in enumerate(normalized_arms):
        if arm["lane_num"] < 1:
            raise ValueError(f"arm[{i}] lane_num must be >= 1.")
        if arm["lane_width"] <= 0.0:
            raise ValueError(f"arm[{i}] lane_width must be positive.")

    id_counter = id_offset
    n_arms = len(normalized_arms)

    arm_incoming_pts: list[list[np.ndarray]] = []
    arm_outgoing_pts: list[list[np.ndarray]] = []
    arm_boundary_lines: list[list[np.ndarray]] = []

    for arm in normalized_arms:
        incoming_pts, outgoing_pts = _arm_lane_centers(
            boundary_pt=arm["point"],
            heading_inward=arm["heading_inward"],
            lane_num=arm["lane_num"],
            lane_width=arm["lane_width"],
        )
        arm_incoming_pts.append(incoming_pts)
        arm_outgoing_pts.append(outgoing_pts)

        arm_boundary_lines.append(
            _arm_boundary_lines(
                boundary_pt=arm["point"],
                heading_inward=arm["heading_inward"],
                lane_num=arm["lane_num"],
                lane_width=arm["lane_width"],
            )
        )

    lanes: list[Lane] = []
    roadlines: list[RoadLine] = []
    incoming_lane_ids: list[list[str]] = [[] for _ in range(n_arms)]
    outgoing_lane_ids: list[list[str]] = [[] for _ in range(n_arms)]
    connection_groups: dict[tuple[int, int], list[Lane]] = {}
    connection_centerlines: list[np.ndarray] = []

    for inc_idx, inc_arm in enumerate(normalized_arms):
        h_in = float(inc_arm["heading_inward"])

        for out_idx, out_arm in enumerate(normalized_arms):
            if inc_idx == out_idx:
                continue

            h_out = float(out_arm["heading_outward"])
            turn = _turn_direction(h_in, h_out)
            pairs = _lane_pairs_for_turn(
                turn=turn,
                incoming_lane_num=inc_arm["lane_num"],
                outgoing_lane_num=out_arm["lane_num"],
            )

            for inc_lane_idx, out_lane_idx in pairs:
                p0 = arm_incoming_pts[inc_idx][inc_lane_idx]
                p3 = arm_outgoing_pts[out_idx][out_lane_idx]
                center_pts = _bezier_connection(p0, h_in, p3, h_out, step_size)

                if len(center_pts) < 2:
                    continue

                lane_w = min(float(inc_arm["lane_width"]), float(out_arm["lane_width"]))
                lane_speed = min(
                    float(inc_arm["speed_limit"]), float(out_arm["speed_limit"]), float(speed_limit)
                )

                left_pts = offset_polyline(center_pts, lane_w / 2.0)
                right_pts = offset_polyline(center_pts, -lane_w / 2.0)

                left_geom = LineString(left_pts)
                right_geom = LineString(right_pts)

                left_roadline = RoadLine(
                    id_=str(id_counter),
                    geometry=left_geom,
                    **_intersection_mark_kwargs(
                        {"from_arm": inc_idx, "to_arm": out_idx, "turn": turn, "side": "left"}
                    ),
                )
                id_counter += 1

                right_roadline = RoadLine(
                    id_=str(id_counter),
                    geometry=right_geom,
                    **_intersection_mark_kwargs(
                        {"from_arm": inc_idx, "to_arm": out_idx, "turn": turn, "side": "right"}
                    ),
                )
                id_counter += 1

                lane = Lane(
                    id_=str(id_counter),
                    left_side=left_geom,
                    right_side=right_geom,
                    subtype="road",
                    speed_limit=lane_speed,
                    speed_limit_unit="km/h",
                    line_ids={"left": [left_roadline.id_], "right": [right_roadline.id_]},
                    custom_tags={
                        "module": "intersection",
                        "from_arm": inc_idx,
                        "to_arm": out_idx,
                        "incoming_lane_index": inc_lane_idx,
                        "outgoing_lane_index": out_lane_idx,
                        "turn": turn,
                    },
                )
                id_counter += 1

                lanes.append(lane)
                roadlines.extend([left_roadline, right_roadline])

                outgoing_lane_ids[inc_idx].append(lane.id_)
                incoming_lane_ids[out_idx].append(lane.id_)
                connection_groups.setdefault((inc_idx, out_idx), []).append(lane)
                connection_centerlines.append(center_pts)

    for group_lanes in connection_groups.values():
        for i, lane in enumerate(group_lanes):
            if i > 0:
                lane.add_related_lane(group_lanes[i - 1].id_, LaneRelationship.LEFT_NEIGHBOR)
            if i < len(group_lanes) - 1:
                lane.add_related_lane(group_lanes[i + 1].id_, LaneRelationship.RIGHT_NEIGHBOR)

    junction_shape = _junction_shape_from_arm_boundaries(
        normalized_arms=normalized_arms, arm_boundary_lines=arm_boundary_lines, step_size=step_size
    )

    corner_rls, id_counter = _corner_boundary_roadlines(
        normalized_arms=normalized_arms,
        arm_boundary_lines=arm_boundary_lines,
        step_size=step_size,
        id_counter=id_counter,
    )
    roadlines.extend(corner_rls)

    junction = Junction(
        id_=str(id_counter),
        custom_tags={
            "sumo_id": f"junction_{id_counter}",
            "x": str(float(center[0])),
            "y": str(float(center[1])),
            "type": "road",
            "junction_type": "intersection",
            "sumo_type": "priority",
            "shape": junction_shape,
        },
    )
    id_counter += 1

    ports: dict[str, RoadPort] = {}
    interfaces: list[dict[str, Any]] = []

    for arm_idx, arm in enumerate(normalized_arms):
        inward_port = RoadPort(
            point=np.asarray(arm["point"], dtype=float),
            heading=float(arm["heading_inward"]),
            lane_num=int(arm["lane_num"]),
            lane_width=float(arm["lane_width"]),
            speed_limit=float(arm["speed_limit"]),
        )
        outward_port = RoadPort(
            point=np.asarray(arm["point"], dtype=float),
            heading=float(arm["heading_outward"]),
            lane_num=int(arm["lane_num"]),
            lane_width=float(arm["lane_width"]),
            speed_limit=float(arm["speed_limit"]),
        )

        ports[f"arm_{arm_idx}_in"] = make_port(
            inward_port,
            kind="intersection_in",
            name=f"arm_{arm_idx}_in",
            lane_ids=tuple(outgoing_lane_ids[arm_idx]),
            metadata={"arm_index": arm_idx, "direction": "into_intersection"},
        )
        ports[f"arm_{arm_idx}_out"] = make_port(
            outward_port,
            kind="intersection_out",
            name=f"arm_{arm_idx}_out",
            lane_ids=tuple(incoming_lane_ids[arm_idx]),
            metadata={"arm_index": arm_idx, "direction": "out_of_intersection"},
        )

        interfaces.append(
            {
                "point": np.asarray(arm["point"], dtype=float),
                "heading": float(arm["heading_outward"]),
                "lane_num": int(arm["lane_num"]),
                "lane_width": float(arm["lane_width"]),
                "speed_limit": float(arm["speed_limit"]),
                "curvature": float(arm["curvature"]),
                "incoming_lane_ids": incoming_lane_ids[arm_idx],
                "outgoing_lane_ids": outgoing_lane_ids[arm_idx],
            }
        )

    max_curvature = 0.0
    max_curvature_rate = 0.0
    total_connection_length = 0.0
    connection_self_intersection = False

    for centerline in connection_centerlines:
        stats = curvature_stats(centerline)
        max_curvature = max(max_curvature, stats["max_abs_curvature"])
        max_curvature_rate = max(max_curvature_rate, stats["max_abs_curvature_rate"])
        total_connection_length += polyline_length(centerline)
        connection_self_intersection = connection_self_intersection or has_self_intersection(
            centerline
        )

    junction_area = 0.0
    try:
        junction_area = float(Polygon(junction_shape).area)
    except Exception:
        junction_area = 0.0

    accepted_reasons = []
    if connection_self_intersection:
        accepted_reasons.append("connection_self_intersection")

    quality = {
        "module": "intersection",
        "arm_num": n_arms,
        "connection_count": len(lanes),
        "total_connection_length": total_connection_length,
        "connection_self_intersection": connection_self_intersection,
        "max_abs_curvature": max_curvature,
        "max_abs_curvature_rate": max_curvature_rate,
        "junction_area": junction_area,
        "accepted_reasons": accepted_reasons,
        "accepted": len(accepted_reasons) == 0,
    }

    return RoadModuleResult(
        lanes=lanes,
        roadlines=roadlines,
        junctions=[junction],
        ports=ports,
        interfaces=interfaces,
        quality=quality,
        id_counter=id_counter,
    )
