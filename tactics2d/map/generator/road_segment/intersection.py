# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Socket-driven intersection generator for T-junction and cross-junction road elements."""

from __future__ import annotations

from typing import Any

import numpy as np

from tactics2d.geometry import polyline, spatial
from tactics2d.map.element import Lane, RoadLine
from tactics2d.map.generator.rules.lane_marking_rules import roadline_render_kwargs
from tactics2d.map.generator.rules.module_types import RoadModuleResult, RoadPort

from .element_builder import (
    add_ordered_lane_neighbors,
    build_junction_arm_ports,
    build_lane_from_boundaries,
    build_pavement_junction,
    build_roadline_from_points,
    merge_marking_kwargs,
)
from .reference_line import bezier_connection, sample_centerline
from .road_segment import RoadSegment


def _left_normal(heading: float) -> np.ndarray:
    """Return the unit left-normal vector for a given heading.

    Args:
        heading: Heading angle in radians.

    Returns:
        Unit normal vector pointing 90 degrees to the left.
    """
    return np.array([-np.sin(heading), np.cos(heading)], dtype=float)


def _arm_boundary_point(
    center: np.ndarray, heading_outward: float, radius: float, curvature: float, step_size: float
) -> tuple[np.ndarray, float]:
    """Compute an intersection arm boundary point and its inward heading.

    Samples a short centreline segment from ``center`` outward, then reads
    the endpoint position and tangent to handle curved approaches.

    Args:
        center: Junction centre in world coordinates.
        heading_outward: Outward heading of the arm in radians.
        radius: Arm reach from the junction centre in metres.
        curvature: Signed approach curvature in m⁻¹.
        step_size: Sampling interval in metres.

    Returns:
        Tuple ``(boundary_point, inward_heading)`` where ``inward_heading``
        is the heading pointing from the boundary into the junction.
    """
    pts = sample_centerline(center, heading_outward, radius, curvature, step_size)

    if len(pts) < 2:
        boundary_pt = center + radius * spatial.heading_unit(heading_outward)
        return boundary_pt, heading_outward + np.pi

    boundary_pt = pts[-1]
    tangent = pts[-1] - pts[-2]
    heading_out = float(np.arctan2(tangent[1], tangent[0]))

    return boundary_pt, heading_out + np.pi


def _normalize_intersection_arm(
    center: np.ndarray,
    arm: dict[str, Any] | RoadPort,
    default_radius: float,
    default_lane_width: float,
    default_speed_limit: float,
    step_size: float,
) -> dict[str, Any]:
    """Convert an arm descriptor into an internal arm record.

    For :class:`RoadPort` arms the socket position is taken as-is.  For dict
    arms the socket is placed at the end of a ``sample_centerline()`` path so
    that non-zero ``curvature`` shifts the boundary point (unlike the roundabout
    arm normaliser, which always places the socket directly on the outer ring
    circle regardless of curvature).

    Args:
        center: Junction centre in world coordinates.
        arm: Arm descriptor — either a :class:`RoadPort` or a ``dict`` with
            keys ``"heading"`` and ``"lane_num"``, and optionally ``"radius"``,
            ``"curvature"``, ``"lane_width"``, ``"speed_limit"``.
        default_radius: Fallback arm boundary radius in metres for dict arms.
        default_lane_width: Fallback lane width in metres for dict arms.
        default_speed_limit: Fallback speed limit in km/h for dict arms.
        step_size: Sampling interval in metres used by ``sample_centerline``.

    Returns:
        Dict with keys ``"point"``, ``"heading_outward"``, ``"heading_inward"``,
        ``"lane_num"``, ``"lane_width"``, ``"speed_limit"``, ``"curvature"``,
        and ``"radius"``.

    Raises:
        ValueError: If a dict arm is missing ``"heading"`` or ``"lane_num"``.
    """
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
    """Return the lateral boundary points across an intersection arm throat.

    Points are ordered from the left extreme (incoming side) to the right
    extreme (outgoing side), centred on ``boundary_pt``.

    Args:
        boundary_pt: Arm socket position in world coordinates.
        heading_inward: Heading pointing from the arm into the junction.
        lane_num: Number of lanes in the arm.
        lane_width: Width per lane in metres.

    Returns:
        List of ``2 * lane_num + 1`` points spanning the full arm throat width.
    """
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
    """Return incoming and outgoing lane centre points at an arm throat.

    Incoming lanes carry traffic into the junction; outgoing lanes carry
    traffic out. Both sets are ordered from lane index 0 outward.

    Args:
        boundary_pt: Arm socket position in world coordinates.
        heading_inward: Heading pointing from the arm into the junction.
        lane_num: Number of lanes in the arm.
        lane_width: Width per lane in metres.

    Returns:
        Tuple ``(incoming_pts, outgoing_pts)`` each of length ``lane_num``.
    """
    normal = _left_normal(heading_inward)

    incoming_pts = [boundary_pt - (i + 0.5) * lane_width * normal for i in range(lane_num)]
    outgoing_pts = [boundary_pt + (i + 0.5) * lane_width * normal for i in range(lane_num)]

    return incoming_pts, outgoing_pts


def _turn_direction(inc_heading: float, out_heading: float) -> str:
    """Classify a connection as a right turn, straight, or left turn.

    Args:
        inc_heading: Inward heading of the incoming arm in radians.
        out_heading: Outward heading of the outgoing arm in radians.

    Returns:
        ``"right"``, ``"straight"``, or ``"left"``.
    """
    inc_outward = inc_heading + np.pi
    angle = spatial.normalize_angle(out_heading - inc_outward)

    if angle > np.pi / 4.0:
        return "left"
    if angle < -np.pi / 4.0:
        return "right"

    return "straight"


def _lane_pairs_for_turn(
    turn: str, incoming_lane_num: int, outgoing_lane_num: int
) -> list[tuple[int, int]]:
    """Return ``(incoming_lane_idx, outgoing_lane_idx)`` pairs for a turn type.

    Right turns connect only the outermost lane pair; left turns connect only
    the innermost; straight turns connect all matching lane pairs.

    Args:
        turn: ``"right"``, ``"straight"``, or ``"left"``.
        incoming_lane_num: Number of lanes in the incoming arm.
        outgoing_lane_num: Number of lanes in the outgoing arm.

    Returns:
        List of ``(inc_idx, out_idx)`` index pairs.
    """
    n = min(incoming_lane_num, outgoing_lane_num)

    if n <= 0:
        return []

    if turn == "right":
        return [(0, 0)]

    if turn == "straight":
        return [(i, i) for i in range(n)]

    return [(n - 1, n - 1)]


def _junction_shape_and_corners(
    normalized_arms: list[dict[str, Any]],
    arm_boundary_lines: list[list[np.ndarray]],
    step_size: float,
    id_counter: int,
) -> tuple[list[tuple[float, float]], list[RoadLine], int]:
    """Build the junction pavement polygon and corner road-border RoadLines.

    A single pass over adjacent arm pairs (sorted by inward heading) accumulates
    the fill polygon vertices and the visible solid-white corner curves.

    Args:
        normalized_arms: Arm records as produced by
            :func:`_normalize_intersection_arm`.
        arm_boundary_lines: Lateral boundary point lists, one per arm.
        step_size: Bezier corner sampling interval in metres.
        id_counter: Starting id counter.

    Returns:
        Tuple ``(shape_pts, corner_roadlines, updated_id_counter)``.
    """
    arm_data = list(zip(normalized_arms, arm_boundary_lines))
    arm_data.sort(key=lambda item: item[0]["heading_inward"])

    shape_pts: list[tuple[float, float]] = []
    corner_roadlines: list[RoadLine] = []
    arm_num = len(arm_data)

    for i in range(arm_num):
        curr_arm, curr_boundary = arm_data[i]
        next_arm, next_boundary = arm_data[(i + 1) % arm_num]

        for pt in curr_boundary:
            shape_pts.append((float(pt[0]), float(pt[1])))

        p0 = curr_boundary[-1]
        h0 = float(curr_arm["heading_inward"])
        p3 = next_boundary[0]
        h3 = float(next_arm["heading_inward"] + np.pi)

        corner_pts = bezier_connection(p0, h0, p3, h3, step_size, min_tangent=0.0)

        if len(corner_pts) > 2:
            for pt in corner_pts[1:-1]:
                shape_pts.append((float(pt[0]), float(pt[1])))

        if len(corner_pts) >= 2:
            corner_roadlines.append(
                build_roadline_from_points(
                    id_=id_counter,
                    points=corner_pts,
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

    if shape_pts:
        shape_pts.append(shape_pts[0])

    return shape_pts, corner_roadlines, id_counter


def _intersection_mark_kwargs(tags: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return virtual RoadLine kwargs for an intersection connection boundary.

    Internal intersection connection boundaries are required by ``Lane.line_ids``
    but must not be rendered as visible markings inside the junction.  The kwargs
    are built through :func:`roadline_render_kwargs` so that all standard metadata
    fields (``marking_standard``, ``opendrive_*``, etc.) are populated consistently
    with every other road-line builder.

    Args:
        tags: Optional extra tags merged into ``custom_tags`` with higher priority
            than the built-in marking metadata.

    Returns:
        Dict with keys ``type_``, ``subtype``, ``color``, and ``custom_tags``
        suitable for passing directly to :class:`~tactics2d.map.element.RoadLine`.
    """
    extra = {"marking_token": "intersection_connection"}
    if tags is not None:
        extra.update(tags)
    return merge_marking_kwargs(roadline_render_kwargs("virtual"), extra)


def _intersection_build(
    center: np.ndarray,
    arms: list[dict[str, Any] | RoadPort],
    lane_width: float,
    radius: float,
    speed_limit: float,
    step_size: float,
    id_offset: int,
) -> RoadModuleResult:
    """Build all lanes, roadlines, and ports for a T-junction or cross-junction.

    This function is the single implementation backing
    :class:`Intersection`.  It normalises arm descriptors, generates
    connection lanes for every valid turn combination, closes the junction
    perimeter with Bezier corner curves, and emits named ports.

    Args:
        center: Junction centre in world coordinates.
        arms: 3 or 4 arm descriptors; each element is a :class:`RoadPort`
            or a ``dict`` with keys ``"heading"``, ``"lane_num"``, and
            optionally ``"radius"``, ``"curvature"``, ``"lane_width"``,
            ``"speed_limit"``.
        lane_width: Default lane width in metres for dict arms that omit it.
        radius: Default arm boundary radius in metres.
        speed_limit: Default speed limit in km/h.
        step_size: Arc sampling interval in metres.
        id_offset: Starting id counter.

    Returns:
        :class:`RoadModuleResult` with ports ``"arm_{i}_in"`` and
        ``"arm_{i}_out"`` for each arm index and a single junction element.

    Raises:
        ValueError: If ``arms`` does not contain exactly 3 or 4 elements,
            ``step_size <= 0``, or any arm has an invalid lane count or width.
    """
    center = np.asarray(center, dtype=float)

    if len(arms) not in (3, 4):
        raise ValueError("intersection requires 3 or 4 arms.")

    if step_size <= 0.0:
        raise ValueError("step_size must be positive.")

    normalized_arms = [
        _normalize_intersection_arm(
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
                center_pts = bezier_connection(p0, h_in, p3, h_out, step_size, min_tangent=0.0)

                if len(center_pts) < 2:
                    continue

                lane_w = min(float(inc_arm["lane_width"]), float(out_arm["lane_width"]))
                lane_speed = min(
                    float(inc_arm["speed_limit"]), float(out_arm["speed_limit"]), float(speed_limit)
                )

                left_pts = polyline.offset(center_pts, lane_w / 2.0)
                right_pts = polyline.offset(center_pts, -lane_w / 2.0)

                left_roadline = build_roadline_from_points(
                    id_=id_counter,
                    points=left_pts,
                    marking_kwargs=_intersection_mark_kwargs(
                        {"from_arm": inc_idx, "to_arm": out_idx, "turn": turn, "side": "left"}
                    ),
                )
                id_counter += 1

                right_roadline = build_roadline_from_points(
                    id_=id_counter,
                    points=right_pts,
                    marking_kwargs=_intersection_mark_kwargs(
                        {"from_arm": inc_idx, "to_arm": out_idx, "turn": turn, "side": "right"}
                    ),
                )
                id_counter += 1

                lane = build_lane_from_boundaries(
                    id_=id_counter,
                    left_points=left_pts,
                    right_points=right_pts,
                    left_roadline_ids=left_roadline.id_,
                    right_roadline_ids=right_roadline.id_,
                    speed_limit=lane_speed,
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

    for group_lanes in connection_groups.values():
        add_ordered_lane_neighbors(group_lanes)

    junction_shape, corner_roadlines, id_counter = _junction_shape_and_corners(
        normalized_arms=normalized_arms,
        arm_boundary_lines=arm_boundary_lines,
        step_size=step_size,
        id_counter=id_counter,
    )
    roadlines.extend(corner_roadlines)

    junction = build_pavement_junction(
        id_=id_counter,
        shape_points=junction_shape,
        center=center,
        junction_type="intersection",
        sumo_type="priority",
    )
    id_counter += 1

    ports = build_junction_arm_ports(
        normalized_arms=normalized_arms,
        outgoing_lane_ids=outgoing_lane_ids,
        incoming_lane_ids=incoming_lane_ids,
        module_name="intersection",
    )

    return RoadModuleResult(
        lanes=lanes, roadlines=roadlines, junctions=[junction], ports=ports, id_counter=id_counter
    )


class Intersection(RoadSegment):
    """T-junction or cross-junction generator.

    Attributes:
        lane_width: Default lane width in metres for arm dicts that omit it.
        radius: Default arm boundary radius in metres for arm dicts that omit it.
        speed_limit: Default speed limit in km/h.
        step_size: Arc sampling interval in metres.
    """

    def __init__(
        self,
        lane_width: float = 3.5,
        radius: float = 8.0,
        speed_limit: float = 30.0,
        step_size: float = 0.1,
    ) -> None:
        """Initialise the generator.

        Args:
            lane_width: Default lane width in metres.
            radius: Default arm boundary radius in metres.
            speed_limit: Default speed limit in km/h.
            step_size: Arc sampling interval in metres.

        Raises:
            ValueError: If ``step_size <= 0``, ``lane_width <= 0``, or
                ``radius <= 0``.
        """
        if lane_width <= 0.0:
            raise ValueError("lane_width must be positive.")
        if radius <= 0.0:
            raise ValueError("radius must be positive.")
        if step_size <= 0.0:
            raise ValueError("step_size must be positive.")
        self.lane_width = lane_width
        self.radius = radius
        self.speed_limit = speed_limit
        self.step_size = step_size

    def build(
        self, center: np.ndarray, arms: list[dict[str, Any] | RoadPort], *, id_offset: int = 0
    ) -> RoadModuleResult:
        """Build a T-junction or cross-junction.

        Args:
            center: Junction centre point as a 2-D world coordinate.
            arms: List of arm descriptors. Each element is either a
                :class:`RoadPort` or a plain ``dict`` with keys ``"heading"``,
                ``"lane_num"``, and optionally ``"radius"``, ``"curvature"``,
                ``"lane_width"``, ``"speed_limit"``.
            id_offset: First element id.

        Returns:
            :class:`RoadModuleResult` with ports ``"arm_{i}_in"`` and
            ``"arm_{i}_out"`` for each arm index and a single junction.

        Raises:
            ValueError: If ``arms`` does not contain exactly 3 or 4 elements,
                or any arm has ``lane_num < 1`` or ``lane_width <= 0``.
        """
        return _intersection_build(
            center=center,
            arms=arms,
            lane_width=self.lane_width,
            radius=self.radius,
            speed_limit=self.speed_limit,
            step_size=self.step_size,
            id_offset=id_offset,
        )
