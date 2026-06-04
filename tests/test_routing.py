# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for lane-level routing."""

import numpy as np
from shapely.geometry import LineString

from tactics2d.map.element import Lane, LaneRelationship, Map
from tactics2d.routing import Router


def _build_lane(lane_id: str, x_left: float, x_right: float, y_start: float, y_end: float) -> Lane:
    left_side = LineString([(x_left, y_start), (x_left, y_end)])
    right_side = LineString([(x_right, y_start), (x_right, y_end)])
    return Lane(
        id_=lane_id,
        left_side=left_side,
        right_side=right_side,
        custom_tags={
            "centerline": np.array(
                [[(x_left + x_right) / 2.0, y_start], [(x_left + x_right) / 2.0, y_end]]
            )
        },
    )


def _build_test_map() -> Map:
    map_ = Map(name="routing_test_map")

    lane_a = _build_lane("A", 0.0, 1.0, 0.0, 10.0)
    lane_b = _build_lane("B", 0.0, 1.0, 10.0, 40.0)
    lane_c = _build_lane("C", 1.0, 2.0, 0.0, 5.0)
    lane_d = _build_lane("D", 1.0, 2.0, 5.0, 10.0)
    lane_e = _build_lane("E", 0.5, 1.5, 40.0, 50.0)

    lane_a.add_related_lane("B", LaneRelationship.SUCCESSOR)
    lane_a.add_related_lane("C", LaneRelationship.RIGHT_NEIGHBOR)
    lane_c.add_related_lane("D", LaneRelationship.SUCCESSOR)
    lane_b.add_related_lane("E", LaneRelationship.SUCCESSOR)
    lane_d.add_related_lane("E", LaneRelationship.SUCCESSOR)

    for lane in [lane_a, lane_b, lane_c, lane_d, lane_e]:
        map_.add_lane(lane)

    return map_


def test_router_returns_successor_route():
    map_ = _build_test_map()
    router = Router(algorithm="dijkstra", include_neighbors=False)

    route = router.plan(map_, start=(0.5, 1.0), goal=(1.0, 49.0))

    assert not route.is_empty
    assert route.start_lane_id == "A"
    assert route.goal_lane_id == "E"
    assert route.lane_ids == ["A", "B", "E"]
    assert route.total_cost > 0.0
    assert route.path is not None


def test_router_lane_change_penalty_changes_route_choice():
    map_ = _build_test_map()

    low_penalty_router = Router(
        algorithm="dijkstra", include_neighbors=True, lane_change_penalty=4.0
    )
    low_penalty_route = low_penalty_router.plan(map_, start=(0.5, 1.0), goal=(1.0, 49.0))
    assert low_penalty_route.lane_ids == ["A", "C", "D", "E"]

    high_penalty_router = Router(
        algorithm="dijkstra", include_neighbors=True, lane_change_penalty=40.0
    )
    high_penalty_route = high_penalty_router.plan(map_, start=(0.5, 1.0), goal=(1.0, 49.0))
    assert high_penalty_route.lane_ids == ["A", "B", "E"]


def test_router_supports_all_builtin_cost_modes_and_custom_callbacks():
    map_ = _build_test_map()
    start = (0.5, 1.0)
    goal = (1.0, 49.0)

    builtin_modes = [
        ("distance", {"lane_change_penalty": 4.0}),
        ("time", {"lane_change_penalty": 4.0}),
        ("lanelet2_distance", {"lane_change_penalty": 4.0}),
        ("lanelet2_time", {"lane_change_penalty": 4.0}),
        ("apollo_inspired", {"lane_change_penalty": 4.0}),
    ]

    for cost_mode, kwargs in builtin_modes:
        router = Router(algorithm="dijkstra", include_neighbors=True, cost_mode=cost_mode, **kwargs)
        route = router.plan(map_, start=start, goal=goal)
        assert not route.is_empty
        assert route.start_lane_id == "A"
        assert route.goal_lane_id == "E"
        assert route.total_cost > 0.0

    custom_router = Router(
        algorithm="dijkstra",
        include_neighbors=True,
        cost_mode="distance",
        cost_fn=lambda _map, _from_lane, _to_lane, relation: (
            1.0 if relation == "successor" else 10.0
        ),
    )
    custom_route = custom_router.plan(map_, start=start, goal=goal)
    assert custom_route.lane_ids == ["A", "B", "E"]


def test_router_rejects_unknown_cost_mode():
    map_ = _build_test_map()
    router = Router(algorithm="dijkstra", include_neighbors=True, cost_mode="not_a_real_cost_mode")

    try:
        router.plan(map_, start=(0.5, 1.0), goal=(1.0, 49.0))
    except ValueError as exc:
        assert "Unsupported cost mode" in str(exc)
    else:
        raise AssertionError("Router should reject unsupported cost modes.")
