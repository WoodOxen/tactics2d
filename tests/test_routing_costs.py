# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for routing cost builders and lane-change filtering."""

import numpy as np
from shapely.geometry import LineString

from tactics2d.map.element import Lane, LaneRelationship, Map, RoadLine
from tactics2d.routing import (
    AveragedLengthCostBuilder,
    CostBuilder,
    DistanceCostBuilder,
    NodeEdgeCostBuilder,
    TravelTimeCostBuilder,
    build_apollo_inspired_cost,
    build_cost_builder,
    build_distance_cost,
    build_lanelet2_distance_cost,
    build_time_cost,
)
from tactics2d.routing.graph_builder import GraphBuilder


def _build_lane(
    lane_id: str,
    x_left: float,
    x_right: float,
    y_start: float,
    y_end: float,
    *,
    speed_limit: float,
    speed_limit_unit: str = "km/h",
    custom_tags=None,
) -> Lane:
    left_side = LineString([(x_left, y_start), (x_left, y_end)])
    right_side = LineString([(x_right, y_start), (x_right, y_end)])
    return Lane(
        id_=lane_id,
        left_side=left_side,
        right_side=right_side,
        speed_limit=speed_limit,
        speed_limit_unit=speed_limit_unit,
        custom_tags=custom_tags
        if custom_tags is not None
        else {
            "centerline": np.array(
                [
                    [(x_left + x_right) / 2.0, y_start],
                    [(x_left + x_right) / 2.0, y_end],
                ]
            )
        },
    )


def _build_cost_test_map(
    *,
    a_speed_limit: float = 36.0,
    b_speed_limit: float = 54.0,
    c_speed_limit: float = 36.0,
) -> Map:
    map_ = Map(name="routing_cost_test_map")

    lane_a = _build_lane("A", 0.0, 1.0, 0.0, 10.0, speed_limit=a_speed_limit)
    lane_b = _build_lane("B", 0.0, 1.0, 10.0, 40.0, speed_limit=b_speed_limit)
    lane_c = _build_lane(
        "C",
        1.0,
        2.0,
        0.0,
        5.0,
        speed_limit=c_speed_limit,
        custom_tags={
            "centerline": np.array([[1.5, 0.0], [1.5, 5.0]]),
            "turn": "left",
        },
    )

    lane_a.add_related_lane("B", LaneRelationship.SUCCESSOR)
    lane_a.add_related_lane("C", LaneRelationship.RIGHT_NEIGHBOR)

    for lane in [lane_a, lane_b, lane_c]:
        map_.add_lane(lane)

    return map_


def _build_lane_change_filtered_map(*, multi_segment: bool = False) -> Map:
    map_ = Map(name="routing_lane_change_filter_map")

    lane_a = _build_lane("A", 0.0, 1.0, 0.0, 10.0, speed_limit=36.0)
    lane_b = _build_lane("B", 0.0, 1.0, 10.0, 20.0, speed_limit=36.0)
    lane_c = _build_lane("C", 1.0, 2.0, 0.0, 10.0, speed_limit=36.0)
    lane_d = _build_lane("D", -1.0, 0.0, 0.0, 10.0, speed_limit=36.0)

    lane_a.add_related_lane("B", LaneRelationship.SUCCESSOR)
    lane_a.add_related_lane("C", LaneRelationship.RIGHT_NEIGHBOR)
    lane_a.add_related_lane("D", LaneRelationship.LEFT_NEIGHBOR)

    if multi_segment:
        lane_a.line_ids = {
            "left": ["left_allow", "left_block"],
            "right": ["right_allow_1", "right_allow_2"],
        }
    else:
        lane_a.line_ids = {"left": ["left_block"], "right": ["right_allow"]}

    map_.add_roadline(
        RoadLine(
            id_="left_block",
            geometry=LineString([(0.0, 0.0), (0.0, 10.0)]),
            lane_change=(False, False),
        )
    )
    map_.add_roadline(
        RoadLine(
            id_="right_allow",
            geometry=LineString([(1.0, 0.0), (1.0, 10.0)]),
            lane_change=(True, False),
        )
    )
    map_.add_roadline(
        RoadLine(
            id_="left_allow",
            geometry=LineString([(0.0, 0.0), (0.0, 5.0)]),
            lane_change=(False, True),
        )
    )
    map_.add_roadline(
        RoadLine(
            id_="right_allow_1",
            geometry=LineString([(1.0, 0.0), (1.0, 5.0)]),
            lane_change=(True, False),
        )
    )
    map_.add_roadline(
        RoadLine(
            id_="right_allow_2",
            geometry=LineString([(1.0, 5.0), (1.0, 10.0)]),
            lane_change=(True, False),
        )
    )

    for lane in [lane_a, lane_b, lane_c, lane_d]:
        map_.add_lane(lane)

    return map_


def test_distance_cost_mode_uses_lane_length_and_lane_change_penalty():
    map_ = _build_cost_test_map()
    graph = GraphBuilder(
        include_neighbors=True,
        cost_mode="distance",
        cost_kwargs={"lane_change_penalty": 4.0},
    ).build(map_)

    a_idx = graph.lane_id_to_index["A"]
    b_idx = graph.lane_id_to_index["B"]
    c_idx = graph.lane_id_to_index["C"]

    assert graph.csr_graph[a_idx, b_idx] == 30.0
    assert graph.csr_graph[a_idx, c_idx] == 9.0


def test_time_cost_mode_uses_lane_travel_time():
    map_ = _build_cost_test_map(
        a_speed_limit=36.0,
        b_speed_limit=54.0,
        c_speed_limit=36.0,
    )
    graph = GraphBuilder(
        include_neighbors=True,
        cost_mode="time",
        cost_kwargs={"lane_change_penalty": 4.0},
    ).build(map_)

    a_idx = graph.lane_id_to_index["A"]
    b_idx = graph.lane_id_to_index["B"]
    c_idx = graph.lane_id_to_index["C"]

    assert graph.csr_graph[a_idx, b_idx] == 2.0
    assert graph.csr_graph[a_idx, c_idx] == 4.5


def test_lanelet2_cost_modes_follow_relation_aware_rules():
    map_ = _build_cost_test_map(
        a_speed_limit=36.0,
        b_speed_limit=54.0,
        c_speed_limit=36.0,
    )
    distance_graph = GraphBuilder(
        include_neighbors=True,
        cost_mode="lanelet2_distance",
        cost_kwargs={"lane_change_cost": 7.0},
    ).build(map_)
    time_graph = GraphBuilder(
        include_neighbors=True,
        cost_mode="lanelet2_time",
        cost_kwargs={"lane_change_cost": 7.0},
    ).build(map_)

    a_idx = distance_graph.lane_id_to_index["A"]
    b_idx = distance_graph.lane_id_to_index["B"]
    c_idx = distance_graph.lane_id_to_index["C"]

    assert distance_graph.csr_graph[a_idx, b_idx] == 20.0
    assert distance_graph.csr_graph[a_idx, c_idx] == 7.0
    assert time_graph.csr_graph[a_idx, b_idx] == 1.5
    assert time_graph.csr_graph[a_idx, c_idx] == 7.0


def test_router_level_lane_change_penalty_alias_maps_to_builtin_costs():
    map_ = _build_cost_test_map(
        a_speed_limit=36.0,
        b_speed_limit=54.0,
        c_speed_limit=36.0,
    )
    lanelet2_graph = GraphBuilder(
        include_neighbors=True,
        lane_change_penalty=9.0,
        cost_mode="lanelet2_time",
    ).build(map_)
    apollo_graph = GraphBuilder(
        include_neighbors=True,
        lane_change_penalty=11.0,
        cost_mode="apollo_inspired",
        cost_kwargs={
            "base_speed_mps": 10.0,
            "base_changing_length": 5.0,
            "left_turn_penalty": 3.0,
        },
    ).build(map_)

    a_idx = lanelet2_graph.lane_id_to_index["A"]
    c_idx = lanelet2_graph.lane_id_to_index["C"]

    assert lanelet2_graph.csr_graph[a_idx, c_idx] == 9.0
    assert apollo_graph.csr_graph[a_idx, c_idx] == 19.0


def test_apollo_inspired_and_custom_costs_can_be_switched():
    map_ = _build_cost_test_map(
        a_speed_limit=36.0,
        b_speed_limit=36.0,
        c_speed_limit=36.0,
    )
    apollo_graph = GraphBuilder(
        include_neighbors=True,
        cost_mode="apollo_inspired",
        cost_kwargs={
            "base_speed_mps": 10.0,
            "change_penalty": 11.0,
            "base_changing_length": 5.0,
            "left_turn_penalty": 3.0,
        },
    ).build(map_)

    a_idx = apollo_graph.lane_id_to_index["A"]
    b_idx = apollo_graph.lane_id_to_index["B"]
    c_idx = apollo_graph.lane_id_to_index["C"]

    assert apollo_graph.csr_graph[a_idx, b_idx] == 30.0
    assert apollo_graph.csr_graph[a_idx, c_idx] == 19.0

    custom_graph = GraphBuilder(
        include_neighbors=True,
        cost_fn=lambda _map, _from_lane, _to_lane, relation: 42.0
        if relation == "successor"
        else 99.0,
    ).build(map_)
    assert custom_graph.csr_graph[a_idx, b_idx] == 42.0
    assert custom_graph.csr_graph[a_idx, c_idx] == 99.0


def test_neighbor_edges_respect_boundary_lane_change_rules():
    map_ = _build_lane_change_filtered_map()
    graph = GraphBuilder(include_neighbors=True, cost_mode="distance").build(map_)

    a_idx = graph.lane_id_to_index["A"]
    c_idx = graph.lane_id_to_index["C"]
    d_idx = graph.lane_id_to_index["D"]

    assert graph.csr_graph[a_idx, c_idx] > 0.0
    assert graph.csr_graph[a_idx, d_idx] == 0.0


def test_multi_segment_boundary_uses_conservative_all_allowed_policy():
    map_ = _build_lane_change_filtered_map(multi_segment=True)
    graph = GraphBuilder(include_neighbors=True, cost_mode="distance").build(map_)

    a_idx = graph.lane_id_to_index["A"]
    c_idx = graph.lane_id_to_index["C"]
    d_idx = graph.lane_id_to_index["D"]

    assert graph.csr_graph[a_idx, c_idx] > 0.0
    assert graph.csr_graph[a_idx, d_idx] == 0.0


def test_cost_builder_factory_exposes_mechanism_oriented_builders():
    assert isinstance(build_cost_builder("distance"), DistanceCostBuilder)
    assert isinstance(build_cost_builder("time"), TravelTimeCostBuilder)
    assert isinstance(build_cost_builder("lanelet2_distance"), AveragedLengthCostBuilder)
    assert isinstance(build_cost_builder("apollo_inspired"), NodeEdgeCostBuilder)
    assert issubclass(DistanceCostBuilder, CostBuilder)

    try:
        build_cost_builder("unsupported_mode")
    except ValueError as exc:
        assert "Unsupported cost mode" in str(exc)
    else:
        raise AssertionError("build_cost_builder should reject unsupported modes.")


def test_backward_compatible_cost_helpers_still_build_callable_costs():
    map_ = _build_cost_test_map()
    lane_a = map_.lanes["A"]
    lane_b = map_.lanes["B"]
    lane_c = map_.lanes["C"]

    assert build_distance_cost(lane_change_penalty=4.0)(map_, lane_a, lane_b, "successor") == 30.0
    assert build_time_cost(lane_change_penalty=4.0)(map_, lane_a, lane_c, "neighbor") == 4.5
    assert build_lanelet2_distance_cost(lane_change_cost=7.0)(
        map_, lane_a, lane_c, "neighbor"
    ) == 7.0
    assert build_apollo_inspired_cost(
        base_speed_mps=10.0,
        change_penalty=11.0,
        base_changing_length=5.0,
        left_turn_penalty=3.0,
    )(map_, lane_a, lane_c, "neighbor") == 19.0
