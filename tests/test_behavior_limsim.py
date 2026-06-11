# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for LimSim-style interactive behavior planning."""

import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon

from tactics2d.behavior import BehaviorModelBase, LimSimBehaviorModel
from tactics2d.behavior.limsim.action import LimSimAction
from tactics2d.behavior.limsim.config import LimSimConfig
from tactics2d.behavior.limsim.evaluation import (
    dimensions_from_participants,
    evaluate_planning_result,
    evaluate_rolling_result,
)
from tactics2d.behavior.limsim.frenet_planner import (
    FrenetTrajectoryPlanner,
    build_reference_path_from_agent,
)
from tactics2d.geometry import ReferencePath
from tactics2d.behavior.limsim.interaction import InteractionGraph, has_trajectory_collision
from tactics2d.behavior.limsim.planner import LaneFollower
from tactics2d.behavior.limsim.prediction import LimSimPredictor
from tactics2d.behavior.limsim.roi import RoISelector
from tactics2d.behavior.limsim.rolling import LimSimRollingRunner
from tactics2d.behavior.limsim.scene import SceneBuilder
from tactics2d.behavior.limsim.schema import states_to_trajectory
from tactics2d.map.element import Junction, Lane, LaneRelationship, Map, Regulatory, RoadLine
from tactics2d.map.query import SemanticMapQuery
from tactics2d.participant.element import Pedestrian, Vehicle
from tactics2d.participant.trajectory import State, Trajectory


def _build_lane(lane_id, x_left, x_right, y_start, y_end):
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


def _build_parallel_map():
    map_ = Map(name="limsim_test_map")
    lane_a = _build_lane("A", 0.0, 2.0, 0.0, 80.0)
    lane_b = _build_lane("B", 2.0, 4.0, 0.0, 80.0)
    lane_a.add_related_lane("B", LaneRelationship.RIGHT_NEIGHBOR)
    lane_b.add_related_lane("A", LaneRelationship.LEFT_NEIGHBOR)
    map_.add_lane(lane_a)
    map_.add_lane(lane_b)
    return map_


def _build_successor_map():
    map_ = Map(name="limsim_successor_map")
    lane_a = _build_lane("A", 0.0, 2.0, 0.0, 30.0)
    lane_b = _build_lane("B", 0.0, 2.0, 30.0, 80.0)
    lane_a.add_related_lane("B", LaneRelationship.SUCCESSOR)
    map_.add_lane(lane_a)
    map_.add_lane(lane_b)
    return map_


def _build_crossing_map():
    map_ = Map(name="crossing_map")
    east_west = Lane(
        id_="EW",
        left_side=LineString([(-10.0, 1.0), (10.0, 1.0)]),
        right_side=LineString([(-10.0, -1.0), (10.0, -1.0)]),
        custom_tags={"centerline": np.array([[-10.0, 0.0], [10.0, 0.0]])},
    )
    north_south = Lane(
        id_="NS",
        left_side=LineString([(-1.0, -10.0), (-1.0, 10.0)]),
        right_side=LineString([(1.0, -10.0), (1.0, 10.0)]),
        custom_tags={"centerline": np.array([[0.0, -10.0], [0.0, 10.0]])},
    )
    map_.add_lane(east_west)
    map_.add_lane(north_south)
    return map_


def _build_semantic_map():
    map_ = _build_parallel_map()
    map_.add_roadline(
        RoadLine(id_="solid_left", geometry=LineString([(0.0, 0.0), (0.0, 80.0)]), subtype="solid")
    )
    map_.add_roadline(
        RoadLine(
            id_="dashed_right", geometry=LineString([(2.0, 0.0), (2.0, 80.0)]), subtype="dashed"
        )
    )
    lane = map_.lanes["A"]
    lane.add_related_lane("B", LaneRelationship.LEFT_NEIGHBOR)
    lane.line_ids = {"left": ["solid_left"], "right": ["dashed_right"]}
    lane.custom_tags = {
        **(lane.custom_tags or {}),
        "boundary_segments": {
            "left": [
                {
                    "roadline_id": "solid_left",
                    "lane_start_index": 0,
                    "lane_end_index": 1,
                    "start_s": 0.0,
                    "end_s": 80.0,
                }
            ],
            "right": [
                {
                    "roadline_id": "dashed_right",
                    "lane_start_index": 0,
                    "lane_end_index": 1,
                    "start_s": 0.0,
                    "end_s": 80.0,
                }
            ],
        },
    }
    map_.add_regulatory(
        Regulatory(
            id_="stop_A", ways={"A": "refers"}, subtype="stop_sign", position=Point(1.0, 20.0)
        )
    )
    map_.add_regulatory(
        Regulatory(
            id_="traffic_light_A",
            ways={"A": "refers"},
            subtype="traffic_light",
            dynamic=True,
            position=Point(1.0, 30.0),
            custom_tags={
                "lane_id": "A",
                "states": [
                    {"time_ms": 0, "state": "LANE_STATE_STOP", "stop_point": [1.0, 30.0]},
                    {"time_ms": 100, "state": "LANE_STATE_GO", "stop_point": [1.0, 31.0]},
                ],
            },
        )
    )
    map_.add_junction(
        Junction(
            id_="J0",
            custom_tags={
                "shape": [(0.0, 18.0), (4.0, 18.0), (4.0, 24.0), (0.0, 24.0)],
                "inside_lanes": ["A"],
            },
        )
    )
    return map_


def _vehicle(agent_id, frame, x, y, heading=np.pi / 2, speed=5.0):
    trajectory = Trajectory(id_=agent_id, fps=10, stable_freq=True)
    trajectory.add_state(
        State(
            frame=frame,
            x=x,
            y=y,
            heading=heading,
            vx=speed * np.cos(heading),
            vy=speed * np.sin(heading),
        )
    )
    return Vehicle(agent_id, "vehicle", trajectory=trajectory, length=4.5, width=1.8)


def _pedestrian(agent_id, frame, x, y, heading=np.pi / 2, speed=1.0):
    trajectory = Trajectory(id_=agent_id, fps=10, stable_freq=True)
    trajectory.add_state(
        State(
            frame=frame,
            x=x,
            y=y,
            heading=heading,
            vx=speed * np.cos(heading),
            vy=speed * np.sin(heading),
        )
    )
    return Pedestrian(agent_id, "adult_male", trajectory=trajectory, width=0.4)


def test_scene_builder_matches_lanes_and_builds_agent_states():
    map_ = _build_parallel_map()
    participants = {1: _vehicle(1, 0, 1.0, 5.0), 2: _vehicle(2, 0, 3.0, 8.0)}
    builder = SceneBuilder(LimSimConfig())

    states = builder.build(participants, map_, frame=0)

    assert set(states) == {1, 2}
    assert states[1].lane_id == "A"
    assert states[2].lane_id == "B"
    assert states[1].speed == pytest.approx(5.0)


def test_scene_builder_prefers_heading_consistent_lane_over_nearest_crossing_lane():
    map_ = Map(name="heading_match_map")
    northbound = Lane(
        id_="N",
        left_side=LineString([(-1.0, -10.0), (-1.0, 10.0)]),
        right_side=LineString([(1.0, -10.0), (1.0, 10.0)]),
        custom_tags={"centerline": np.array([[0.0, -10.0], [0.0, 10.0]])},
    )
    crossing = Lane(
        id_="X",
        left_side=LineString([(-10.0, 0.3), (10.0, 0.3)]),
        right_side=LineString([(-10.0, -0.7), (10.0, -0.7)]),
        custom_tags={"centerline": np.array([[-10.0, -0.2], [10.0, -0.2]])},
    )
    map_.add_lane(northbound)
    map_.add_lane(crossing)
    participant = _vehicle(1, 0, 0.0, 0.1, heading=np.pi / 2, speed=5.0)

    states = SceneBuilder(LimSimConfig()).build({1: participant}, map_, frame=0)

    assert states[1].lane_id == "N"


def test_lane_geometry_helpers_project_points_and_width():
    lane = _build_lane("A", 0.0, 2.0, 0.0, 80.0)

    assert lane.get_width() == pytest.approx(2.0)

    projection = lane.project_point((1.5, 10.0))
    assert projection is not None
    assert projection.s == pytest.approx(10.0)
    assert projection.d == pytest.approx(-0.5)
    assert projection.distance == pytest.approx(0.5)


def test_reference_path_converts_cartesian_and_frenet_coordinates():
    reference_path = ReferencePath(LineString([(0.0, 0.0), (10.0, 0.0)]), ("A",), 2.0)

    frenet = reference_path.cartesian_to_frenet(4.0, 1.0)
    x, y, heading = reference_path.frenet_to_cartesian(frenet.s, frenet.d)

    assert frenet.s == pytest.approx(4.0)
    assert frenet.d == pytest.approx(1.0)
    assert x == pytest.approx(4.0)
    assert y == pytest.approx(1.0)
    assert heading == pytest.approx(0.0)


def test_map_get_speed_limit_by_lane_id():
    map_ = Map(name="speed_limit_map")
    lane = _build_lane("A", 0.0, 2.0, 0.0, 80.0)
    lane.speed_limit = 13.4
    map_.add_lane(lane)

    assert map_.get_speed_limit("A") == pytest.approx(13.4)
    assert map_.get_speed_limit("missing", default=8.0) == pytest.approx(8.0)


def test_semantic_map_query_infers_crossing_lane_conflict():
    query = SemanticMapQuery(_build_crossing_map())

    assert query.has_conflict("EW", "NS")
    assert query.get_conflict_lanes("EW") == ["NS"]
    conflicts = query.get_junction_conflicts("EW")
    points = query.get_conflict_points("EW", "NS")
    assert len(conflicts) == 1
    assert conflicts[0].lane_id == "EW"
    assert conflicts[0].conflict_lane_id == "NS"
    assert conflicts[0].inferred
    assert conflicts[0].source == "geometry"
    assert len(conflicts[0].points) == 1
    assert len(points) == 1
    assert points[0].x == pytest.approx(0.0)
    assert points[0].y == pytest.approx(0.0)


def test_semantic_map_query_does_not_treat_neighbors_as_conflict():
    query = SemanticMapQuery(_build_parallel_map())

    assert not query.has_conflict("A", "B")
    assert query.get_conflict_lanes("A") == []


def test_semantic_map_query_builds_reference_path_and_region_lane_query():
    query = SemanticMapQuery(_build_successor_map())

    reference = query.get_reference_path("A", lookahead_lanes=2)

    assert reference is not None
    assert isinstance(reference, ReferencePath)
    assert reference.lane_ids == ("A", "B")
    assert reference.path.length == pytest.approx(80.0)
    assert query.query_lanes_in_region((0.0, 2.0, 20.0, 40.0)) == ["A", "B"]


def test_semantic_map_query_reads_stop_and_traffic_light_targets():
    query = SemanticMapQuery(_build_semantic_map())

    traffic_light = query.get_traffic_light_for_lane("A")
    traffic_state = query.get_traffic_light_state("A", time_ms=90)
    stop_signs = query.get_stop_signs("A")
    stop_targets = query.get_stop_targets("A", time_ms=0)
    stop_line = query.get_stop_line("A", stop_targets[0])
    stop_line_geometry = query.get_stop_line_geometry("A", stop_targets[0])

    assert traffic_light.id_ == "traffic_light_A"
    assert traffic_state["state"] == "LANE_STATE_GO"
    assert [stop_sign.id_ for stop_sign in stop_signs] == ["stop_A"]
    assert {target.reason for target in stop_targets} == {"stop_sign", "traffic_light"}
    assert stop_line.virtual
    assert stop_line.source_id == stop_targets[0].source_id
    assert stop_line.reason == stop_targets[0].reason
    assert stop_line.geometry.length == pytest.approx(2.0)
    assert stop_line_geometry.length == pytest.approx(2.0)


def test_semantic_map_query_checks_lane_change_permission_from_roadlines():
    query = SemanticMapQuery(_build_semantic_map())

    assert not query.get_lane_change_permission("A", "left", s=10.0)
    assert query.get_lane_change_permission("A", "right", s=10.0)
    assert not query.get_lane_change_permission("missing", "right")


def test_action_validity_uses_lane_change_permission():
    from tactics2d.behavior.limsim.planner import action_is_valid

    map_ = _build_semantic_map()
    state = SceneBuilder(LimSimConfig()).build({1: _vehicle(1, 0, 1.0, 5.0)}, map_, frame=0)[1]

    assert not action_is_valid(state, LimSimAction.LCL, map_)
    assert action_is_valid(state, LimSimAction.LCR, map_)


def test_semantic_map_query_returns_existing_junction_area():
    query = SemanticMapQuery(_build_semantic_map())

    area = query.get_junction_area(lane_id="A")

    assert isinstance(area, Polygon)
    assert area.area == pytest.approx(24.0)


def test_interaction_graph_groups_nearby_agents_only():
    config = LimSimConfig(interaction_distance=10.0)
    map_ = _build_parallel_map()
    participants = {
        1: _vehicle(1, 0, 1.0, 5.0),
        2: _vehicle(2, 0, 3.0, 8.0),
        3: _vehicle(3, 0, 3.0, 60.0),
    }
    states = SceneBuilder(config).build(participants, map_, frame=0)

    groups = InteractionGraph(config).build_groups(states)

    assert sorted(sorted(group) for group in groups) == [[1, 2], [3]]


def test_roi_selector_splits_inner_and_background_agents_by_radius():
    participants = {
        1: _vehicle(1, 0, 0.0, 0.0),
        2: _vehicle(2, 0, 3.0, 4.0),
        3: _vehicle(3, 0, 8.0, 0.0),
        4: _vehicle(4, 0, 20.0, 0.0),
    }

    selection = RoISelector.select_by_radius(
        participants, frame=0, center=(0.0, 0.0), radius=5.0, outer_radius=10.0
    )

    assert selection.agent_ids == [1, 2]
    assert selection.background_agent_ids == [3]
    assert selection.center == (0.0, 0.0)


def test_roi_selector_selects_around_ego_agent():
    participants = {
        1: _vehicle(1, 0, 0.0, 0.0),
        2: _vehicle(2, 0, 4.0, 0.0),
        3: _vehicle(3, 0, 9.0, 0.0),
    }

    selection = RoISelector.select_around_agent(participants, frame=0, ego_id=1, radius=5.0)

    assert selection.agent_ids == [1, 2]
    assert selection.background_agent_ids == [3]


def test_roi_selector_selects_dense_region():
    participants = {
        1: _vehicle(1, 0, 0.0, 0.0),
        2: _vehicle(2, 0, 1.0, 0.0),
        3: _vehicle(3, 0, 0.0, 1.0),
        4: _vehicle(4, 0, 100.0, 100.0),
    }

    selection = RoISelector.select_dense_region(participants, frame=0, max_agents=3)

    assert set(selection.agent_ids) == {1, 2, 3}


def test_lane_follower_rolls_out_forward_and_lane_change():
    config = LimSimConfig(horizon_steps=5)
    map_ = _build_parallel_map()
    state = SceneBuilder(config).build({1: _vehicle(1, 0, 1.0, 5.0)}, map_, frame=0)[1]
    follower = LaneFollower(config)

    keep_states = follower.rollout(state, LimSimAction.KEEP, map_)
    lane_change_states = follower.rollout(state, LimSimAction.LCR, map_)

    assert keep_states[-1].y > state.y
    assert keep_states[-1].lane_id == "A"
    assert lane_change_states[-1].lane_id == "A"
    assert state.x < lane_change_states[-1].x


def test_non_lane_change_actions_keep_lateral_offset():
    config = LimSimConfig(horizon_steps=5)
    map_ = _build_parallel_map()
    state = SceneBuilder(config).build({1: _vehicle(1, 0, 1.5, 5.0, speed=0.0)}, map_, frame=0)[1]
    follower_states = LaneFollower(config).rollout(state, LimSimAction.KS, map_)
    planner_states = FrenetTrajectoryPlanner(config).plan(state, LimSimAction.KS, map_)

    assert {round(state_.lateral_offset, 6) for state_ in follower_states} == {
        round(state.lateral_offset, 6)
    }
    assert {round(state_.lateral_offset, 6) for state_ in planner_states} == {
        round(state.lateral_offset, 6)
    }


def test_lane_follower_switches_lane_after_lateral_boundary_crossing():
    config = LimSimConfig(horizon_steps=20)
    map_ = _build_parallel_map()
    state = SceneBuilder(config).build({1: _vehicle(1, 0, 1.0, 5.0)}, map_, frame=0)[1]

    lane_change_states = LaneFollower(config).rollout(state, LimSimAction.LCR, map_)

    assert lane_change_states[-1].lane_id == "B"
    assert lane_change_states[-1].x > 2.0


def test_reference_path_converts_between_cartesian_and_frenet():
    config = LimSimConfig()
    map_ = _build_parallel_map()
    state = SceneBuilder(config).build({1: _vehicle(1, 0, 1.5, 5.0)}, map_, frame=0)[1]

    reference_path = build_reference_path_from_agent(state, map_, config)
    frenet = reference_path.cartesian_to_frenet(state.x, state.y)
    x, y, heading = reference_path.frenet_to_cartesian(frenet.s, frenet.d)

    assert frenet.s == pytest.approx(5.0)
    assert x == pytest.approx(state.x)
    assert y == pytest.approx(state.y)
    assert heading == pytest.approx(np.pi / 2)


def test_frenet_planner_generates_lane_change_candidate():
    config = LimSimConfig(horizon_steps=30)
    map_ = _build_parallel_map()
    state = SceneBuilder(config).build({1: _vehicle(1, 0, 1.0, 5.0)}, map_, frame=0)[1]

    planned = FrenetTrajectoryPlanner(config).plan(state, LimSimAction.LCR, map_)

    assert len(planned) == config.horizon_steps
    assert planned[-1].lane_id == "B"
    assert planned[-1].x > 2.5
    assert planned[-1].y > state.y


def test_frenet_planner_filters_illegal_lane_change_candidate():
    config = LimSimConfig(horizon_steps=15)
    map_ = _build_semantic_map()
    state = SceneBuilder(config).build({1: _vehicle(1, 0, 1.0, 5.0)}, map_, frame=0)[1]
    planner = FrenetTrajectoryPlanner(config)
    reference_path = build_reference_path_from_agent(state, map_, config)

    assert planner.sample_candidates(state, LimSimAction.LCL, reference_path, map_) == []
    assert planner.sample_candidates(state, LimSimAction.LCR, reference_path, map_)


def test_frenet_planner_stops_for_stop_target():
    config = LimSimConfig(horizon_steps=40, frenet_stop_line_penalty=10000.0)
    map_ = _build_semantic_map()
    state = SceneBuilder(config).build({1: _vehicle(1, 0, 1.0, 5.0, speed=8.0)}, map_, frame=0)[1]

    trajectory = FrenetTrajectoryPlanner(config).plan(state, LimSimAction.KS, map_)

    assert trajectory[-1].speed <= config.frenet_stop_speed_threshold
    assert trajectory[-1].route_progress <= 20.0


def test_frenet_planner_scores_obstacle_collision_risk():
    config = LimSimConfig(horizon_steps=20, frenet_obstacle_buffer=4.0)
    map_ = _build_parallel_map()
    states = SceneBuilder(config).build(
        {1: _vehicle(1, 0, 1.0, 5.0, speed=8.0), 2: _vehicle(2, 0, 1.0, 14.0, speed=0.0)},
        map_,
        frame=0,
    )
    planner = FrenetTrajectoryPlanner(config)
    obstacle = LaneFollower(config).rollout(states[2], LimSimAction.KS, map_)

    keep_reference = build_reference_path_from_agent(states[1], map_, config)
    candidates = planner.sample_candidates(
        states[1], LimSimAction.DC, keep_reference, map_, obstacle_trajectories=[obstacle]
    )
    chosen = planner.plan(states[1], LimSimAction.DC, map_, obstacle_trajectories=[obstacle])

    assert candidates
    assert min(candidate.cost for candidate in candidates) < max(
        candidate.cost for candidate in candidates
    )
    assert chosen[-1].y < obstacle[-1].y


def test_frenet_planner_adds_junction_conflict_cost():
    config = LimSimConfig(horizon_steps=20)
    map_ = _build_crossing_map()
    planner = FrenetTrajectoryPlanner(config)
    ego = SceneBuilder(config).build(
        {1: _vehicle(1, 0, -6.0, 0.0, heading=0.0, speed=6.0)}, map_, frame=0
    )[1]
    obstacle = SceneBuilder(config).build(
        {2: _vehicle(2, 0, 0.0, -6.0, heading=np.pi / 2, speed=6.0)}, map_, frame=0
    )[2]
    reference_path = build_reference_path_from_agent(ego, map_, config)
    states = planner.sample_candidates(ego, LimSimAction.KS, reference_path, map_)[0].states
    obstacle_states = LaneFollower(config).rollout(obstacle, LimSimAction.KS, map_)

    no_obstacle_cost = planner._cost(states, 6.0, 0.0, 0.0, 0.0, [], reference_path, map_)
    conflict_cost = planner._cost(
        states, 6.0, 0.0, 0.0, 0.0, [obstacle_states], reference_path, map_
    )

    assert conflict_cost > no_obstacle_cost


def test_frenet_planner_stop_fallback_avoids_collision_candidate():
    config = LimSimConfig(horizon_steps=20)
    map_ = _build_parallel_map()
    states = SceneBuilder(config).build(
        {1: _vehicle(1, 0, 1.0, 5.0, speed=8.0), 2: _vehicle(2, 0, 1.0, 13.0, speed=0.0)},
        map_,
        frame=0,
    )
    obstacle = LaneFollower(config).rollout(states[2], LimSimAction.KS, map_)

    planned = FrenetTrajectoryPlanner(config).plan(
        states[1], LimSimAction.AC, map_, obstacle_trajectories=[obstacle]
    )

    assert has_trajectory_collision([planned, obstacle]) is False
    assert planned[-1].speed == pytest.approx(0.0)


def test_limsim_behavior_model_outputs_trajectories_for_group():
    config = LimSimConfig(
        horizon_steps=6,
        dt=0.2,
        mcts_iterations=20,
        interaction_distance=15.0,
    )
    map_ = _build_parallel_map()
    participants = {1: _vehicle(1, 0, 1.0, 5.0, speed=5.0), 2: _vehicle(2, 0, 1.0, 10.0, speed=2.0)}

    result = LimSimBehaviorModel(config).plan(participants, map_, frame=0)

    assert config.planning_steps == 6
    assert config.step_ms == 200
    assert sorted(result.groups[0]) == [1, 2]
    assert set(result.actions) == {1, 2}
    assert set(result.trajectories) == {1, 2}
    assert len(result.trajectories[1].frames) == config.horizon_steps


def test_limsim_behavior_model_predict_returns_shared_trajectory_dict():
    config = LimSimConfig(horizon_steps=4, mcts_iterations=10, interaction_distance=15.0)
    map_ = _build_parallel_map()
    participants = {1: _vehicle(1, 0, 1.0, 5.0, speed=5.0)}
    model = LimSimBehaviorModel(config)

    trajectories = model.predict(participants, map_, frame=0)

    assert isinstance(model, BehaviorModelBase)
    assert set(trajectories) == {1}
    assert len(trajectories[1].frames) == config.horizon_steps


def test_limsim_behavior_model_can_select_roi_around_ego():
    config = LimSimConfig(horizon_steps=6, mcts_iterations=20, interaction_distance=15.0)
    map_ = _build_parallel_map()
    participants = {
        1: _vehicle(1, 0, 1.0, 5.0, speed=5.0),
        2: _vehicle(2, 0, 1.0, 10.0, speed=2.0),
        3: _vehicle(3, 0, 1.0, 28.0, speed=2.0),
        4: _vehicle(4, 0, 1.0, 70.0, speed=2.0),
    }

    result = LimSimBehaviorModel(config).plan(
        participants, map_, frame=0, ego_id=1, roi_radius=12.0
    )

    assert result.roi_agent_ids == [1, 2]
    assert result.background_agent_ids == [3]
    assert set(result.actions) == {1, 2}


def test_limsim_behavior_model_controls_vehicles_only_in_roi():
    config = LimSimConfig(horizon_steps=6, mcts_iterations=20, interaction_distance=15.0)
    map_ = _build_parallel_map()
    participants = {
        1: _vehicle(1, 0, 1.0, 5.0, speed=5.0),
        2: _pedestrian(2, 0, 1.0, 6.0, speed=1.0),
        3: _vehicle(3, 0, 1.0, 14.0, speed=2.0),
        4: _pedestrian(4, 0, 1.0, 18.0, speed=1.0),
    }

    result = LimSimBehaviorModel(config).plan(
        participants, map_, frame=0, ego_id=1, roi_radius=5.0, roi_outer_radius=20.0
    )

    assert result.roi_agent_ids == [1]
    assert result.background_agent_ids == [3]
    assert set(result.actions) == {1}
    assert set(result.trajectories) == {1}


def test_limsim_behavior_model_filters_explicit_non_vehicle_agents():
    config = LimSimConfig(horizon_steps=6, mcts_iterations=20, interaction_distance=15.0)
    map_ = _build_parallel_map()
    participants = {
        1: _vehicle(1, 0, 1.0, 5.0, speed=5.0),
        2: _pedestrian(2, 0, 1.0, 6.0, speed=1.0),
    }

    result = LimSimBehaviorModel(config).plan(participants, map_, frame=0, agent_ids=[1, 2])

    assert result.roi_agent_ids == [1]
    assert result.background_agent_ids == []
    assert set(result.actions) == {1}
    assert set(result.trajectories) == {1}


def test_limsim_behavior_model_filters_non_vehicle_ego_from_controlled_agents():
    config = LimSimConfig(horizon_steps=6, mcts_iterations=20, interaction_distance=15.0)
    map_ = _build_parallel_map()
    participants = {
        1: _pedestrian(1, 0, 1.0, 5.0, speed=1.0),
        2: _vehicle(2, 0, 1.0, 14.0, speed=2.0),
    }

    result = LimSimBehaviorModel(config).plan(
        participants, map_, frame=0, ego_id=1, roi_radius=5.0, roi_outer_radius=20.0
    )

    assert result.roi_agent_ids == []
    assert result.background_agent_ids == [2]
    assert result.actions == {}
    assert result.trajectories == {}


def test_limsim_rolling_runner_advances_controlled_and_background_vehicles():
    config = LimSimConfig(horizon_steps=6, mcts_iterations=20, interaction_distance=12.0)
    map_ = _build_parallel_map()
    participants = {
        1: _vehicle(1, 0, 1.0, 5.0, speed=5.0),
        2: _vehicle(2, 0, 1.0, 20.0, speed=3.0),
        3: _vehicle(3, 0, 3.0, 30.0, speed=4.0),
        4: _pedestrian(4, 0, 1.0, 7.0, speed=1.0),
    }

    rolling = LimSimRollingRunner(config)
    result = rolling.run(
        participants,
        map_,
        start_frame=0,
        simulation_steps=3,
        ego_id=1,
        roi_radius=18.0,
        roi_outer_radius=35.0,
    )

    assert result.frames == [0, 100, 200, 300]
    assert len(result.results) == 3
    assert 4 not in result.participants
    assert result.participants[1].trajectory.has_state(300)
    assert result.participants[3].trajectory.has_state(300)
    assert result.participants[3].trajectory.get_state(300).y > participants[3].trajectory.get_state(0).y
    assert all(set(step_result.actions).issubset(result.participants) for step_result in result.results)
    assert len(result.predicted_trajectories) == 3
    assert set(result.predicted_trajectories[1]).intersection(result.results[0].trajectories)


def test_limsim_rolling_runner_reuses_previous_planned_trajectory():
    config = LimSimConfig(horizon_steps=6, mcts_iterations=20, interaction_distance=12.0)
    map_ = _build_parallel_map()
    participants = {
        1: _vehicle(1, 0, 1.0, 5.0, speed=5.0),
        2: _vehicle(2, 0, 1.0, 14.0, speed=2.0),
    }

    result = LimSimRollingRunner(config).run(
        participants,
        map_,
        start_frame=0,
        simulation_steps=2,
        ego_id=1,
        roi_radius=18.0,
    )

    agent_id = result.results[0].roi_agent_ids[0]
    reused = result.predicted_trajectories[1][agent_id]

    assert reused.first_frame == 200
    assert reused.last_state.location == result.results[0].trajectories[agent_id].last_state.location


def test_limsim_rolling_runner_commits_lane_change_trajectory():
    config = LimSimConfig(horizon_steps=12, mcts_iterations=20, interaction_distance=12.0)
    map_ = _build_parallel_map()
    participants = {1: _vehicle(1, 0, 1.0, 5.0, speed=5.0)}

    result = LimSimBehaviorModel(config).plan(participants, map_, frame=0, agent_ids=[1])
    result.actions[1] = LimSimAction.LCR
    state = SceneBuilder(config).build(participants, map_, frame=0)[1]
    planned_states = LimSimBehaviorModel(config).trajectory_planner.plan(
        state, LimSimAction.LCR, map_
    )
    result.trajectories[1] = states_to_trajectory(1, planned_states, 0, config.dt)

    runner = LimSimRollingRunner(config)
    committed_trajectories = {}
    committed_actions = {}
    runner._update_committed_trajectories(
        result, frame=0, committed_trajectories=committed_trajectories, committed_actions=committed_actions
    )

    assert committed_actions[1] == LimSimAction.LCR
    assert committed_trajectories[1].last_state.location == result.trajectories[1].last_state.location


def test_limsim_rolling_runner_keeps_lane_change_commitment_until_trajectory_end():
    config = LimSimConfig(horizon_steps=12, mcts_iterations=20, interaction_distance=12.0)
    map_ = _build_parallel_map()
    participants = {1: _vehicle(1, 0, 1.0, 5.0, speed=5.0)}
    runner = LimSimRollingRunner(config)
    state = SceneBuilder(config).build(participants, map_, frame=0)[1]
    planned_states = LimSimBehaviorModel(config).trajectory_planner.plan(
        state, LimSimAction.LCR, map_
    )
    trajectory = states_to_trajectory(1, planned_states, 0, config.dt)
    first_result = LimSimBehaviorModel(config).plan(participants, map_, frame=0, agent_ids=[1])
    first_result.actions[1] = LimSimAction.LCR
    first_result.trajectories[1] = trajectory
    committed_trajectories = {}
    committed_actions = {}

    runner._update_committed_trajectories(
        first_result,
        frame=0,
        committed_trajectories=committed_trajectories,
        committed_actions=committed_actions,
    )
    second_result = LimSimBehaviorModel(config).plan(participants, map_, frame=100, agent_ids=[1])
    second_result.actions[1] = LimSimAction.KS
    second_result.trajectories[1] = trajectory

    runner._update_committed_trajectories(
        second_result,
        frame=100,
        committed_trajectories=committed_trajectories,
        committed_actions=committed_actions,
    )

    assert second_result.actions[1] == LimSimAction.LCR
    assert committed_trajectories[1] is trajectory


def test_evaluate_rolling_result_reports_safety_continuity_and_memory_metrics():
    config = LimSimConfig(horizon_steps=6, mcts_iterations=20, interaction_distance=12.0)
    map_ = _build_parallel_map()
    participants = {
        1: _vehicle(1, 0, 1.0, 5.0, speed=5.0),
        2: _vehicle(2, 0, 1.0, 14.0, speed=2.0),
    }

    result = LimSimRollingRunner(config).run(
        participants,
        map_,
        start_frame=0,
        simulation_steps=3,
        ego_id=1,
        roi_radius=18.0,
    )
    evaluation = evaluate_rolling_result(result, dimensions_from_participants(result.participants))

    assert evaluation.action_counts
    assert evaluation.min_distance > 0.0
    assert evaluation.collision_count == 0
    assert evaluation.first_collision is None
    assert evaluation.memory_hit_count > 0
    assert evaluation.roi_sizes == tuple(len(step.roi_agent_ids) for step in result.results)
    assert evaluation.background_sizes == tuple(
        len(step.background_agent_ids) for step in result.results
    )


def test_background_roi_vehicle_influences_single_agent_action():
    config = LimSimConfig(horizon_steps=20, mcts_iterations=30, interaction_distance=10.0)
    map_ = _build_parallel_map()
    participants = {1: _vehicle(1, 0, 1.0, 5.0, speed=8.0), 2: _vehicle(2, 0, 1.0, 22.0, speed=0.0)}

    result = LimSimBehaviorModel(config).plan(
        participants, map_, frame=0, ego_id=1, roi_radius=10.0
    )

    assert result.roi_agent_ids == [1]
    assert result.background_agent_ids == [2]
    assert result.actions[1] == LimSimAction.DC


def test_background_roi_vehicle_influences_decision_search_group_action():
    config = LimSimConfig(horizon_steps=20, mcts_iterations=120, interaction_distance=12.0)
    map_ = _build_parallel_map()
    participants = {
        1: _vehicle(1, 0, 1.0, 5.0, speed=8.0),
        2: _vehicle(2, 0, 3.0, 8.0, speed=8.0),
        3: _vehicle(3, 0, 1.0, 23.0, speed=0.0),
    }

    result = LimSimBehaviorModel(config).plan(
        participants, map_, frame=0, ego_id=1, roi_radius=10.0
    )

    assert set(result.roi_agent_ids) == {1, 2}
    assert result.background_agent_ids == [3]
    assert result.actions[1] == LimSimAction.DC


def test_evaluate_planning_result_reports_actions_and_reference_error():
    config = LimSimConfig(horizon_steps=6, mcts_iterations=20, interaction_distance=15.0)
    map_ = _build_parallel_map()
    participants = {1: _vehicle(1, 0, 1.0, 5.0, speed=5.0), 2: _vehicle(2, 0, 1.0, 20.0, speed=5.0)}
    result = LimSimBehaviorModel(config).plan(participants, map_, frame=0)

    evaluation = evaluate_planning_result(
        result,
        reference_trajectories=result.trajectories,
        dimensions={1: (4.5, 1.8), 2: (4.5, 1.8)},
    )

    assert evaluation.action_counts
    assert evaluation.has_collision is False
    assert evaluation.mean_ade == pytest.approx(0.0)
    assert evaluation.mean_fde == pytest.approx(0.0)


def test_interaction_graph_uses_successor_lane_relationship():
    config = LimSimConfig(interaction_distance=1.0)
    map_ = _build_successor_map()
    participants = {
        1: _vehicle(1, 0, 1.0, 27.0, speed=5.0),
        2: _vehicle(2, 0, 1.0, 33.0, speed=2.0),
    }
    states = SceneBuilder(config).build(participants, map_, frame=0)

    groups = InteractionGraph(config).build_groups(states, map_)

    assert sorted(sorted(group) for group in groups) == [[1, 2]]


def test_limsim_predictor_reuses_remaining_planned_trajectory():
    config = LimSimConfig(horizon_steps=4)
    map_ = _build_parallel_map()
    participants = {1: _vehicle(1, 0, 1.0, 5.0)}
    previous = LimSimBehaviorModel(config).plan(participants, map_, frame=0).trajectories

    prediction = LimSimPredictor(config).predict(
        participants, map_, frame=100, last_planned_trajectories=previous
    )

    assert 1 in prediction
    assert prediction[1].first_frame > 100


def test_mcts_selects_deceleration_for_rear_end_risk():
    config = LimSimConfig(horizon_steps=20, mcts_iterations=300, interaction_distance=20.0)
    map_ = _build_parallel_map()
    participants = {
        1: _vehicle(1, 0, 1.0, 10.0, speed=8.0),
        2: _vehicle(2, 0, 1.0, 24.0, speed=1.0),
    }

    result = LimSimBehaviorModel(config).plan(participants, map_, frame=0)

    assert result.actions[1] == LimSimAction.DC


def test_mcts_prefers_braking_when_collision_is_hard_to_avoid():
    config = LimSimConfig(horizon_steps=20, mcts_iterations=300, interaction_distance=20.0)
    map_ = _build_parallel_map()
    participants = {
        1: _vehicle(1, 0, 1.0, 10.0, speed=8.0),
        2: _vehicle(2, 0, 1.0, 16.0, speed=1.0),
    }

    result = LimSimBehaviorModel(config).plan(participants, map_, frame=0)

    assert result.actions[1] == LimSimAction.DC


def test_collision_checker_detects_overlapping_rollouts():
    config = LimSimConfig(horizon_steps=3)
    map_ = _build_parallel_map()
    participants = {1: _vehicle(1, 0, 1.0, 5.0, speed=0.0), 2: _vehicle(2, 0, 1.0, 5.0, speed=0.0)}
    states = SceneBuilder(config).build(participants, map_, frame=0)
    follower = LaneFollower(config)
    trajectories = [
        follower.rollout(states[1], LimSimAction.KEEP, map_),
        follower.rollout(states[2], LimSimAction.KEEP, map_),
    ]

    assert has_trajectory_collision(trajectories)
