# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Smoke tests for LimSim-style interactive behavior planning.

Each test runs a scenario end-to-end and checks the key outcome — no
exhaustive field-by-field assertions.
"""

import numpy as np
import pytest
from shapely.geometry import LineString, Point

from tactics2d.behavior import LimSimBehaviorModel
from tactics2d.behavior.limsim.action import LimSimAction
from tactics2d.behavior.limsim.config import LimSimConfig
from tactics2d.behavior.limsim.frenet_planner import (
    FrenetTrajectoryPlanner,
    reference_path_from_agent,
)
from tactics2d.behavior.limsim.interaction import InteractionGraph, has_trajectory_collision
from tactics2d.behavior.limsim.lane_follower import LaneFollower
from tactics2d.behavior.limsim.roi import RoISelector
from tactics2d.behavior.limsim.scene import SceneBuilder
from tactics2d.behavior.limsim.schema import evaluate_planning_result
from tactics2d.geometry import frenet
from tactics2d.map.element import Junction, Lane, LaneRelationship, Map, Regulatory, RoadLine
from tactics2d.map.query import SemanticMapQuery
from tactics2d.participant.element import Pedestrian, Vehicle
from tactics2d.participant.trajectory import State, Trajectory

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _lane(lane_id, x_left, x_right, y_start, y_end):
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


def _parallel_map():
    map_ = Map(name="parallel")
    a = _lane("A", 0.0, 2.0, 0.0, 80.0)
    b = _lane("B", 2.0, 4.0, 0.0, 80.0)
    a.add_related_lane("B", LaneRelationship.RIGHT_NEIGHBOR)
    b.add_related_lane("A", LaneRelationship.LEFT_NEIGHBOR)
    map_.add_lane(a)
    map_.add_lane(b)
    return map_


def _successor_map():
    map_ = Map(name="successor")
    a = _lane("A", 0.0, 2.0, 0.0, 30.0)
    b = _lane("B", 0.0, 2.0, 30.0, 80.0)
    a.add_related_lane("B", LaneRelationship.SUCCESSOR)
    map_.add_lane(a)
    map_.add_lane(b)
    return map_


def _crossing_map():
    map_ = Map(name="crossing")
    ew = Lane(
        id_="EW",
        left_side=LineString([(-10.0, 1.0), (10.0, 1.0)]),
        right_side=LineString([(-10.0, -1.0), (10.0, -1.0)]),
        custom_tags={"centerline": np.array([[-10.0, 0.0], [10.0, 0.0]])},
    )
    ns = Lane(
        id_="NS",
        left_side=LineString([(-1.0, -10.0), (-1.0, 10.0)]),
        right_side=LineString([(1.0, -10.0), (1.0, 10.0)]),
        custom_tags={"centerline": np.array([[0.0, -10.0], [0.0, 10.0]])},
    )
    map_.add_lane(ew)
    map_.add_lane(ns)
    return map_


def _semantic_map():
    map_ = _parallel_map()
    map_.add_roadline(
        RoadLine(id_="solid_left", geometry=LineString([(0.0, 0.0), (0.0, 80.0)]), subtype="solid")
    )
    map_.add_roadline(
        RoadLine(
            id_="dashed_right", geometry=LineString([(2.0, 0.0), (2.0, 80.0)]), subtype="dashed"
        )
    )
    lane = map_.lanes["A"]
    lane.line_ids = {"left": ["solid_left"], "right": ["dashed_right"]}
    lane.custom_tags = {
        **(lane.custom_tags or {}),
        "boundary_segments": {
            "left": [{"roadline_id": "solid_left", "start_s": 0.0, "end_s": 80.0}],
            "right": [{"roadline_id": "dashed_right", "start_s": 0.0, "end_s": 80.0}],
        },
    }
    map_.add_regulatory(
        Regulatory(
            id_="stop_A", ways={"A": "refers"}, subtype="stop_sign", position=Point(1.0, 20.0)
        )
    )
    map_.add_regulatory(
        Regulatory(
            id_="tl_A",
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


# ---------------------------------------------------------------------------
# scene building + lane matching
# ---------------------------------------------------------------------------


@pytest.mark.map_parser
@pytest.mark.env
def test_scene_and_lane_matching():
    """Lane matching assigns correct lanes in a simple parallel map."""
    map_ = _parallel_map()
    participants = {1: _vehicle(1, 0, 1.0, 5.0), 2: _vehicle(2, 0, 3.0, 8.0)}

    states = SceneBuilder(LimSimConfig()).build(participants, map_, frame=0)

    assert states[1].lane_id == "A"
    assert states[2].lane_id == "B"


def test_scene_prefers_heading_consistent_lane():
    """Heading-consistent lane wins over geometrically nearest."""
    map_ = Map(name="heading_match")
    nb = Lane(
        id_="N",
        left_side=LineString([(-1.0, -10.0), (-1.0, 10.0)]),
        right_side=LineString([(1.0, -10.0), (1.0, 10.0)]),
        custom_tags={"centerline": np.array([[0.0, -10.0], [0.0, 10.0]])},
    )
    x = Lane(
        id_="X",
        left_side=LineString([(-10.0, 0.3), (10.0, 0.3)]),
        right_side=LineString([(-10.0, -0.7), (10.0, -0.7)]),
        custom_tags={"centerline": np.array([[-10.0, -0.2], [10.0, -0.2]])},
    )
    map_.add_lane(nb)
    map_.add_lane(x)

    states = SceneBuilder(LimSimConfig()).build(
        {1: _vehicle(1, 0, 0.0, 0.1, heading=np.pi / 2)}, map_, frame=0
    )

    assert states[1].lane_id == "N"


# ---------------------------------------------------------------------------
# semantic map queries
# ---------------------------------------------------------------------------


@pytest.mark.map_parser
def test_semantic_query():
    """Conflict detection, stop targets, lane-change permission, junctions."""
    # crossing conflict
    cq = SemanticMapQuery(_crossing_map())
    assert cq.has_conflict("EW", "NS")

    # stop & traffic-light targets
    sq = SemanticMapQuery(_semantic_map())
    stop_targets = sq.get_stop_targets("A", time_ms=0)
    assert {t.reason for t in stop_targets} == {"stop_sign", "traffic_light"}

    # lane-change permission
    assert sq.get_lane_change_permission("A", "right", s=10.0)
    assert not sq.get_lane_change_permission("A", "left", s=10.0)

    # junction area
    area = sq.get_junction_area(lane_id="A")
    assert area is not None

    # reference path with lookahead
    ref = SemanticMapQuery(_successor_map()).get_reference_path("A", lookahead_lanes=2)
    assert ref is not None and ref.lane_ids == ("A", "B")


# ---------------------------------------------------------------------------
# lane follower + frenet planner
# ---------------------------------------------------------------------------


def test_lane_follower_and_frenet_planner():
    """Keep, lane-change, stop-for-obstacle, and candidate filtering."""
    config = LimSimConfig(horizon_steps=20)
    map_ = _parallel_map()
    state = SceneBuilder(config).build({1: _vehicle(1, 0, 1.0, 5.0, speed=8.0)}, map_, frame=0)[1]
    follower = LaneFollower(config)

    # lane-keep moves forward
    keep = follower.rollout(state, LimSimAction.KEEP, map_)
    assert keep[-1].y > state.y
    assert keep[-1].lane_id == "A"

    # lane-change-right eventually switches lane
    lcr = follower.rollout(state, LimSimAction.LCR, map_)
    assert lcr[-1].lane_id == "B"

    # frenet planner: illegal LCL filtered, legal LCR allowed
    smap = _semantic_map()
    smap_state = SceneBuilder(config).build({1: _vehicle(1, 0, 1.0, 5.0)}, smap, frame=0)[1]
    planner = FrenetTrajectoryPlanner(config)
    ref = reference_path_from_agent(smap_state, smap, config)
    assert planner.sample_candidates(smap_state, LimSimAction.LCL, ref, smap) == []
    assert planner.sample_candidates(smap_state, LimSimAction.LCR, ref, smap)

    # frenet planner stops for stop target
    stop_config = LimSimConfig(horizon_steps=40, frenet_stop_line_penalty=10000.0)
    stop_state = SceneBuilder(stop_config).build(
        {1: _vehicle(1, 0, 1.0, 5.0, speed=8.0)}, smap, frame=0
    )[1]
    traj = FrenetTrajectoryPlanner(stop_config).plan(stop_state, LimSimAction.KS, smap)
    assert traj[-1].speed <= stop_config.frenet_stop_speed_threshold

    # collision checker
    collision_config = LimSimConfig(horizon_steps=3)
    cs = SceneBuilder(collision_config).build(
        {1: _vehicle(1, 0, 1.0, 5.0, speed=0.0), 2: _vehicle(2, 0, 1.0, 5.0, speed=0.0)},
        map_,
        frame=0,
    )
    traj_a = LaneFollower(collision_config).rollout(cs[1], LimSimAction.KEEP, map_)
    traj_b = LaneFollower(collision_config).rollout(cs[2], LimSimAction.KEEP, map_)
    assert has_trajectory_collision([traj_a, traj_b])


# ---------------------------------------------------------------------------
# behavior model end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.env
def test_behavior_model_basic():
    """Single-step plan produces trajectories for all controlled agents."""
    config = LimSimConfig(horizon_steps=6, dt=0.2, mcts_iterations=20, interaction_distance=15.0)
    map_ = _parallel_map()
    participants = {1: _vehicle(1, 0, 1.0, 5.0, speed=5.0), 2: _vehicle(2, 0, 1.0, 10.0, speed=2.0)}

    result = LimSimBehaviorModel(config).plan(participants, map_, route_map={}, frame=0)

    assert set(result.actions) == {1, 2}
    assert set(result.trajectories) == {1, 2}
    assert len(result.trajectories[1].frames) == config.horizon_steps


def test_behavior_model_with_obstacle_yields_deceleration():
    """Faster agent behind slower one decelerates."""
    config = LimSimConfig(horizon_steps=20, mcts_iterations=200, interaction_distance=20.0)
    participants = {
        1: _vehicle(1, 0, 1.0, 10.0, speed=8.0),
        2: _vehicle(2, 0, 1.0, 20.0, speed=1.0),
    }

    result = LimSimBehaviorModel(config).plan(participants, _parallel_map(), route_map={}, frame=0)

    assert result.actions[1] == LimSimAction.DC


# ---------------------------------------------------------------------------
# roi selection + filtering
# ---------------------------------------------------------------------------


def test_roi_and_filtering():
    """RoI selects agents within radius; non-vehicles filtered."""
    participants = {
        1: _vehicle(1, 0, 0.0, 0.0),
        2: _vehicle(2, 0, 4.0, 0.0),
        3: _vehicle(3, 0, 9.0, 0.0),
    }

    sel = RoISelector.select_around_agent(participants, frame=0, ego_id=1, radius=5.0)
    assert sel.agent_ids == [1, 2]
    assert sel.background_agent_ids == [3]

    # pedestrian not included as controlled agent
    config = LimSimConfig(horizon_steps=6, mcts_iterations=20, interaction_distance=15.0)
    mixed = {1: _vehicle(1, 0, 1.0, 5.0), 2: _pedestrian(2, 0, 1.0, 6.0)}
    result = LimSimBehaviorModel(config).plan(mixed, _parallel_map(), route_map={}, frame=0)
    assert set(result.actions) == {1}  # pedestrian filtered


# ---------------------------------------------------------------------------
# interaction graph
# ---------------------------------------------------------------------------


def test_interaction_graph():
    """Nearby agents grouped; distant ones isolated."""
    config = LimSimConfig(interaction_distance=10.0)
    participants = {
        1: _vehicle(1, 0, 1.0, 5.0),
        2: _vehicle(2, 0, 3.0, 8.0),
        3: _vehicle(3, 0, 3.0, 60.0),
    }
    states = SceneBuilder(config).build(participants, _parallel_map(), frame=0)

    groups = InteractionGraph(config).build_groups(states)

    assert sorted(sorted(g) for g in groups) == [[1, 2], [3]]

    # successor relationship connects lanes
    sc = LimSimConfig(interaction_distance=1.0)
    ss = SceneBuilder(sc).build(
        {1: _vehicle(1, 0, 1.0, 27.0), 2: _vehicle(2, 0, 1.0, 33.0)}, _successor_map(), frame=0
    )
    assert sorted(sorted(g) for g in InteractionGraph(sc).build_groups(ss, _successor_map())) == [
        [1, 2]
    ]


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def test_evaluation():
    """Self-comparison yields zero ADE/FDE."""
    config = LimSimConfig(horizon_steps=6, mcts_iterations=20, interaction_distance=15.0)
    participants = {1: _vehicle(1, 0, 1.0, 5.0, speed=5.0)}
    result = LimSimBehaviorModel(config).plan(participants, _parallel_map(), route_map={}, frame=0)

    ev = evaluate_planning_result(
        result, reference_trajectories=result.trajectories, dimensions={1: (4.5, 1.8)}
    )

    assert ev.has_collision is False
    assert ev.mean_ade == pytest.approx(0.0)
