# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for polyline concatenation at lane-centerline joints."""

import numpy as np
from shapely.geometry import LineString

from tactics2d.behavior.limsim.action import LimSimAction
from tactics2d.behavior.limsim.config import LimSimConfig
from tactics2d.behavior.limsim.frenet_planner import FrenetTrajectoryPlanner
from tactics2d.behavior.limsim.lane_follower import LaneFollower
from tactics2d.behavior.limsim.schema import AgentDecisionState
from tactics2d.geometry import polyline
from tactics2d.geometry.frenet import ReferencePath
from tactics2d.map.element import Lane, Map


def _heading_steps(path):
    segments = np.diff(path, axis=0)
    headings = np.unwrap(np.arctan2(segments[:, 1], segments[:, 0]))
    return np.abs(np.diff(headings))


def test_concatenate_replaces_kink_joint_without_backtracking():
    incoming = np.asarray([[-2.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])
    outgoing = np.asarray([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])

    path = polyline.concatenate([incoming, outgoing])

    assert path is not None
    assert np.all(np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-8)
    assert np.max(_heading_steps(path)) < np.deg2rad(60.0)
    assert np.allclose(path[0], incoming[0])
    assert np.allclose(path[-1], outgoing[-1])


def test_concatenate_deduplicates_straight_joint():
    first = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    second = np.asarray([[1.0, 0.0], [2.0, 0.0]])

    path = polyline.concatenate([first, second])

    assert np.array_equal(path, np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))


def test_concatenate_preserves_disconnected_polylines():
    first = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    second = np.asarray([[3.0, 0.0], [4.0, 0.0]])

    path = polyline.concatenate([first, second])

    assert np.array_equal(path, np.vstack([first, second]))


def test_concatenate_ignores_single_point_joint_between_segments():
    first = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    joint = np.asarray([[1.0, 0.0]])
    third = np.asarray([[1.0, 0.0], [2.0, 0.0]])

    path = polyline.concatenate([first, joint, third])

    assert np.array_equal(path, np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))


def test_lane_follower_stays_on_short_route_at_route_end():
    lane = Lane(
        id_="short",
        left_side=LineString([(0.0, -1.0), (10.0, -1.0)]),
        right_side=LineString([(0.0, 1.0), (10.0, 1.0)]),
        custom_tags={"centerline": np.asarray([(0.0, 0.0), (10.0, 0.0)])},
    )
    map_ = Map(name="short_route")
    map_.add_lane(lane)
    agent = AgentDecisionState(
        agent_id=1,
        x=8.0,
        y=0.0,
        heading=0.0,
        speed=8.0,
        lane_id="short",
        lateral_offset=0.0,
        route_lane_ids=("short",),
        route_progress=8.0,
        length=4.5,
        width=1.8,
    )

    rollout = LaneFollower(LimSimConfig(horizon_steps=8, max_speed=20.0)).rollout(
        agent, LimSimAction.KS, map_
    )
    points = np.asarray([(state.x, state.y) for state in rollout])
    assert np.max(points[:, 0]) <= 10.0


def test_lane_follower_projects_start_on_current_lane_prefix():
    current = Lane(
        id_="current",
        left_side=LineString([(0.0, -1.0), (10.0, -1.0)]),
        right_side=LineString([(0.0, 1.0), (10.0, 1.0)]),
        custom_tags={"centerline": np.asarray([(0.0, 0.0), (10.0, 0.0)])},
    )
    successor = Lane(
        id_="successor",
        left_side=LineString([(10.0, 9.0), (2.0, -0.98)]),
        right_side=LineString([(10.0, 11.0), (2.0, 1.02)]),
        custom_tags={"centerline": np.asarray([(10.0, 0.0), (10.0, 10.0), (2.0, 0.02)])},
    )
    current.successors.add("successor")
    map_ = Map(name="self_near_route")
    map_.add_lane(current)
    map_.add_lane(successor)
    agent = AgentDecisionState(
        agent_id=1,
        x=2.0,
        y=0.1,
        heading=0.0,
        speed=1.0,
        lane_id="current",
        route_lane_ids=("current", "successor"),
        route_progress=2.0,
    )

    rollout = LaneFollower(LimSimConfig(horizon_steps=1)).rollout(agent, LimSimAction.KS, map_)

    assert rollout[0].x < 3.0
    assert abs(rollout[0].y) < 0.2


def test_lane_follower_preserves_progress_across_mcts_segments():
    current = Lane(
        id_="current",
        left_side=LineString([(0.0, -1.0), (10.0, -1.0)]),
        right_side=LineString([(0.0, 1.0), (10.0, 1.0)]),
        custom_tags={"centerline": np.asarray([(0.0, 0.0), (10.0, 0.0)])},
    )
    successor = Lane(
        id_="successor",
        left_side=LineString([(10.0, -1.0), (30.0, -1.0)]),
        right_side=LineString([(10.0, 1.0), (30.0, 1.0)]),
        custom_tags={"centerline": np.asarray([(10.0, 0.0), (30.0, 0.0)])},
    )
    current.successors.add("successor")
    map_ = Map(name="segmented_rollout")
    map_.add_lane(current)
    map_.add_lane(successor)
    agent = AgentDecisionState(
        agent_id=1,
        x=8.0,
        y=0.0,
        heading=0.0,
        speed=8.0,
        lane_id="current",
        route_lane_ids=("current", "successor"),
        route_progress=8.0,
    )
    follower = LaneFollower(LimSimConfig(horizon_steps=15, max_speed=20.0))

    first_segment = follower.rollout(agent, LimSimAction.KS, map_, steps=15)
    second_segment = follower.rollout(first_segment[-1], LimSimAction.KS, map_, steps=15)
    seam_distance = np.hypot(
        second_segment[0].x - first_segment[-1].x, second_segment[0].y - first_segment[-1].y
    )

    assert np.isclose(seam_distance, 0.8, atol=1e-6)
    assert second_segment[0].x > first_segment[-1].x


def test_lane_follower_does_not_bridge_disconnected_successor():
    current = Lane(
        id_="current",
        left_side=LineString([(0.0, -1.0), (10.0, -1.0)]),
        right_side=LineString([(0.0, 1.0), (10.0, 1.0)]),
        custom_tags={"centerline": np.asarray([(0.0, 0.0), (10.0, 0.0)])},
    )
    disconnected = Lane(
        id_="disconnected",
        left_side=LineString([(20.0, -1.0), (30.0, -1.0)]),
        right_side=LineString([(20.0, 1.0), (30.0, 1.0)]),
        custom_tags={"centerline": np.asarray([(20.0, 0.0), (30.0, 0.0)])},
    )
    current.successors.add("disconnected")
    map_ = Map(name="disconnected_route")
    map_.add_lane(current)
    map_.add_lane(disconnected)
    agent = AgentDecisionState(
        agent_id=1,
        x=8.0,
        y=0.0,
        heading=0.0,
        speed=8.0,
        lane_id="current",
        route_lane_ids=("current", "disconnected"),
        route_progress=8.0,
    )

    rollout = LaneFollower(LimSimConfig(horizon_steps=15, max_speed=20.0)).rollout(
        agent, LimSimAction.KS, map_
    )
    points = np.asarray([(state.x, state.y) for state in rollout])

    assert np.max(np.linalg.norm(np.diff(points, axis=0), axis=1)) < 1.0
    assert all(state.route_lane_ids == ("current", "disconnected") for state in rollout)


def test_frenet_planner_stays_on_short_route_at_route_end():
    lane = Lane(
        id_="short",
        left_side=LineString([(0.0, -1.0), (10.0, -1.0)]),
        right_side=LineString([(0.0, 1.0), (10.0, 1.0)]),
        custom_tags={"centerline": np.asarray([(0.0, 0.0), (10.0, 0.0)])},
    )
    map_ = Map(name="short_frenet_route")
    map_.add_lane(lane)
    agent = AgentDecisionState(
        agent_id=1,
        x=8.0,
        y=0.0,
        heading=0.0,
        speed=8.0,
        lane_id="short",
        lateral_offset=0.0,
        route_lane_ids=("short",),
        route_progress=8.0,
        length=4.5,
        width=1.8,
    )

    rollout = FrenetTrajectoryPlanner(LimSimConfig(horizon_steps=8, max_speed=20.0)).plan(
        agent, LimSimAction.KS, map_
    )
    points = np.asarray([(state.x, state.y) for state in rollout])
    assert np.max(points[:, 0]) <= 10.0


def test_reference_path_accepts_local_projection_hint_on_nearby_route_geometry():
    path = LineString([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (5.0, 0.1)])
    reference = ReferencePath(path, initial_s=5.0)

    hinted = reference.cartesian_to_frenet(5.0, 0.1, hint_s=5.0)
    projected = reference.cartesian_to_frenet(5.0, 0.1)

    assert hinted.s == 5.0
    assert projected.s > hinted.s
    endpoint = reference.frenet_to_cartesian(path.length + 2.0, 0.0)
    assert np.allclose(endpoint[:2], path.coords[-1])
