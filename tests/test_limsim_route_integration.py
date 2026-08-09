# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Regression tests for LimSim route extraction and route-bound lane matching."""

import numpy as np
import pytest
from shapely.geometry import LineString

from tactics2d.behavior.limsim import LimSimConfig
from tactics2d.behavior.limsim.action import LimSimAction
from tactics2d.behavior.limsim.decision_search import LimSimDecisionSearch
from tactics2d.behavior.limsim.frenet_planner import (
    FrenetTrajectoryPlanner,
    reference_path_from_agent,
)
from tactics2d.behavior.limsim.lane_follower import (
    LaneFollower,
    _route_path_from_lanes,
    available_lanes_from_route,
    route_lanes_from_agent,
)
from tactics2d.behavior.limsim.scene import SceneBuilder
from tactics2d.behavior.limsim.schema import AgentDecisionState
from tactics2d.dataset_parser.route_extractor import extract_lane_sequence, infer_lane_topology
from tactics2d.geometry.frenet import ReferencePath
from tactics2d.map.element import Lane, Map
from tactics2d.participant.element import Vehicle
from tactics2d.participant.trajectory import State, Trajectory


def _lane(lane_id, center_x, y_start, y_end, line_ids=None):
    return Lane(
        id_=lane_id,
        left_side=LineString([(center_x - 1.0, y_start), (center_x - 1.0, y_end)]),
        right_side=LineString([(center_x + 1.0, y_start), (center_x + 1.0, y_end)]),
        line_ids=line_ids or {"left": [], "right": []},
        custom_tags={
            "centerline": np.asarray([(center_x, y_start), (center_x, y_end)], dtype=float)
        },
    )


def _vehicle(agent_id, states):
    trajectory = Trajectory(id_=agent_id, fps=10, stable_freq=True)
    for frame, x, y in states:
        trajectory.add_state(State(frame=frame, x=x, y=y, heading=np.pi / 2, vx=0.0, vy=5.0))
    return Vehicle(id_=agent_id, type_="vehicle", trajectory=trajectory, length=4.5, width=1.8)


def _route_choice_map():
    map_ = Map(name="route_choice")
    for lane in (
        _lane("A", 0.0, 0.0, 200.0),
        _lane("B", 3.0, 0.0, 200.0),
        _lane("J", 0.0, 200.0, 220.0),
        _lane("X", 3.0, 200.0, 220.0),
        _lane("C", 0.0, 220.0, 320.0),
        _lane("D", 3.0, 220.0, 320.0),
        _lane("E", 6.0, 220.0, 320.0),
    ):
        map_.add_lane(lane)

    map_.lanes["A"].right_neighbors.add("B")
    map_.lanes["B"].left_neighbors.add("A")
    map_.lanes["C"].right_neighbors.add("D")
    map_.lanes["D"].left_neighbors.add("C")
    map_.lanes["J"].custom_tags["is_internal"] = True
    map_.lanes["X"].custom_tags["is_internal"] = True
    map_.lanes["A"].successors.add("J")
    map_.lanes["J"].successors.add("C")
    map_.lanes["B"].successors.add("X")
    map_.lanes["X"].successors.add("E")
    return map_


def _route_agent(lane_id, x, y, progress):
    return AgentDecisionState(
        agent_id=1,
        x=x,
        y=y,
        heading=np.pi / 2.0,
        speed=5.0,
        lane_id=lane_id,
        route_lane_ids=("A", "J", "C"),
        route_progress=progress,
    )


def test_available_lanes_are_dynamic_and_route_compatible():
    map_ = _route_choice_map()

    far = available_lanes_from_route(_route_agent("A", 0.0, 20.0, 20.0), map_)
    near = available_lanes_from_route(_route_agent("A", 0.0, 150.0, 150.0), map_)
    wrong_lane = available_lanes_from_route(_route_agent("B", 3.0, 150.0, 150.0), map_)
    connector = available_lanes_from_route(_route_agent("J", 0.0, 205.0, 5.0), map_)

    assert far == frozenset({"A", "B"})
    assert near == frozenset({"A", "J"})
    assert wrong_lane == frozenset({"A", "J"})
    assert connector == frozenset({"J", "C", "D"})


def test_available_lanes_include_alternative_connectors_to_the_route_section():
    map_ = _route_choice_map()
    map_.lanes["X"].successors = {"D"}

    available = available_lanes_from_route(_route_agent("A", 0.0, 150.0, 150.0), map_)

    assert available == frozenset({"A", "B", "J", "X"})


def test_available_lanes_apply_the_official_short_lane_entry_window():
    map_ = Map(name="short_available")
    for lane in (
        _lane("A", 0.0, 0.0, 50.0),
        _lane("B", 3.0, 0.0, 50.0),
        _lane("C", 0.0, 50.0, 100.0),
    ):
        map_.add_lane(lane)
    map_.lanes["A"].right_neighbors.add("B")
    map_.lanes["B"].left_neighbors.add("A")
    map_.lanes["A"].successors.add("C")

    agent = AgentDecisionState(
        1, 0.0, 2.0, np.pi / 2.0, 5.0, lane_id="A", route_lane_ids=("A", "C"), route_progress=2.0
    )

    assert available_lanes_from_route(agent, map_) == frozenset({"A", "B"})
    assert available_lanes_from_route(
        agent.with_updates(y=10.0, route_progress=10.0), map_
    ) == frozenset({"A", "C"})


def test_internal_available_lanes_keep_current_connector_without_topology_path():
    map_ = Map(name="internal_without_path")
    map_.add_lane(_lane("J", 0.0, 0.0, 10.0))
    map_.add_lane(_lane("C", 0.0, 20.0, 30.0))
    map_.lanes["J"].custom_tags["is_internal"] = True
    agent = AgentDecisionState(
        1, 0.0, 5.0, np.pi / 2.0, 5.0, lane_id="J", route_lane_ids=("J", "C"), route_progress=5.0
    )

    assert available_lanes_from_route(agent, map_) == frozenset({"J", "C"})


def test_mcts_actions_follow_available_lane_direction_and_braking_boundary():
    map_ = _route_choice_map()
    search = LimSimDecisionSearch(LimSimConfig())
    wrong = _route_agent("B", 3.0, 150.0, 150.0).with_updates(
        available_lane_ids=frozenset({"A", "J"})
    )
    valid = _route_agent("A", 0.0, 150.0, 150.0).with_updates(
        available_lane_ids=frozenset({"A", "J"})
    )

    assert search._candidate_actions(valid, map_) == [
        LimSimAction.KS,
        LimSimAction.AC,
        LimSimAction.DC,
    ]
    assert search._candidate_actions(wrong, map_) == [
        LimSimAction.KS,
        LimSimAction.AC,
        LimSimAction.DC,
        LimSimAction.LCL,
    ]
    assert search._candidate_actions(wrong.with_updates(y=199.0, route_progress=199.0), map_) == [
        LimSimAction.DC
    ]


def test_lane_follower_updates_lane_id_and_respects_available_successors():
    map_ = Map(name="successor_available")
    map_.add_lane(_lane("A", 0.0, 0.0, 10.0))
    map_.add_lane(_lane("C", 0.0, 10.0, 20.0))
    map_.lanes["A"].successors.add("C")
    agent = AgentDecisionState(
        1,
        0.0,
        8.0,
        np.pi / 2.0,
        5.0,
        lane_id="A",
        route_lane_ids=("A", "C"),
        available_lane_ids=frozenset({"A", "C"}),
        route_progress=8.0,
    )
    follower = LaneFollower(LimSimConfig(horizon_steps=5))

    allowed = follower.rollout(agent, LimSimAction.KS, map_)
    blocked = follower.rollout(
        agent.with_updates(available_lane_ids=frozenset({"A"})), LimSimAction.KS, map_
    )

    assert allowed[-1].lane_id == "C"
    assert allowed[-1].route_progress == 0.5
    assert blocked
    assert all(state.lane_id == "A" for state in blocked)


def test_scene_without_route_map_keeps_topology_fallback_unconstrained():
    map_ = Map(name="topology_fallback")
    map_.add_lane(_lane("A", 0.0, 0.0, 10.0))
    map_.add_lane(_lane("C", 0.0, 10.0, 20.0))
    map_.lanes["A"].successors.add("C")
    participant = _vehicle(1, [(0, 0.0, 8.0)])
    config = LimSimConfig(horizon_steps=5)

    state = SceneBuilder(config).build({1: participant}, map_, frame=0, route_map={})[1]
    rollout = LaneFollower(config).rollout(state, LimSimAction.KS, map_)

    assert state.available_lane_ids == frozenset()
    assert rollout[-1].lane_id == "C"


def test_reward_uses_available_lanes_instead_of_oracle_route_membership():
    search = LimSimDecisionSearch(LimSimConfig())
    agent = AgentDecisionState(
        1,
        0.0,
        0.0,
        0.0,
        5.0,
        lane_id="A",
        route_lane_ids=("A", "B"),
        available_lane_ids=frozenset({"A"}),
    )
    on_available = agent.with_updates(action=LimSimAction.KS)
    on_oracle_only = agent.with_updates(lane_id="B", action=LimSimAction.KS)

    steps_per_decision = int(round(search.config.decision_resolution / search.config.dt))
    available_reward = search.reward.evaluate([agent], {1: [on_available] * steps_per_decision})
    oracle_only_reward = search.reward.evaluate([agent], {1: [on_oracle_only] * steps_per_decision})

    assert available_reward == 1.0
    assert oracle_only_reward == 0.4


def test_reward_scores_action_continuity_once_per_decision_stage():
    config = LimSimConfig()
    search = LimSimDecisionSearch(config)
    agent = AgentDecisionState(
        1, 0.0, 0.0, 0.0, 5.0, lane_id="A", available_lane_ids=frozenset({"A"})
    )
    steps_per_decision = int(round(config.decision_resolution / config.dt))

    def trajectory_for(actions):
        trajectory = []
        for index, action in enumerate(actions):
            lane_id = "B" if index == 3 else "A"
            trajectory.extend(
                [agent.with_updates(action=action, lane_id=lane_id)] * steps_per_decision
            )
        return trajectory

    alternating_reward = search.reward.evaluate(
        [agent], {1: trajectory_for((LimSimAction.KS, LimSimAction.DC) * 2)}
    )
    repeated_reward = search.reward.evaluate(
        [agent],
        {1: trajectory_for((LimSimAction.KS, LimSimAction.KS, LimSimAction.DC, LimSimAction.DC))},
    )

    assert alternating_reward == pytest.approx(0.44)
    assert repeated_reward == pytest.approx(0.52)


def test_topology_inference_and_route_decode_reject_disconnected_lane():
    map_ = Map(name="route")
    map_.add_lane(_lane("A", 0.0, 0.0, 10.0))
    map_.add_lane(_lane("B", 0.0, 10.0, 20.0))
    map_.add_lane(_lane("X", 0.4, 3.0, 7.0))
    participant = _vehicle(
        1,
        [
            (0, 0.0, 1.0),
            (100, 0.0, 4.0),
            (200, 0.4, 6.0),
            (300, 0.0, 8.0),
            (400, 0.0, 11.0),
            (500, 0.0, 15.0),
        ],
    )

    counts = infer_lane_topology(map_)
    route = extract_lane_sequence(participant, map_)

    assert counts["successors"] == 1
    assert map_.lanes["B"].id_ in map_.lanes["A"].successors
    assert route == ["A", "B"]


def test_scene_lane_matching_stays_in_route_corridor():
    map_ = Map(name="overlap")
    map_.add_lane(_lane("route", 0.0, 0.0, 20.0))
    map_.add_lane(_lane("unrelated", 0.4, 0.0, 20.0))
    participant = _vehicle(1, [(0, 0.4, 5.0)])

    state = SceneBuilder(LimSimConfig()).build(
        {1: participant}, map_, frame=0, route_map={1: ("route",)}
    )[1]

    assert state.lane_id == "route"


def test_route_cursor_never_returns_to_an_earlier_occurrence():
    builder = SceneBuilder(LimSimConfig())
    route = ("A", "B", "A", "C")

    assert builder._advance_route_cursor(1, route, "A") == route
    assert builder._advance_route_cursor(1, route, "B") == ("B", "A", "C")
    assert builder._advance_route_cursor(1, route, "A") == ("A", "C")


def test_route_prefix_uses_only_legal_unique_successor():
    map_ = Map(name="route_prefix")
    map_.add_lane(_lane("A", 0.0, 0.0, 10.0))
    map_.add_lane(_lane("B", 0.0, 10.0, 20.0))
    map_.add_lane(_lane("X", 4.0, 0.0, 10.0))
    map_.lanes["A"].successors.add("B")

    agent = AgentDecisionState(
        agent_id=1,
        x=0.0,
        y=5.0,
        heading=np.pi / 2,
        speed=5.0,
        lane_id="A",
        route_lane_ids=("A", "X"),
    )

    assert route_lanes_from_agent(agent, map_, max_routes=3) == ["A", "B"]


def test_route_prefix_stops_at_an_ambiguous_branch():
    map_ = Map(name="route_branch")
    for lane_id, x in (("A", 0.0), ("B", -1.0), ("C", 1.0)):
        map_.add_lane(_lane(lane_id, x, 0.0, 10.0))
    map_.lanes["A"].successors.update({"B", "C"})

    agent = AgentDecisionState(
        agent_id=1,
        x=0.0,
        y=5.0,
        heading=np.pi / 2,
        speed=5.0,
        lane_id="A",
        route_lane_ids=("A", "X"),
    )

    assert route_lanes_from_agent(agent, map_, max_routes=3) == ["A"]


def test_route_prefix_accepts_only_forward_local_connection():
    map_ = Map(name="route_local_connection")
    map_.add_lane(_lane("A", 0.0, 0.0, 10.0))
    map_.add_lane(_lane("B", 0.0, 12.5, 22.5))
    map_.add_lane(_lane("backward", 0.0, 7.0, -3.0))
    agent = AgentDecisionState(
        agent_id=1,
        x=0.0,
        y=5.0,
        heading=np.pi / 2,
        speed=5.0,
        lane_id="A",
        route_lane_ids=("A", "B"),
    )

    assert route_lanes_from_agent(agent, map_, max_routes=2) == ["A", "B"]
    assert "B" not in map_.lanes["A"].successors

    backward_agent = agent.with_updates(route_lane_ids=("A", "backward"))
    assert route_lanes_from_agent(backward_agent, map_, max_routes=2) == ["A"]


def test_recorded_route_splices_an_interior_lane_exit():
    map_ = Map(name="route_interior_exit")
    map_.add_lane(_lane("A", 0.0, 0.0, 10.0))
    map_.add_lane(_lane("straight", 0.0, 10.0, 20.0))
    map_.add_lane(_lane("exit", 0.1, 6.0, 16.0))
    map_.lanes["A"].successors.add("straight")
    agent = AgentDecisionState(
        agent_id=1,
        x=0.0,
        y=4.0,
        heading=np.pi / 2,
        speed=5.0,
        lane_id="A",
        route_lane_ids=("A", "exit"),
    )

    route_lanes = route_lanes_from_agent(agent, map_, max_routes=2)
    route_path = _route_path_from_lanes(route_lanes, map_, agent.route_lane_ids)
    reference = reference_path_from_agent(agent, map_, LimSimConfig(max_routes_per_agent=2))

    assert route_lanes == ["A", "exit"]
    assert route_path is not None
    assert np.max(route_path[:, 1]) == 16.0
    assert not np.any(np.isclose(route_path[:, 1], 10.0))
    assert reference is not None
    assert reference.lane_ids == ("A", "exit")
    assert reference.path.length < 17.0


def test_collision_stop_candidate_brakes_continuously():
    config = LimSimConfig(horizon_steps=15, dt=0.1, frenet_stop_deceleration=4.0)
    planner = FrenetTrajectoryPlanner(config)
    reference_path = ReferencePath(LineString([(0.0, 0.0), (30.0, 0.0)]), lane_ids=("A",))
    agent = AgentDecisionState(
        agent_id=1,
        x=5.0,
        y=0.0,
        heading=0.0,
        speed=4.5,
        lane_id="A",
        route_lane_ids=("A",),
        route_progress=5.0,
    )
    start = reference_path.cartesian_to_frenet(agent.x, agent.y)

    candidate = planner._stop_candidate(agent, LimSimAction.KS, reference_path, start, ())
    speeds = [state.speed for state in candidate.states]

    assert 0.0 < speeds[0] < agent.speed
    assert all(next_speed <= speed for speed, next_speed in zip(speeds, speeds[1:]))
    assert candidate.states[0].x > agent.x
    assert speeds[-1] == 0.0


def test_collision_fallback_returns_state_sequence_from_public_plan():
    map_ = Map(name="collision_stop")
    map_.add_lane(_lane("A", 0.0, 0.0, 30.0))
    config = LimSimConfig(horizon_steps=15, dt=0.1, frenet_stop_deceleration=4.0)
    planner = FrenetTrajectoryPlanner(config)
    agent = AgentDecisionState(
        agent_id=1,
        x=0.0,
        y=5.0,
        heading=np.pi / 2,
        speed=4.5,
        lane_id="A",
        route_lane_ids=("A",),
        route_progress=5.0,
    )
    obstacle = AgentDecisionState(
        agent_id=2,
        x=0.0,
        y=7.0,
        heading=np.pi / 2,
        speed=0.0,
        lane_id="A",
        route_lane_ids=("A",),
        route_progress=7.0,
    )

    states = planner.plan(
        agent, LimSimAction.KS, map_, obstacle_trajectories=([obstacle] * config.horizon_steps,)
    )

    assert isinstance(states, list)
    assert states
    assert 0.0 < states[0].speed < agent.speed
