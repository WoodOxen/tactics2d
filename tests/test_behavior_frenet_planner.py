# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the LimSim Frenet trajectory planner."""

from dataclasses import replace

import numpy as np
import pytest
from shapely.geometry import LineString, Point

from tactics2d.behavior.limsim import LimSimConfig
from tactics2d.behavior.limsim.action import LimSimAction
from tactics2d.behavior.limsim.frenet_planner import (
    FrenetTrajectoryPlanner,
    _lane_width,
    _QuarticPolynomial,
    _QuinticPolynomial,
    reference_path_from_agent,
)
from tactics2d.behavior.limsim.schema import AgentDecisionState
from tactics2d.map.element import Lane, Map, Regulatory, RoadLine
from tactics2d.map.query import StopTarget

HALF_WIDTH = 1.75
FORWARD = np.pi / 2
HORIZON = 5


def _lane(lane_id, x, y_start, y_end):
    """Build a lane along +y at ``x`` with its centerline in ``custom_tags``."""
    return Lane(
        id_=lane_id,
        left_side=LineString([(x + HALF_WIDTH, y_start), (x + HALF_WIDTH, y_end)]),
        right_side=LineString([(x - HALF_WIDTH, y_start), (x - HALF_WIDTH, y_end)]),
        custom_tags={"centerline": np.array([[x, y_start], [x, y_end]], dtype=float)},
    )


def _lane_x(lane_id, y, x_start, x_end):
    """Build a lane along +x at ``y``."""
    return Lane(
        id_=lane_id,
        left_side=LineString([(x_start, y + HALF_WIDTH), (x_end, y + HALF_WIDTH)]),
        right_side=LineString([(x_start, y - HALF_WIDTH), (x_end, y - HALF_WIDTH)]),
        custom_tags={"centerline": np.array([[x_start, y], [x_end, y]], dtype=float)},
    )


def _planning_map():
    """Lane A feeding lane B, flanked by a left and a right neighbour lane."""
    map_ = Map(name="planning")
    map_.add_lane(_lane("A", 0.0, 0.0, 100.0))
    map_.add_lane(_lane("B", 0.0, 100.0, 200.0))
    map_.add_lane(_lane("L", -3.6, 0.0, 100.0))
    map_.add_lane(_lane("R", 3.6, 0.0, 100.0))
    map_.lanes["A"].successors = {"B"}
    map_.lanes["B"].predecessors = {"A"}
    map_.lanes["A"].left_neighbors = {"L"}
    map_.lanes["A"].right_neighbors = {"R"}
    map_.lanes["L"].right_neighbors = {"A"}
    map_.lanes["R"].left_neighbors = {"A"}
    return map_


def _config(horizon_steps=HORIZON, dt=0.1, **overrides):
    """Build a short-horizon planning configuration."""
    return replace(LimSimConfig(), horizon_steps=horizon_steps, dt=dt, **overrides)


def _planner(config=None):
    """Build a planner on a short horizon."""
    return FrenetTrajectoryPlanner(config if config is not None else _config())


def _agent(
    x=0.0,
    y=10.0,
    heading=FORWARD,
    speed=5.0,
    lane_id="A",
    route_lane_ids=("A", "B"),
    agent_id=0,
    **kwargs,
):
    """Build an agent state on lane A."""
    return AgentDecisionState(
        agent_id=agent_id,
        x=x,
        y=y,
        heading=heading,
        speed=speed,
        lane_id=lane_id,
        route_lane_ids=route_lane_ids,
        **kwargs,
    )


def _small_agent(x=0.0, y=0.0, heading=FORWARD, speed=0.0, agent_id=1, **kwargs):
    """Build a compact agent whose footprint never touches its neighbour's."""
    return AgentDecisionState(
        agent_id=agent_id, x=x, y=y, heading=heading, speed=speed, length=1.0, width=1.0, **kwargs
    )


def _close_left_boundary(map_, lane_id="A"):
    """Add a roadline that forbids crossing the left boundary of ``lane_id``."""
    line_id = f"{lane_id}_left_closed"
    map_.add_roadline(
        RoadLine(
            id_=line_id,
            geometry=LineString([(HALF_WIDTH, 0.0), (HALF_WIDTH, 100.0)]),
            lane_change=(False, False),
        )
    )
    map_.lanes[lane_id].line_ids = {"left": [line_id]}


def _traffic_light(lane_id="A", stop_point=(0.0, 13.0), state="GO", light_id="tl"):
    """Build a traffic light bound to a lane with a single dynamic state."""
    return Regulatory(
        id_=light_id,
        subtype="traffic_light",
        position=Point(stop_point[0], stop_point[1]),
        custom_tags={
            "lane_id": lane_id,
            "states": [{"time_ms": 1000, "state": state, "stop_point": stop_point}],
        },
    )


def _stop_sign(lane_id="A", point=(0.0, 13.0), sign_id="ss"):
    """Build a stop sign bound to a lane, positioned as the parsers position it."""
    return Regulatory(
        id_=sign_id,
        subtype="stop_sign",
        position=Point(point[0], point[1]),
        custom_tags={"lane_id": lane_id},
    )


@pytest.mark.integration
def test_quintic_polynomial_matches_its_boundary_conditions():
    """The quintic honours position, speed and acceleration at both ends."""
    polynomial = _QuinticPolynomial(0.0, 1.0, 0.0, 10.0, 2.0, 0.0, 2.0)

    assert polynomial.calculate(0.0) == pytest.approx(0.0)
    assert polynomial.calculate(0.0, order=1) == pytest.approx(1.0)
    assert polynomial.calculate(0.0, order=2) == pytest.approx(0.0)
    assert polynomial.calculate(2.0) == pytest.approx(10.0)
    assert polynomial.calculate(2.0, order=1) == pytest.approx(2.0)
    assert polynomial.calculate(2.0, order=2) == pytest.approx(0.0)
    assert np.isfinite(polynomial.calculate(1.0, order=3))


@pytest.mark.integration
def test_quartic_polynomial_matches_its_boundary_conditions():
    """The quartic fixes the initial pose and the terminal speed and acceleration."""
    polynomial = _QuarticPolynomial(0.0, 1.0, 0.0, 3.0, 0.0, 2.0)

    assert polynomial.calculate(0.0) == pytest.approx(0.0)
    assert polynomial.calculate(0.0, order=1) == pytest.approx(1.0)
    assert polynomial.calculate(0.0, order=2) == pytest.approx(0.0)
    assert polynomial.calculate(2.0, order=1) == pytest.approx(3.0)
    assert polynomial.calculate(2.0, order=2) == pytest.approx(0.0)
    assert np.isfinite(polynomial.calculate(1.0, order=3))


@pytest.mark.integration
@pytest.mark.parametrize("action", list(LimSimAction))
def test_plan_returns_one_state_per_horizon_step(action):
    """Every action produces a finite fixed-horizon trajectory on the route."""
    planner = _planner()

    states = planner.plan(_agent(), action, _planning_map())

    assert len(states) == HORIZON
    assert all(state.route_lane_ids == ("A", "B") for state in states)
    assert all(
        np.isfinite([state.x, state.y, state.heading, state.speed]).all() for state in states
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("action", "expected_offset", "expected_lane"),
    [
        (LimSimAction.KS, 0.0, "A"),
        (LimSimAction.AC, 0.0, "A"),
        (LimSimAction.DC, 0.0, "A"),
        (LimSimAction.LCL, None, "L"),
        (LimSimAction.LCR, None, "R"),
    ],
)
def test_plan_offsets_lateral_actions_into_the_target_lane(action, expected_offset, expected_lane):
    """A lane change crosses into the neighbour lane, other actions stay put."""
    planner = _planner()

    states = planner.plan(_agent(), action, _planning_map())

    assert states[-1].lane_id == expected_lane
    if expected_offset is None:
        assert abs(states[-1].lateral_offset) > HALF_WIDTH
        assert np.sign(states[-1].lateral_offset) == (1.0 if action == LimSimAction.LCL else -1.0)
    else:
        assert states[-1].lateral_offset == pytest.approx(expected_offset)


@pytest.mark.integration
@pytest.mark.parametrize(("lane_id", "x"), [(None, 0.0), ("ghost", 0.0), ("A", 10.0)])
def test_plan_falls_back_to_the_lane_follower(lane_id, x):
    """Without a usable reference path the lane follower produces the rollout."""
    planner = _planner()
    map_ = _planning_map()
    agent = _agent(x=x, lane_id=lane_id)

    states = planner.plan(agent, LimSimAction.KS, map_)

    assert states == planner.fallback.rollout(agent, LimSimAction.KS, map_)


@pytest.mark.integration
def test_plan_falls_back_when_no_candidate_is_available():
    """A lane change refused by the map leaves the lane follower to plan."""
    map_ = _planning_map()
    _close_left_boundary(map_)
    planner = _planner()
    agent = _agent()

    states = planner.plan(agent, LimSimAction.LCL, map_)

    assert states == planner.fallback.rollout(agent, LimSimAction.LCL, map_)


@pytest.mark.integration
def test_plan_falls_back_on_a_zero_step_horizon():
    """A horizon of no steps samples no candidate, so planning falls back."""
    map_ = _planning_map()
    planner = _planner(_config(horizon_steps=0))
    agent = _agent()
    path = reference_path_from_agent(agent, map_, planner.config)

    assert planner.sample_candidates(agent, LimSimAction.KS, path, map_) == []
    assert planner.plan(agent, LimSimAction.KS, map_) == planner.fallback.rollout(
        agent, LimSimAction.KS, map_
    )


@pytest.mark.integration
def test_plan_holds_a_stationary_agent_in_place():
    """A stopped agent keeps its pose and its heading over the whole horizon."""
    planner = _planner()

    states = planner.plan(_agent(speed=0.0), LimSimAction.KS, _planning_map())

    assert all(state.speed == 0.0 for state in states)
    assert all(state.x == pytest.approx(0.0) for state in states)
    assert all(state.y == pytest.approx(10.0) for state in states)
    assert all(state.heading == pytest.approx(FORWARD) for state in states)


@pytest.mark.integration
def test_plan_handles_a_single_step_horizon():
    """A one-step horizon returns one state without a look-ahead step."""
    planner = _planner(_config(horizon_steps=1))

    states = planner.plan(_agent(), LimSimAction.KS, _planning_map())

    assert len(states) == 1
    assert states[0].heading == pytest.approx(FORWARD)


@pytest.mark.integration
def test_plan_brakes_for_a_stop_sign():
    """A stop sign ahead makes the planned trajectory decelerate below the sign speed."""
    map_ = _planning_map()
    map_.add_regulatory(_stop_sign(point=(0.0, 13.0)))
    planner = _planner()

    states = planner.plan(_agent(speed=5.0), LimSimAction.KS, map_)

    assert states[-1].speed < 5.0
    assert states[-1].route_progress <= 13.0


@pytest.mark.integration
def test_plan_stands_still_when_every_candidate_collides():
    """An obstacle every rollout runs into is avoided by the stopping candidate."""
    map_ = _planning_map()
    obstacle = [_agent(y=16.0, speed=0.0, agent_id=1, lane_id="A") for _ in range(HORIZON)]
    planner = _planner()

    states = planner.plan(_agent(), LimSimAction.KS, map_, obstacle_trajectories=[obstacle])

    assert all(state.speed == 0.0 for state in states)
    assert states[-1].y == pytest.approx(10.0)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("lane_id", "route_lane_ids", "expected"),
    [("A", ("A", "B"), ("A", "B")), ("B", ("A", "B"), ("B",)), ("A", (), ("A", "B"))],
)
def test_reference_path_from_agent_selects_the_route(lane_id, route_lane_ids, expected):
    """The route is the agent's own suffix, or the topology walk without one."""
    path = reference_path_from_agent(
        _agent(lane_id=lane_id, route_lane_ids=route_lane_ids), _planning_map(), _config()
    )

    assert path.lane_ids == expected


@pytest.mark.integration
def test_reference_path_from_agent_caps_the_route_length():
    """At most ``max_routes_per_agent`` lanes end up in the path."""
    path = reference_path_from_agent(_agent(), _planning_map(), _config(max_routes_per_agent=1))

    assert path.lane_ids == ("A",)


@pytest.mark.integration
def test_reference_path_from_agent_needs_a_known_lane():
    """An absent map, an absent lane id or an unknown lane yields no path."""
    config = _config()
    map_ = _planning_map()

    assert reference_path_from_agent(_agent(), None, config) is None
    assert reference_path_from_agent(_agent(lane_id=None), map_, config) is None
    assert reference_path_from_agent(_agent(lane_id="ghost"), map_, config) is None


@pytest.mark.integration
def test_reference_path_from_agent_returns_none_without_centerlines():
    """Lanes that carry no centerline leave nothing to concatenate."""
    map_ = _planning_map()
    map_.add_lane(Lane(id_="empty"))

    assert reference_path_from_agent(_agent(lane_id="empty"), map_, _config()) is None


@pytest.mark.integration
def test_reference_path_from_agent_walks_past_a_cyclic_topology():
    """A successor that is already on the route ends the walk."""
    map_ = _planning_map()
    map_.lanes["B"].successors = {"A"}

    path = reference_path_from_agent(_agent(), map_, _config())

    assert path.lane_ids == ("A", "B")


@pytest.mark.integration
def test_reference_path_from_agent_walks_past_a_cycle_for_a_foreign_route():
    """A cycle ends the walk even when the route names other lanes."""
    map_ = _planning_map()
    map_.lanes["B"].successors = {"A"}

    path = reference_path_from_agent(_agent(route_lane_ids=("W", "V")), map_, _config())

    assert path.lane_ids == ("A", "B")


@pytest.mark.integration
def test_reference_path_from_agent_walks_past_a_cycle_without_a_route():
    """A cycle ends the walk that starts from the agent's lane."""
    map_ = _planning_map()
    map_.lanes["B"].successors = {"A"}

    path = reference_path_from_agent(_agent(route_lane_ids=()), map_, _config())

    assert path.lane_ids == ("A", "B")


@pytest.mark.integration
def test_reference_path_from_agent_prefers_a_requested_successor():
    """A route that does not start on the agent's lane still steers the walk."""
    map_ = _planning_map()
    map_.add_lane(_lane("Z", 5.0, 100.0, 200.0))
    map_.lanes["A"].successors = {"B", "Z"}

    path = reference_path_from_agent(_agent(route_lane_ids=("Z", "W")), map_, _config())

    assert path.lane_ids == ("A", "Z")


@pytest.mark.integration
def test_reference_path_from_agent_reuses_the_cache():
    """A cached route is returned without rebuilding the centerline."""
    map_ = _planning_map()
    config = _config()
    cache = {}

    first = reference_path_from_agent(_agent(), map_, config, cache=cache)
    second = reference_path_from_agent(_agent(), map_, config, cache=cache)

    assert (id(map_), ("A", "B")) in cache
    assert first.lane_ids == second.lane_ids == ("A", "B")
    assert second.path.equals(first.path)


@pytest.mark.integration
def test_reference_path_from_agent_follows_a_reversed_heading():
    """A route that opposes the agent heading is reversed before use."""
    path = reference_path_from_agent(_agent(heading=-FORWARD), _planning_map(), _config())

    assert path.path.coords[0] == pytest.approx((0.0, 200.0))


@pytest.mark.integration
@pytest.mark.parametrize(
    ("action", "expected_count"), [(LimSimAction.KS, 3), (LimSimAction.LCL, 9)]
)
def test_sample_candidates_covers_speeds_and_lateral_offsets(action, expected_count):
    """Longitudinal actions sample speeds only, lane changes add lateral offsets."""
    planner = _planner()
    map_ = _planning_map()
    path = reference_path_from_agent(_agent(), map_, planner.config)

    candidates = planner.sample_candidates(_agent(), action, path, map_)

    assert len(candidates) == expected_count
    assert all(np.isfinite(candidate.cost) for candidate in candidates)


@pytest.mark.integration
@pytest.mark.parametrize("lane_id", ["solo", "blocked"])
def test_sample_candidates_refuses_a_forbidden_lane_change(lane_id):
    """A lane change is dropped when the neighbour lane is missing or closed."""
    map_ = _planning_map()
    map_.add_lane(_lane("solo", 0.0, 0.0, 100.0))
    map_.add_lane(_lane("blocked", 0.0, 0.0, 100.0))
    map_.lanes["blocked"].left_neighbors = {"L"}
    _close_left_boundary(map_, "blocked")
    planner = _planner()
    agent = _agent(lane_id=lane_id)
    path = reference_path_from_agent(agent, map_, planner.config)

    assert planner.sample_candidates(agent, LimSimAction.LCL, path, map_) == []


@pytest.mark.integration
def test_sample_lateral_offsets_follows_the_action():
    """The lateral samples span the configured offsets around the nominal one."""
    planner = _planner()

    assert planner._sample_lateral_offsets(0.4, LimSimAction.KS) == [0.4]
    assert planner._sample_lateral_offsets(0.0, LimSimAction.LCL) == list(
        planner.config.frenet_lateral_offsets
    )
    assert planner._sample_lateral_offsets(3.5, LimSimAction.LCL) == pytest.approx([3.2, 3.5, 3.8])


@pytest.mark.integration
def test_target_lateral_offset_reads_the_neighbour_lane():
    """The lane change targets the centre of the requested neighbour lane."""
    planner = _planner()
    map_ = _planning_map()
    path = reference_path_from_agent(_agent(), map_, planner.config)

    assert (
        planner._target_lateral_offset(_agent(lateral_offset=0.7), LimSimAction.KS, map_, path)
        == 0.7
    )
    assert planner._target_lateral_offset(_agent(), LimSimAction.LCL, None, path) == 0.0
    assert (
        planner._target_lateral_offset(_agent(lane_id="ghost"), LimSimAction.LCL, map_, path) == 0.0
    )
    assert planner._target_lateral_offset(_agent(), LimSimAction.LCL, map_, path) == pytest.approx(
        3.5
    )
    assert planner._target_lateral_offset(_agent(), LimSimAction.LCR, map_, path) == pytest.approx(
        -3.5
    )

    map_.lanes["A"].left_neighbors = {"ghost"}

    assert planner._target_lateral_offset(_agent(), LimSimAction.LCL, map_, path) == pytest.approx(
        3.5
    )


@pytest.mark.integration
def test_target_lateral_offset_without_a_neighbour_lane():
    """A lane with no neighbour on the requested side has no offset to aim for."""
    map_ = _planning_map()
    map_.add_lane(_lane("solo", 0.0, 0.0, 100.0))
    planner = _planner()
    agent = _agent(lane_id="solo")
    path = reference_path_from_agent(agent, map_, planner.config)

    assert planner._target_lateral_offset(agent, LimSimAction.LCL, map_, path) == 0.0


@pytest.mark.integration
def test_lane_id_for_offset_crosses_the_boundary_midway():
    """Halfway into the neighbour lane the state reports the neighbour lane id."""
    planner = _planner()
    map_ = _planning_map()
    agent = _agent()

    assert planner._lane_id_for_offset(agent, 3.0, LimSimAction.KS, map_) == "A"
    assert planner._lane_id_for_offset(agent, 3.0, LimSimAction.LCL, None) == "A"
    assert (
        planner._lane_id_for_offset(_agent(lane_id="ghost"), 3.0, LimSimAction.LCL, map_) == "ghost"
    )
    assert planner._lane_id_for_offset(agent, 0.5, LimSimAction.LCL, map_) == "A"
    assert planner._lane_id_for_offset(agent, 3.0, LimSimAction.LCL, map_) == "L"
    assert planner._lane_id_for_offset(agent, -3.0, LimSimAction.LCR, map_) == "R"

    map_.lanes["A"].right_neighbors = set()

    assert planner._lane_id_for_offset(agent, -3.0, LimSimAction.LCR, map_) == "A"


@pytest.mark.integration
def test_lane_width_falls_back_to_the_default():
    """A missing lane reports the configured default width."""
    map_ = _planning_map()

    assert _lane_width(None, 3.6) == 3.6
    assert _lane_width(map_.lanes["A"], 3.6) == pytest.approx(2 * HALF_WIDTH)
    assert _lane_width(Lane(id_="empty"), 3.6) == 3.6


@pytest.mark.integration
def test_cost_penalises_collisions_and_proximity():
    """Overlapping boxes are charged the collision penalty, near ones the buffer cost."""
    planner = _planner()
    ego = [_small_agent()]

    def cost_at(distance):
        obstacle = [_small_agent(y=distance, agent_id=1)]
        return planner._cost(ego, 0.0, 0.0, 0.0, 0.0, [obstacle])

    assert cost_at(0.4) >= planner.config.frenet_collision_penalty
    assert cost_at(1.2) == pytest.approx(30.0 * (2.0 - 1.2) ** 2)
    assert cost_at(1.8) == pytest.approx(30.0 * (2.0 - 1.8) ** 2)
    assert cost_at(4.0) == 0.0


@pytest.mark.integration
def test_cost_accepts_precomputed_obstacle_footprints():
    """The pre-computed footprints produce the same cost as computing them inline."""
    planner = _planner()
    ego = [_small_agent()]
    obstacle = [_small_agent(y=0.4, agent_id=1)]
    footprints = planner._precompute_obstacle_footprints([obstacle])

    with_footprints = planner._cost(
        ego, 0.0, 0.0, 0.0, 0.0, [obstacle], obstacle_footprints=footprints
    )
    without = planner._cost(ego, 0.0, 0.0, 0.0, 0.0, [obstacle], obstacle_footprints=None)

    assert with_footprints == pytest.approx(without)
    assert with_footprints >= planner.config.frenet_collision_penalty

    empty_cache = planner._cost(ego, 0.0, 0.0, 0.0, 0.0, [obstacle], obstacle_footprints=[])

    assert empty_cache == pytest.approx(without)

    empty_obstacle, filled_obstacle = planner._precompute_obstacle_footprints([[], obstacle])

    assert empty_obstacle == []
    assert len(filled_obstacle) == len(obstacle)


@pytest.mark.integration
def test_cost_of_an_empty_trajectory_is_zero():
    """An empty trajectory has nothing to score."""
    planner = _planner()

    assert planner._cost([], 0.0, 0.0, 0.0, 0.0, []) == 0.0


@pytest.mark.integration
def test_cost_ignores_empty_obstacle_trajectories():
    """Obstacle slots without states contribute nothing."""
    planner = _planner()

    assert planner._cost([_small_agent()], 0.0, 0.0, 0.0, 0.0, [[], []]) == 0.0


@pytest.mark.integration
def test_has_collision_reports_overlapping_footprints():
    """Only obstacles whose boxes actually overlap count as a collision."""
    planner = _planner()
    ego = [_small_agent()]

    assert planner._has_collision(ego, [[_small_agent(y=0.4, agent_id=1)]]) is True
    assert planner._has_collision(ego, [[_small_agent(y=4.0, agent_id=1)]]) is False
    assert planner._has_collision(ego, [[], []]) is False
    assert (
        planner._has_collision(ego, [[_small_agent(y=4.0, agent_id=1)]], obstacle_footprints=[[]])
        is False
    )


@pytest.mark.integration
def test_stop_candidate_holds_the_agent_still():
    """The stopping candidate repeats the current pose at zero speed."""
    planner = _planner()

    candidate = planner._stop_candidate(_agent(), LimSimAction.KS, [])

    assert len(candidate.states) == HORIZON
    assert all(state.speed == 0.0 for state in candidate.states)
    assert all(state.x == pytest.approx(0.0) for state in candidate.states)
    assert candidate.cost == 0.0


@pytest.mark.integration
def test_stop_candidate_ignores_an_unknown_lane_change_side():
    """A lane change without a neighbour lane is refused before planning."""
    planner = _planner()

    assert planner._lane_change_is_allowed(_agent(), LimSimAction.KS, None, 10.0) is True
    assert planner._lane_change_is_allowed(_agent(), LimSimAction.LCL, None, 10.0) is False
    assert (
        planner._lane_change_is_allowed(_agent(lane_id=None), LimSimAction.LCL, Map(), 10.0)
        is False
    )
    assert (
        planner._lane_change_is_allowed(_agent(), LimSimAction.LCL, _planning_map(), 10.0) is True
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("reason", "state", "expected"),
    [
        ("stop_sign", None, True),
        ("traffic_light", "red", True),
        ("traffic_light", "LANE_STATE_STOP", True),
        ("traffic_light", "green", False),
        ("traffic_light", None, False),
        ("virtual", "red", False),
    ],
)
def test_target_requires_stop_by_reason_and_state(reason, state, expected):
    """Stop signs always stop the agent; traffic lights only in a stop state."""
    planner = _planner()
    target = StopTarget(
        lane_id="A", point=Point(0.0, 10.0), reason=reason, source_id="x", state=state
    )

    assert planner._target_requires_stop(target) is expected


@pytest.mark.integration
def test_nearest_required_stop_target_picks_the_closest_one_ahead():
    """Only stop targets inside the planning window are reported."""
    map_ = _planning_map()
    map_.add_regulatory(_stop_sign(point=(0.0, 14.0), sign_id="ss_far"))
    map_.add_regulatory(_stop_sign(point=(0.0, 13.0), sign_id="ss_near"))
    planner = _planner()
    agent = _agent()
    path = reference_path_from_agent(agent, map_, planner.config)

    target, target_s = planner._nearest_required_stop_target(agent, path, map_)

    assert target.source_id == "ss_near"
    assert target_s == pytest.approx(13.0)


@pytest.mark.integration
def test_nearest_required_stop_target_reads_the_traffic_light_state():
    """A traffic light head only stops the agent while it shows a stop state."""
    map_ = _planning_map()
    map_.add_regulatory(_traffic_light(state="GO"))
    planner = _planner()
    agent = _agent()
    path = reference_path_from_agent(agent, map_, planner.config)

    assert planner._nearest_required_stop_target(agent, path, map_) is None

    map_.regulations["tl"].custom_tags["states"][0]["state"] = "STOP"

    assert planner._nearest_required_stop_target(agent, path, map_) is not None


@pytest.mark.integration
def test_nearest_required_stop_target_skips_distant_and_passed_targets():
    """A target beyond the window or already behind the agent is ignored."""
    map_ = _planning_map()
    map_.add_regulatory(_stop_sign(point=(0.0, 50.0)))
    planner = _planner()
    map_none = planner._nearest_required_stop_target(
        _agent(), reference_path_from_agent(_agent(), map_, planner.config), None
    )

    assert map_none is None
    assert (
        planner._nearest_required_stop_target(
            _agent(), reference_path_from_agent(_agent(), map_, planner.config), map_
        )
        is None
    )

    passed = _planning_map()
    passed.add_regulatory(_stop_sign(point=(0.0, 5.0)))

    assert (
        planner._nearest_required_stop_target(
            _agent(), reference_path_from_agent(_agent(), passed, planner.config), passed
        )
        is None
    )


@pytest.mark.integration
def test_stop_target_candidate_decelerates_to_the_target():
    """The stopping candidate approaches the stop line and stops on it."""
    map_ = _planning_map()
    map_.add_regulatory(_stop_sign(point=(0.0, 13.0)))
    planner = _planner()
    agent = _agent()
    path = reference_path_from_agent(agent, map_, planner.config)
    info = planner._nearest_required_stop_target(agent, path, map_)
    start = path.cartesian_to_frenet(agent.x, agent.y)

    candidate = planner._stop_target_candidate(agent, LimSimAction.KS, path, start, info, [], map_)

    speeds = [state.speed for state in candidate.states]
    assert speeds == sorted(speeds, reverse=True)
    assert speeds[-1] < agent.speed
    assert candidate.states[-1].route_progress <= 13.0


@pytest.mark.integration
def test_stop_target_candidate_holds_the_agent_when_the_target_is_reached():
    """A target inside the stop buffer clamps the rollout to a standstill."""
    map_ = _planning_map()
    map_.add_regulatory(_stop_sign(point=(0.0, 10.5)))
    planner = _planner()
    agent = _agent()
    path = reference_path_from_agent(agent, map_, planner.config)
    info = planner._nearest_required_stop_target(agent, path, map_)
    start = path.cartesian_to_frenet(agent.x, agent.y)

    candidate = planner._stop_target_candidate(agent, LimSimAction.KS, path, start, info, [], map_)

    assert all(state.speed == 0.0 for state in candidate.states)
    assert all(state.route_progress == pytest.approx(10.0) for state in candidate.states)
    assert candidate.cost > 0.0


@pytest.mark.integration
def test_stop_rule_cost_charges_for_passing_the_stop_line():
    """States past the stop line at speed are charged; stopped ones are not."""
    map_ = _planning_map()
    map_.add_regulatory(_stop_sign(point=(0.0, 13.0)))
    planner = _planner()
    path = reference_path_from_agent(_agent(), map_, planner.config)

    crossing = [_agent(y=11.0, speed=3.0), _agent(y=20.0, speed=3.0)]
    before = [_agent(y=11.0, speed=3.0)]
    stopped = [_agent(y=11.0, speed=0.0), _agent(y=20.0, speed=0.0)]

    assert planner._stop_rule_cost(crossing, path, map_) == pytest.approx(
        3000.0 * (1.0 + 3.0 - 0.5)
    )
    assert planner._stop_rule_cost(before, path, map_) == 0.0
    assert planner._stop_rule_cost(stopped, path, map_) == 0.0


@pytest.mark.integration
def test_stop_rule_cost_without_a_target_or_a_state():
    """No stop target, or no state to score, costs nothing."""
    planner = _planner()
    map_ = _planning_map()
    path = reference_path_from_agent(_agent(), map_, planner.config)

    assert planner._stop_rule_cost([_agent(y=20.0, speed=3.0)], path, map_) == 0.0
    assert planner._stop_rule_cost([], path, map_) == 0.0


@pytest.mark.integration
def test_junction_conflict_cost_charges_a_simultaneous_arrival():
    """Two agents reaching a conflict point together are both charged the penalty."""
    map_ = _planning_map()
    map_.add_lane(_lane_x("X", 12.0, -50.0, 50.0))
    planner = _planner()
    agent = _agent()
    path = reference_path_from_agent(agent, map_, planner.config)
    ego = planner.plan(agent, LimSimAction.KS, map_)
    obstacle = [_agent(y=12.0, speed=0.0, agent_id=1, lane_id="X") for _ in range(HORIZON)]

    cost = planner._junction_conflict_cost(ego, path, map_, [obstacle])
    cached = planner._junction_conflict_cost(
        ego,
        path,
        map_,
        [obstacle],
        conflict_points_cache=planner._build_conflict_cache(path, map_, [obstacle]),
    )

    assert cost > 0.0
    assert cached == pytest.approx(cost)


@pytest.mark.integration
def test_junction_conflict_cost_ignores_unrelated_agents():
    """Empty obstacles, lane-less obstacles and distant arrivals cost nothing."""
    map_ = _planning_map()
    map_.add_lane(_lane_x("X", 12.0, -50.0, 50.0))
    planner = _planner()
    agent = _agent()
    path = reference_path_from_agent(agent, map_, planner.config)
    ego = planner.plan(agent, LimSimAction.KS, map_)
    far = [_agent(y=12.0, speed=0.0, agent_id=1, lane_id="ghost") for _ in range(HORIZON)]
    lane_less = [_agent(y=12.0, speed=0.0, agent_id=2, lane_id=None) for _ in range(HORIZON)]

    assert planner._junction_conflict_cost(ego, path, map_, []) == 0.0
    assert planner._junction_conflict_cost([], path, map_, [far]) == 0.0
    assert planner._junction_conflict_cost(ego, path, map_, [[]]) == 0.0
    assert planner._junction_conflict_cost(ego, path, map_, [lane_less]) == 0.0
    assert planner._junction_conflict_cost(ego, path, map_, [far]) == 0.0


@pytest.mark.integration
def test_build_conflict_cache_skips_unusable_obstacles():
    """Only obstacles with a lane on the route contribute cache entries."""
    map_ = _planning_map()
    map_.add_lane(_lane_x("X", 12.0, -50.0, 50.0))
    planner = _planner()
    path = reference_path_from_agent(_agent(), map_, planner.config)
    obstacle = [_agent(y=12.0, speed=0.0, agent_id=1, lane_id="X")]
    lane_less = [_agent(y=12.0, speed=0.0, agent_id=2, lane_id=None)]

    assert planner._build_conflict_cache(path, None, [obstacle]) == {}
    assert planner._build_conflict_cache(path, map_, []) == {}
    assert planner._build_conflict_cache(path, map_, [[], lane_less]) == {}

    cache = planner._build_conflict_cache(path, map_, [obstacle, obstacle])

    assert set(cache) == {("A", "X"), ("B", "X")}
    assert cache[("A", "X")]
