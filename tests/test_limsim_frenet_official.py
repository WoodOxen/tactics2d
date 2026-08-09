# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Official-semantics regressions for LimSim Frenet trajectory selection."""

import numpy as np
import pytest
from shapely.geometry import LineString

import tactics2d.behavior.limsim.frenet_planner as frenet_planner_module
from tactics2d.behavior.limsim import LimSimConfig
from tactics2d.behavior.limsim.action import LimSimAction
from tactics2d.behavior.limsim.frenet_planner import (
    FrenetCandidate,
    FrenetTrajectoryPlanner,
    reference_path_from_agent,
)
from tactics2d.behavior.limsim.schema import AgentDecisionState
from tactics2d.geometry.frenet import ReferencePath
from tactics2d.map.element import Lane, Map


def _state(agent_id, x, y=0.0, speed=5.0):
    return AgentDecisionState(
        agent_id=agent_id,
        x=x,
        y=y,
        heading=0.0,
        speed=speed,
        lane_id="A",
        route_lane_ids=("A",),
        route_progress=x,
    )


def _straight_reference():
    return ReferencePath(
        LineString([(0.0, 0.0), (100.0, 0.0)]),
        lane_ids=("A",),
        lane_width=3.6,
    )


def test_short_obstacle_prediction_is_not_frozen_past_its_horizon():
    planner = FrenetTrajectoryPlanner(LimSimConfig(horizon_steps=2))
    states = [_state("ego", 10.0), _state("ego", 0.0)]
    obstacle_trajectories = ([_state("obstacle", 0.0, speed=0.0)],)
    obstacle_footprints = planner._precompute_obstacle_footprints(obstacle_trajectories)

    baseline_cost = planner._cost(
        states,
        target_speed=5.0,
        nominal_d=0.0,
        accel_cost=0.0,
        jerk_cost=0.0,
        obstacle_trajectories=(),
    )
    obstacle_cost = planner._cost(
        states,
        target_speed=5.0,
        nominal_d=0.0,
        accel_cost=0.0,
        jerk_cost=0.0,
        obstacle_trajectories=obstacle_trajectories,
        obstacle_footprints=obstacle_footprints,
    )

    assert obstacle_cost == pytest.approx(baseline_cost)


def test_collision_free_candidate_is_selected_before_cost(monkeypatch):
    planner = FrenetTrajectoryPlanner(LimSimConfig(horizon_steps=1))
    agent = _state("ego", 0.0)
    colliding = FrenetCandidate(states=[_state("ego", 0.0)], cost=float("inf"))
    safe = FrenetCandidate(states=[_state("ego", 10.0)], cost=10.0)
    obstacle_trajectories = ([_state("obstacle", 0.0, speed=0.0)],)

    monkeypatch.setattr(
        frenet_planner_module,
        "reference_path_from_agent",
        lambda *args, **kwargs: _straight_reference(),
    )
    monkeypatch.setattr(planner, "sample_candidates", lambda *args, **kwargs: [colliding, safe])

    result = planner._plan_action(
        agent,
        LimSimAction.KS,
        map_=None,
        obstacle_trajectories=obstacle_trajectories,
        time_ms=0,
        steps=1,
    )

    assert result == safe.states


def test_all_colliding_candidates_trigger_continuous_braking(monkeypatch):
    config = LimSimConfig(horizon_steps=15, dt=0.1, frenet_stop_deceleration=4.0)
    planner = FrenetTrajectoryPlanner(config)
    agent = _state("ego", 0.0, speed=4.5)
    colliding = FrenetCandidate(states=[_state("ego", 10.0)], cost=float("inf"))
    obstacle_trajectories = ([_state("obstacle", 10.0, speed=0.0)],)

    monkeypatch.setattr(
        frenet_planner_module,
        "reference_path_from_agent",
        lambda *args, **kwargs: _straight_reference(),
    )
    monkeypatch.setattr(planner, "sample_candidates", lambda *args, **kwargs: [colliding])

    result = planner._plan_action(
        agent,
        LimSimAction.KS,
        map_=None,
        obstacle_trajectories=obstacle_trajectories,
        time_ms=0,
        steps=config.horizon_steps,
    )
    speeds = [state.speed for state in result]

    assert 0.0 < speeds[0] < agent.speed
    assert all(next_speed <= speed for speed, next_speed in zip(speeds, speeds[1:]))
    assert result[0].x > agent.x
    assert speeds[-1] == 0.0


def test_lane_keeping_tries_official_nudge_before_emergency_stop(monkeypatch):
    planner = FrenetTrajectoryPlanner(LimSimConfig(horizon_steps=2))
    agent = _state("ego", 0.0)
    blocked = FrenetCandidate(states=[agent], cost=float("inf"))
    nudged = FrenetCandidate(states=[agent.with_updates(y=1.0)], cost=1.0)
    sampled_offsets = []

    monkeypatch.setattr(
        frenet_planner_module,
        "reference_path_from_agent",
        lambda *args, **kwargs: _straight_reference(),
    )

    def sample(*args, **kwargs):
        offsets = kwargs.get("lateral_offsets")
        sampled_offsets.append(offsets)
        return [blocked] if offsets is None else [nudged]

    monkeypatch.setattr(planner, "sample_candidates", sample)

    result = planner._plan_action(
        agent,
        LimSimAction.KS,
        map_=None,
        obstacle_trajectories=([_state("obstacle", 0.0)],),
        time_ms=0,
        steps=2,
    )

    assert result == nudged.states
    assert sampled_offsets[0] is None
    assert sampled_offsets[1]


def test_reference_path_stops_when_topology_leaves_available_lanes():
    map_ = Map(name="available_terminal")
    for lane_id, start_x in (("A", 0.0), ("B", 10.0)):
        map_.add_lane(
            Lane(
                id_=lane_id,
                left_side=LineString([(start_x, 1.0), (start_x + 10.0, 1.0)]),
                right_side=LineString([(start_x, -1.0), (start_x + 10.0, -1.0)]),
                custom_tags={"centerline": np.asarray([(start_x, 0.0), (start_x + 10.0, 0.0)])},
            )
        )
    map_.lanes["A"].successors.add("B")
    map_.lanes["B"].predecessors.add("A")
    agent = _state("ego", 0.0).with_updates(
        route_lane_ids=("A",),
        available_lane_ids=frozenset({"A"}),
    )

    reference = reference_path_from_agent(agent, map_, LimSimConfig())

    assert reference is not None
    assert reference.lane_ids == ("A",)
    assert reference.terminal

    recorded_reference = reference_path_from_agent(
        agent.with_updates(route_lane_ids=("A", "B")),
        map_,
        LimSimConfig(),
    )
    assert recorded_reference is not None
    assert recorded_reference.lane_ids == ("A", "B")


def test_frenet_uses_configured_action_acceleration():
    planner = FrenetTrajectoryPlanner(LimSimConfig(acceleration=1.2, deceleration=-1.4))

    assert planner._action_acceleration(LimSimAction.AC) == pytest.approx(1.2)
    assert planner._action_acceleration(LimSimAction.DC) == pytest.approx(-1.4)
    assert planner._action_acceleration(LimSimAction.KS) == 0.0


def test_clear_road_uses_persistent_target_speed_after_emergency_stop(monkeypatch):
    config = LimSimConfig(horizon_steps=30, dt=0.1)
    planner = FrenetTrajectoryPlanner(config)
    agent = _state("ego", 0.0, speed=0.0)

    monkeypatch.setattr(
        frenet_planner_module,
        "reference_path_from_agent",
        lambda *args, **kwargs: _straight_reference(),
    )

    result = planner._plan_action(
        agent,
        LimSimAction.KS,
        map_=None,
        obstacle_trajectories=(),
        time_ms=0,
        steps=config.horizon_steps,
    )

    assert result[0].speed > 0.0
    assert result[-1].speed > result[0].speed


def test_official_speed_samples_keep_candidate_speed_separate_from_cost_target(monkeypatch):
    planner = FrenetTrajectoryPlanner(LimSimConfig(horizon_steps=2))
    agent = _state("ego", 0.0, speed=0.0)
    cost_targets = []

    def capture_cost(states, target_speed, *args, **kwargs):
        cost_targets.append(target_speed)
        return 0.0

    monkeypatch.setattr(planner, "_cost", capture_cost)
    candidates = planner.sample_candidates(
        agent,
        LimSimAction.KS,
        _straight_reference(),
        map_=None,
    )

    assert len(candidates) == 5
    assert all(target == pytest.approx(agent.target_speed) for target in cost_targets)
