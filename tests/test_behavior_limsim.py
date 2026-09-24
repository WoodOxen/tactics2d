# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the LimSim behavior model (public API only)."""

import numpy as np
import pytest
from shapely.geometry import LineString

from tactics2d.behavior import BehaviorModelBase, LimSimBehaviorModel, LimSimConfig
from tactics2d.map.element import Lane, Map
from tactics2d.participant.element import Vehicle
from tactics2d.participant.trajectory import State, Trajectory


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
    map_.add_lane(_lane("A", 0.0, 2.0, 0.0, 80.0))
    map_.add_lane(_lane("B", 2.0, 4.0, 0.0, 80.0))
    return map_


def _vehicle(agent_id, frames, x, y, heading=np.pi / 2, speed=5.0):
    """Build a vehicle moving at ``speed`` along ``heading`` over ``frames``."""
    trajectory = Trajectory(id_=agent_id, fps=10, stable_freq=True)
    for index, frame in enumerate(frames):
        travelled = speed * 0.1 * index
        trajectory.add_state(
            State(
                frame=frame,
                x=x + travelled * np.cos(heading),
                y=y + travelled * np.sin(heading),
                heading=heading,
                vx=speed * np.cos(heading),
                vy=speed * np.sin(heading),
            )
        )
    return Vehicle(agent_id, "vehicle", trajectory=trajectory, length=4.5, width=1.8)


@pytest.mark.integration
def test_limsim_plan_runs():
    """The public model-specific planning entry point runs."""
    config = LimSimConfig(horizon_steps=6, dt=0.2, mcts_iterations=20, interaction_distance=15.0)
    map_ = _parallel_map()
    participants = {
        1: _vehicle(1, [0], 1.0, 5.0, speed=5.0),
        2: _vehicle(2, [0], 3.0, 10.0, speed=2.0),
    }

    result = LimSimBehaviorModel(config).plan(
        participants, map_, frame=0, route_map={1: ("A",), 2: ("B",)}
    )

    assert set(result.actions) == {1, 2}
    assert set(result.trajectories) == {1, 2}


@pytest.mark.integration
def test_limsim_predict_runs():
    """The shared prediction entry point runs."""
    config = LimSimConfig(horizon_steps=6, dt=0.2, mcts_iterations=20, interaction_distance=15.0)
    participants = {1: _vehicle(1, [0], 3.0, 5.0, speed=5.0)}
    model = LimSimBehaviorModel(config)

    assert isinstance(model, BehaviorModelBase)
    predicted = model.predict(
        participants, _parallel_map(), frame=0, agent_ids=[1], route_map={1: ("B",)}
    )

    assert set(predicted) == {1}
    assert isinstance(predicted[1], Trajectory)


@pytest.mark.integration
def test_limsim_rollout_runs():
    """The public closed-loop entry point runs on recorded participants."""
    config = LimSimConfig(horizon_steps=10, dt=0.1, mcts_iterations=50, interaction_distance=20.0)
    map_ = _parallel_map()
    participants = {
        1: _vehicle(1, range(0, 2100, 100), 1.0, 0.0, speed=5.0),
        2: _vehicle(2, range(0, 2100, 100), 3.0, 20.0, speed=3.0),
    }
    model = LimSimBehaviorModel(config)
    route_map = {1: ("A",), 2: ("B",)}

    result = model.rollout(
        participants, map_, ego_id=1, frame_ms=1000, horizon_ms=1000, route_map=route_map
    )

    assert result.cycles > 0
    assert isinstance(result.trajectory, Trajectory)
