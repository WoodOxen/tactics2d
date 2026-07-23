# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the LimSim behavior model (public API only)."""

import numpy as np
import pytest
from shapely.geometry import LineString

from tactics2d.behavior.limsim import LimSimBehaviorModel, LimSimConfig
from tactics2d.behavior.limsim.action import LimSimAction
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
    a = _lane("A", 0.0, 2.0, 0.0, 80.0)
    b = _lane("B", 2.0, 4.0, 0.0, 80.0)
    map_.add_lane(a)
    map_.add_lane(b)
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


def test_limsim_plans_for_all_agents():
    """plan() produces actions and trajectories for every controlled agent."""
    config = LimSimConfig(horizon_steps=6, dt=0.2, mcts_iterations=20, interaction_distance=15.0)
    participants = {1: _vehicle(1, 0, 1.0, 5.0, speed=5.0), 2: _vehicle(2, 0, 1.0, 10.0, speed=2.0)}

    result = LimSimBehaviorModel(config).plan(participants, _parallel_map(), route_map={}, frame=0)

    assert set(result.actions) == {1, 2}
    assert set(result.trajectories) == {1, 2}
    assert len(result.trajectories[1].frames) == config.horizon_steps


def test_limsim_decelerates_for_obstacle():
    """A faster agent behind a slower one decelerates (DC) to avoid collision."""
    config = LimSimConfig(horizon_steps=50, mcts_iterations=200, interaction_distance=20.0)
    participants = {1: _vehicle(1, 0, 1.0, 0.0, speed=8.0), 2: _vehicle(2, 0, 1.0, 25.0, speed=3.0)}

    result = LimSimBehaviorModel(config).plan(participants, _parallel_map(), route_map={}, frame=0)

    assert result.actions[1] == LimSimAction.DC
