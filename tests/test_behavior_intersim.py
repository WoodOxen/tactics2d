# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the InterSim-style relation-driven behavior model."""

from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString

from tactics2d.behavior import InterSimBehaviorModel, InterSimConfig
from tactics2d.map.element import Lane, Map
from tactics2d.participant.element import Pedestrian, Vehicle
from tactics2d.participant.trajectory import State, Trajectory

RELATION_CHECKPOINT = (
    Path(__file__).resolve().parent.parent
    / "tactics2d/data/checkpoints/intersim/relation_predictor_waymo.pt"
)


def _lane(lane_id, start, end, width=1.9):
    """Build a straight lane from ``start`` to ``end`` along an axis."""
    (x0, y0), (x1, y1) = start, end
    if x0 == x1:
        left = LineString([(x0 - width, y0), (x0 - width, y1)])
        right = LineString([(x0 + width, y0), (x0 + width, y1)])
    else:
        left = LineString([(x0, y0 - width), (x1, y1 - width)])
        right = LineString([(x0, y0 + width), (x1, y1 + width)])
    samples = np.linspace(0.0, 1.0, 41)
    centerline = np.asarray(start) + np.outer(samples, np.asarray(end) - np.asarray(start))
    return Lane(
        id_=lane_id, left_side=left, right_side=right, custom_tags={"centerline": centerline}
    )


def _horizontal_lane(lane_id, y=0.0, x_start=-80.0, x_end=80.0):
    return _lane(lane_id, (x_start, y), (x_end, y))


def _vertical_lane(lane_id, x=0.0, y_start=-80.0, y_end=80.0):
    return _lane(lane_id, (x, y_start), (x, y_end))


def _cross_map():
    map_ = Map(name="intersim_cross")
    map_.add_lane(_horizontal_lane("horizontal"))
    map_.add_lane(_vertical_lane("vertical"))
    return map_


def _participant(participant_type, agent_id, x, y, heading, speed, frame=0):
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
    return participant_type(agent_id, "vehicle", trajectory=trajectory, length=4.8, width=1.9)


def _vehicle(agent_id, x, y, heading, speed):
    return _participant(Vehicle, agent_id, x, y, heading, speed)


def _pedestrian(agent_id, x, y, heading=0.0):
    return _participant(Pedestrian, agent_id, x, y, heading, 0.0)


def _moving_participant(agent_id, x0, y0, heading, speed, steps):
    trajectory = Trajectory(id_=agent_id, fps=10, stable_freq=True)
    for index in range(steps):
        distance = speed * index * 0.1
        trajectory.add_state(
            State(
                frame=index * 100,
                x=x0 + distance * np.cos(heading),
                y=y0 + distance * np.sin(heading),
                heading=heading,
                vx=speed * np.cos(heading),
                vy=speed * np.sin(heading),
            )
        )
    return Vehicle(agent_id, "vehicle", trajectory=trajectory, length=4.8, width=1.9)


@pytest.mark.integration
def test_intersim_predict_runs():
    """The integrated prediction entry point runs on CPU."""
    map_ = _cross_map()
    participants = {
        "A": _vehicle("A", -20.0, 0.0, 0.0, 5.0),
        "B": _vehicle("B", 0.0, -20.0, np.pi / 2, 5.5),
        "P": _pedestrian("P", 20.0, 0.0),
    }
    config = InterSimConfig(cruise_speed=8.0)
    model = InterSimBehaviorModel(config)

    predicted = model.predict(participants, map_, frame=0, agent_ids=["A", "B"])
    assert set(predicted) == {"A", "B"}
    assert all(isinstance(trajectory, Trajectory) for trajectory in predicted.values())


@pytest.mark.integration
@pytest.mark.slow
def test_intersim_rollout_runs():
    """The checkpoint-backed integrated closed loop runs on CPU."""
    if not RELATION_CHECKPOINT.exists():
        pytest.skip(f"Released InterSim checkpoint is not present at {RELATION_CHECKPOINT}.")
    config = InterSimConfig(
        relation_mode="nn",
        relation_model_path=str(RELATION_CHECKPOINT),
    )
    steps = config.scenario_steps
    participants = {
        "ego": _moving_participant("ego", -20.0, 0.0, 0.0, 5.0, steps),
        "car": _moving_participant("car", 0.0, -20.0, np.pi / 2, 5.0, steps),
    }

    result = InterSimBehaviorModel(config).rollout(participants, _cross_map(), ego_id="ego")

    assert result.ego_id == "ego"
    assert set(result.poses) == {"ego", "car"}
