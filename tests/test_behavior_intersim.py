# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the InterSim-style relation-driven behavior model."""

import json

import numpy as np
import pytest
from shapely.geometry import LineString

from tactics2d.behavior.base import BehaviorModelBase
from tactics2d.behavior.intersim.config import InterSimConfig
from tactics2d.behavior.intersim.model import InterSimBehaviorModel
from tactics2d.map.element import Lane, Map
from tactics2d.participant.element import Pedestrian, Vehicle
from tactics2d.participant.trajectory import State, Trajectory


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


def _straight_lane_map():
    samples = np.linspace(-100.0, 100.0, 81)
    centerline = np.stack([samples, np.zeros_like(samples)], axis=1)
    lane = Lane(
        id_="main",
        left_side=LineString([(x, -1.9) for x in samples]),
        right_side=LineString([(x, 1.9) for x in samples]),
        custom_tags={"centerline": centerline},
    )
    map_ = Map(name="rolling_straight")
    map_.add_lane(lane)
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


def _straight_participant(agent_id, x0, speed, steps, length=4.8, width=1.9):
    trajectory = Trajectory(id_=agent_id, fps=10, stable_freq=True)
    for index in range(steps):
        trajectory.add_state(
            State(
                frame=index * 100,
                x=x0 + speed * (index * 0.1),
                y=0.0,
                heading=0.0,
                vx=speed,
                vy=0.0,
            )
        )
    return Vehicle(agent_id, "vehicle", trajectory=trajectory, length=length, width=width)


def _poses(trajectory):
    """Return the frames and ``[x, y, heading, speed]`` poses of a trajectory."""
    frames = sorted(trajectory.frames)
    poses = np.zeros((len(frames), 4), dtype=float)
    for index, frame in enumerate(frames):
        state = trajectory.get_state(frame)
        poses[index] = [state.x, state.y, state.heading, state.speed]
    return np.asarray(frames, dtype=float), poses


def _trajectory_arrays(trajectories):
    """Flatten trajectories into a ``frames_<agent>`` / ``poses_<agent>`` mapping."""
    arrays = {}
    for agent_id, trajectory in trajectories.items():
        frames, poses = _poses(trajectory)
        arrays[f"frames_{agent_id}"] = frames
        arrays[f"poses_{agent_id}"] = poses
    return arrays


def _is_finite(poses, valid=None):
    """Return whether the pose rows hold finite numbers."""
    rows = np.asarray(poses, dtype=float)
    return bool(np.isfinite(rows if valid is None else rows[valid]).all())


def _dump(runtime_dir, name, arrays, meta=None):
    """Write the arrays (and optional summary) under the test's runtime directory."""
    if arrays:
        np.savez_compressed(str(runtime_dir / f"{name}.npz"), **arrays)
    if meta is not None:
        (runtime_dir / f"{name}.json").write_text(json.dumps(meta, indent=2, default=str))
    return runtime_dir / f"{name}.json" if meta is not None else runtime_dir / f"{name}.npz"


@pytest.mark.integration
def test_intersim_plan_and_predict_contract(runtime_dir):
    """plan() and predict() cover a crossing scene that also holds a pedestrian."""
    map_ = _cross_map()
    participants = {
        "A": _vehicle("A", -20.0, 0.0, 0.0, 5.0),
        "B": _vehicle("B", 0.0, -20.0, np.pi / 2, 5.5),
        "P": _pedestrian("P", 20.0, 0.0),
    }
    config = InterSimConfig(cruise_speed=8.0)
    model = InterSimBehaviorModel(config)

    assert isinstance(model, BehaviorModelBase)
    predicted = model.predict(participants, map_, frame=0, agent_ids=["A", "B"])
    assert set(predicted) == {"A", "B"}
    for trajectory in predicted.values():
        assert isinstance(trajectory, Trajectory)
        assert 1 <= len(trajectory.frames) <= config.planning_steps
        assert _is_finite(_poses(trajectory)[1])

    result = model.plan(participants, map_, frame=0)
    assert {"A", "B"} <= set(result.scene_agent_ids)
    assert set(result.actions) == set(result.scene_agent_ids)
    assert result.trajectories
    assert set(result.trajectories) <= set(result.scene_agent_ids)
    for trajectory in result.trajectories.values():
        assert _is_finite(_poses(trajectory)[1])

    assert _dump(runtime_dir, "intersim_predict", _trajectory_arrays(predicted)).exists()
    assert _dump(
        runtime_dir,
        "intersim_plan",
        {},
        {
            "scene_agent_ids": result.scene_agent_ids,
            "actions": result.actions,
            "n_relations": len(result.relations),
        },
    ).exists()


@pytest.mark.integration
@pytest.mark.slow
def test_intersim_closed_loop_runs_default_cadence(runtime_dir):
    """The closed loop runs a full-length scenario at the model's own cadence."""
    config = InterSimConfig()
    steps = config.scenario_steps
    participants = {
        "ego": _straight_participant("ego", 0.0, 8.0, steps),
        "car": _straight_participant("car", 20.0, 3.0, steps),
    }

    result = InterSimBehaviorModel(config).rollout(participants, _straight_lane_map(), ego_id="ego")

    assert result.ego_id == "ego"
    assert set(result.poses) == {"ego", "car"}
    for poses in result.poses.values():
        assert poses.shape == (steps, 4)
        assert _is_finite(poses, valid=poses[:, 0] != -1.0)
    assert set(result.relevant_ids) <= set(result.poses)
    assert result.total_agents_controlled >= 1
    assert 0 <= result.end_index < steps

    arrays = {f"poses_{key}": value for key, value in result.poses.items()}
    path = _dump(
        runtime_dir,
        "intersim_closed_loop",
        arrays,
        {
            "progress": result.progress,
            "end_index": result.end_index,
            "total_agents_controlled": result.total_agents_controlled,
            "collided": result.collided,
        },
    )
    assert path.exists()


@pytest.mark.integration
def test_intersim_closed_loop_short_scenario(runtime_dir):
    """A short scenario with a dense replanning cadence also runs through."""
    config = InterSimConfig(
        dt=0.1, scenario_steps=25, planning_warmup_steps=2, planning_interval=5, horizon_steps=14
    )
    steps = config.scenario_steps
    participants = {
        "ego": _straight_participant("ego", 0.0, 1.0, steps),
        "car": _straight_participant("car", -8.0, 8.0, steps),
    }

    result = InterSimBehaviorModel(config).rollout(participants, _straight_lane_map(), ego_id="ego")

    assert set(result.poses) == {"ego", "car"}
    for poses in result.poses.values():
        assert poses.shape == (steps, 4)
        assert _is_finite(poses, valid=poses[:, 0] != -1.0)
    assert result.total_agents_controlled >= 1

    arrays = {f"poses_{key}": value for key, value in result.poses.items()}
    path = _dump(
        runtime_dir,
        "intersim_closed_loop_short",
        arrays,
        {
            "progress": result.progress,
            "end_index": result.end_index,
            "total_agents_controlled": result.total_agents_controlled,
            "collided": result.collided,
        },
    )
    assert path.exists()
