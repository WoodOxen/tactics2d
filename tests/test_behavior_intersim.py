# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the InterSim-style relation-driven behavior model."""

import numpy as np
import pytest
from shapely.geometry import LineString

from tactics2d.behavior.base import BehaviorModelBase
from tactics2d.behavior.intersim.config import InterSimConfig
from tactics2d.behavior.intersim.model import InterSimBehaviorModel
from tactics2d.behavior.intersim.relation_geometry import AgentBody, check_body_collision
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
        id_=lane_id,
        left_side=left,
        right_side=right,
        custom_tags={"centerline": centerline},
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


def _trajectory_to_poses(trajectory, length, width):
    """Return a pose array ``[x, y, 0, yaw]`` over the trajectory frames."""
    frames = sorted(trajectory.frames)
    poses = np.zeros((len(frames), 4), dtype=float)
    for index, frame in enumerate(frames):
        state = trajectory.get_state(frame)
        poses[index, :3] = [state.x, state.y, 0.0]
        poses[index, 3] = state.heading
    return poses


def _same_time_overlap_free(trajectories, dims, margin=1.0):
    """Return whether no two trajectories overlap at any equal frame index."""
    agent_ids = list(trajectories)
    for i in range(len(agent_ids)):
        for j in range(i + 1, len(agent_ids)):
            poses_a = _trajectory_to_poses(trajectories[agent_ids[i]], *dims[agent_ids[i]])
            poses_b = _trajectory_to_poses(trajectories[agent_ids[j]], *dims[agent_ids[j]])
            for index in range(min(len(poses_a), len(poses_b))):
                body_a = AgentBody(
                    poses_a[index, 0], poses_a[index, 1], poses_a[index, 3], *dims[agent_ids[i]]
                )
                body_b = AgentBody(
                    poses_b[index, 0], poses_b[index, 1], poses_b[index, 3], *dims[agent_ids[j]]
                )
                if check_body_collision(body_a, body_b, margin):
                    return False
    return True


def _max_speed(trajectory):
    return max(trajectory.get_state(frame).speed for frame in trajectory.frames)


def _end_speed(trajectory):
    return trajectory.get_state(sorted(trajectory.frames)[-1]).speed


# -- scenarios ----------------------------------------------------------------


def test_intersim_model_crossing_yields_to_earlier_vehicle():
    """At an intersection the later vehicle yields to the earlier one."""
    map_ = _cross_map()
    # B (vertical, +y) reaches the crossing first; A (horizontal, +x) arrives later.
    participants = {
        "A": _vehicle("A", -20.0, 0.0, 0.0, 5.0),
        "B": _vehicle("B", 0.0, -20.0, np.pi / 2, 5.5),
    }
    model = InterSimBehaviorModel(InterSimConfig(cruise_speed=8.0))
    result = model.plan(participants, map_, frame=0)

    assert any(edge.influencer == "B" and edge.reactor == "A" for edge in result.relations)
    assert result.actions["A"] == "yield"
    assert result.actions["B"] == "follow"
    assert set(result.trajectories) == {"A", "B"}
    assert _same_time_overlap_free(
        result.trajectories, {"A": (4.8, 1.9), "B": (4.8, 1.9)}
    )
    # The influencer keeps cruising; the reactor slows toward yield_speed_ratio
    # (0.8) of its own speed (5 m/s -> 4 m/s), not below the old 0.6 value.
    assert _end_speed(result.trajectories["A"]) < 4.1
    assert _max_speed(result.trajectories["B"]) == pytest.approx(8.0, abs=1.0)


def test_intersim_model_catch_up_lets_trailing_vehicle_yield():
    """A fast follower behind a slow leader becomes the reactor."""
    map_ = Map(name="intersim_straight")
    map_.add_lane(_horizontal_lane("main"))
    participants = {
        "leader": _vehicle("leader", -35.0, 0.0, 0.0, 3.0),
        "follower": _vehicle("follower", -55.0, 0.0, 0.0, 9.0),
    }
    # A low cruise keeps the leader slow so the faster follower catches it within
    # the horizon under the (now brisk) free acceleration of the leader.
    model = InterSimBehaviorModel(InterSimConfig(cruise_speed=3.0))
    result = model.plan(participants, map_, frame=0)

    assert any(edge.influencer == "leader" and edge.reactor == "follower" for edge in result.relations)
    assert result.actions["follower"] == "yield"
    assert result.actions["leader"] == "follow"
    assert _same_time_overlap_free(
        result.trajectories, {"leader": (4.8, 1.9), "follower": (4.8, 1.9)}
    )
    assert _end_speed(result.trajectories["follower"]) < 7.3


def test_intersim_model_yields_to_pedestrian():
    """A vehicle approaching a pedestrian must slow and stop."""
    participants = {
        "vehicle": _vehicle("vehicle", 0.0, 0.0, 0.0, 5.0),
        "pedestrian": _pedestrian("pedestrian", 12.0, 0.0),
    }
    model = InterSimBehaviorModel()
    result = model.plan(participants, map_=None, frame=0)

    assert any(
        edge.influencer == "pedestrian" and edge.reactor == "vehicle"
        for edge in result.relations
    )
    assert result.actions["vehicle"] == "yield"

    vehicle_trajectory = result.trajectories["vehicle"]
    vehicle_poses = _trajectory_to_poses(vehicle_trajectory, 4.8, 1.9)
    min_gap = min(np.hypot(pose[0] - 12.0, pose[1]) for pose in vehicle_poses)
    assert min_gap >= 2.0
    last_state = vehicle_trajectory.get_state(sorted(vehicle_trajectory.frames)[-1])
    assert last_state.x < 11.0


def test_intersim_predict_matches_behavior_interface():
    """``predict`` returns future trajectories for a constant-velocity agent."""
    participants = {"ego": _vehicle("ego", 0.0, 0.0, 0.0, 5.0)}
    config = InterSimConfig(horizon_steps=25)
    model = InterSimBehaviorModel(config)

    assert isinstance(model, BehaviorModelBase)
    trajectories = model.predict(participants, map_=None, frame=0, agent_ids=["ego"])

    assert set(trajectories) == {"ego"}
    trajectory = trajectories["ego"]
    assert len(trajectory.frames) == config.planning_steps
    first_frame = min(trajectory.frames)
    assert first_frame == pytest.approx(config.step_ms)
    last_state = trajectory.get_state(sorted(trajectory.frames)[-1])
    assert last_state.x >= 5.0 * config.horizon_steps * config.dt
    assert last_state.x <= config.cruise_speed * config.horizon_steps * config.dt + 1e-9
    assert last_state.y == pytest.approx(0.0)

    result = model.plan(participants, map_=None, frame=0)
    assert result.actions["ego"] == "follow"


def test_intersim_config_rejects_learned_options():
    """Enabling a weight-backed predictor is not yet implemented."""
    with pytest.raises(NotImplementedError):
        InterSimBehaviorModel(InterSimConfig(use_relation_model=True))
    with pytest.raises(NotImplementedError):
        InterSimBehaviorModel(InterSimConfig(use_marginal_model=True))

# -- closed-loop replay --------------------------------------------------


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


def _straight_participant(agent_id, x0, speed, steps, length=4.8, width=1.9):
    trajectory = Trajectory(id_=agent_id, fps=10, stable_freq=True)
    for index in range(steps):
        frame = index * 100
        x = x0 + speed * (index * 0.1)
        trajectory.add_state(
            State(frame=frame, x=x, y=0.0, heading=0.0, vx=speed, vy=0.0)
        )
    return Vehicle(agent_id, "vehicle", trajectory=trajectory, length=length, width=width)


def _catch_up_participants(steps=25):
    # A slow leader (ego) ahead of a fast follower that would catch up.
    return {
        "ego": _straight_participant("ego", 0.0, 1.0, steps),
        "car": _straight_participant("car", -8.0, 8.0, steps),
    }


def test_closed_loop_replans_relevant_set_and_stays_clear():
    """The follower is drawn into the relevant set and braked by the loop."""
    config = InterSimConfig(
        dt=0.1,
        scenario_steps=25,
        planning_warmup_steps=2,
        planning_interval=5,
        horizon_steps=14,
    )
    participants = _catch_up_participants()
    result = InterSimBehaviorModel(config).run_closed_loop(participants, _straight_lane_map(), ego_id="ego")

    assert result.ego_id == "ego"
    assert "car" in result.relevant_ids
    assert result.total_agents_controlled >= 2
    assert not result.collided
    assert (result.front_collisions, result.side_collisions, result.rear_collisions) == (0, 0, 0)
    assert result.progress >= 0.0
    assert set(result.poses) == {"ego", "car"}
    for array in result.poses.values():
        assert array.shape == (25, 4)

    # The fast follower is committed to brake well before the leader.
    car = result.poses["car"]
    index = result.end_index
    stop_distance = float(np.hypot(car[index, 0] - car[index - 1, 0], car[index, 1] - car[index - 1, 1]))
    assert stop_distance < 1.0


def test_closed_loop_yield_all_mode_runs():
    """The all-yield relation mode runs without error and stays clear."""
    config = InterSimConfig(
        dt=0.1,
        scenario_steps=25,
        planning_warmup_steps=2,
        planning_interval=5,
        horizon_steps=14,
        relation_mode="yield_all",
    )
    participants = _catch_up_participants()
    result = InterSimBehaviorModel(config).run_closed_loop(participants, _straight_lane_map(), ego_id="ego")

    assert not result.collided
    assert (result.front_collisions, result.side_collisions, result.rear_collisions) == (0, 0, 0)
    assert "car" in result.relevant_ids


def test_closed_loop_rejects_bad_relation_mode():
    with pytest.raises(ValueError):
        InterSimBehaviorModel(InterSimConfig(relation_mode="unknown"))
