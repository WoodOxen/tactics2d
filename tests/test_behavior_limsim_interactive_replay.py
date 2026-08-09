# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the reusable LimSim interactive replay controller."""

from types import SimpleNamespace

import pytest

from tactics2d.behavior.limsim import (
    InteractiveReplayController,
    apply_rollout_states,
    restore_recorded_snapshots,
    snapshot_vehicle_trajectories,
)
from tactics2d.behavior.limsim.interactive_replay import collision_pairs_at_frame
from tactics2d.participant.element import Vehicle
from tactics2d.participant.trajectory import State, Trajectory


def _recorded_vehicle(agent_id, start_y, speed, end_frame=1600):
    trajectory = Trajectory(id_=agent_id, fps=12.5, stable_freq=True)
    for frame in range(0, end_frame + 1, 80):
        trajectory.add_state(
            State(
                frame=frame,
                x=0.0,
                y=start_y + speed * frame / 1000.0,
                heading=0.0,
                vx=0.0,
                vy=speed,
            )
        )
    return Vehicle(agent_id, "vehicle", trajectory=trajectory, length=4.5, width=1.8)


def _planned_trajectory(agent_id, frame, state, y_step, points=5):
    trajectory = Trajectory(id_=agent_id, fps=10, stable_freq=True)
    for offset in range(1, points + 1):
        trajectory.add_state(
            State(
                frame=frame + offset * 100,
                x=state.x,
                y=state.y + offset * y_step,
                heading=state.heading,
                vx=0.0,
                vy=y_step / 0.1,
            )
        )
    return trajectory


class _FakeLimSimModel:
    """Return two agents once, then let the controller carry ownership forward."""

    def __init__(self, near_step=5.0, planning_interval=0.5, points=5):
        self.config = SimpleNamespace(step_ms=100, dt=0.1, planning_interval=planning_interval)
        self.calls = 0
        self.call_frames = []
        self.near_step = near_step
        self.points = points

    def plan(self, participants, map_, frame, **kwargs):
        ego_state = participants["ego"].trajectory.get_state(frame)
        trajectories = {
            "ego": _planned_trajectory("ego", frame, ego_state, y_step=0.5, points=self.points),
        }
        roi_agent_ids = ["ego"]
        groups = [["ego"]]
        if self.calls == 0 and "near" in participants:
            near_state = participants["near"].trajectory.get_state(frame)
            trajectories["near"] = _planned_trajectory(
                "near", frame, near_state, y_step=self.near_step, points=self.points
            )
            roi_agent_ids.append("near")
            groups = [["ego", "near"]]
        if "quiet" in participants:
            quiet_state = participants["quiet"].trajectory.get_state(frame)
            trajectories["quiet"] = _planned_trajectory(
                "quiet", frame, quiet_state, y_step=0.2, points=self.points
            )
        if "short" in participants:
            short_state = participants["short"].trajectory.get_state(frame)
            trajectories["short"] = _planned_trajectory(
                "short", frame, short_state, y_step=0.2, points=self.points
            )
        self.calls += 1
        self.call_frames.append(frame)
        return SimpleNamespace(
            trajectories=trajectories,
            roi_agent_ids=roi_agent_ids,
            background_agent_ids=["outer"],
            groups=groups,
        )


def test_interactive_replay_controller_runs_closed_loop_and_preserves_ownership():
    participants = {
        "ego": _recorded_vehicle("ego", start_y=0.0, speed=5.0),
        # Its replay future overlaps the ego plan, while its planned future avoids it.
        "near": _recorded_vehicle("near", start_y=2.0, speed=-3.0),
        "outer": _recorded_vehicle("outer", start_y=60.0, speed=2.0),
        "replay_overlap": _recorded_vehicle("replay_overlap", start_y=60.0, speed=2.0),
        "quiet": _recorded_vehicle("quiet", start_y=30.0, speed=2.0),
    }
    recorded_snapshots = snapshot_vehicle_trajectories(participants)
    route_update_frames = []

    def update_routes(participants, map_, frame, ego_id, outer_radius, route_map):
        route_update_frames.append(frame)
        route_map.setdefault(ego_id, ("lane",))

    model = _FakeLimSimModel()
    controller = InteractiveReplayController(
        model=model,
        participants=participants,
        map_=None,
        ego_id="ego",
        recorded_snapshots=recorded_snapshots,
        roi_radius=20.0,
        roi_outer_radius=100.0,
        route_updater=update_routes,
    )

    result = controller.run(start_frame=0, steps=3)

    assert result.status == "completed"
    assert result.num_cycles == result.expected_cycles == 3
    assert result.owned_vehicle_count == 2
    assert result.avg_controlled_vehicles == pytest.approx(2.0)
    assert result.multi_agent_cycles == 1
    assert result.collision_events == 0
    assert route_update_frames == [0]
    assert model.call_frames == [0]
    assert participants["near"].trajectory.get_state(300).y == pytest.approx(17.0)
    assert participants["outer"].trajectory.get_state(100).y == pytest.approx(60.2)
    assert participants["quiet"].trajectory.get_state(300).y == pytest.approx(30.6)

    restore_recorded_snapshots(participants, recorded_snapshots)
    assert not participants["ego"].trajectory.has_state(100)
    assert participants["ego"].trajectory.frames == list(range(0, 1601, 80))

    apply_rollout_states(participants, result.rollout_states)
    assert participants["ego"].trajectory.get_state(100).y == pytest.approx(0.5)
    assert participants["outer"].trajectory.get_state(100).y == pytest.approx(60.2)
    assert participants["ego"].trajectory.get_state().frame == 300


def test_interactive_replay_controller_reports_route_update_failure():
    participants = {"ego": _recorded_vehicle("ego", start_y=0.0, speed=5.0)}

    def fail_route_update(*args):
        raise RuntimeError("route unavailable")

    controller = InteractiveReplayController(
        model=_FakeLimSimModel(),
        participants=participants,
        map_=None,
        ego_id="ego",
        roi_radius=20.0,
        roi_outer_radius=100.0,
        route_updater=fail_route_update,
    )

    result = controller.run(start_frame=0, steps=1)

    assert result.status == "incomplete"
    assert result.num_cycles == 0
    assert result.error_message == "route update failed: RuntimeError: route unavailable"


def test_collision_metric_uses_vehicle_footprints():
    participants = {
        "first": _recorded_vehicle("first", start_y=0.0, speed=0.0),
        "second": _recorded_vehicle("second", start_y=2.5, speed=0.0),
    }

    assert collision_pairs_at_frame(participants, participants, frame=0) == set()

    participants["second"].trajectory.get_state(0).y = 1.5
    assert collision_pairs_at_frame(participants, participants, frame=0) == {
        frozenset(("first", "second"))
    }


def test_short_recorded_track_forces_takeover():
    participants = {
        "ego": _recorded_vehicle("ego", start_y=0.0, speed=5.0),
        "near": _recorded_vehicle("near", start_y=10.0, speed=3.0),
        "short": _recorded_vehicle("short", start_y=30.0, speed=2.0, end_frame=80),
    }
    controller = InteractiveReplayController(
        model=_FakeLimSimModel(),
        participants=participants,
        map_=None,
        ego_id="ego",
        roi_radius=50.0,
        roi_outer_radius=100.0,
    )

    result = controller.run(start_frame=0, steps=1)

    assert result.status == "completed"
    assert result.owned_vehicle_count == 2
    assert result.avg_controlled_vehicles == pytest.approx(2.0)


def test_owned_vehicle_retires_when_cached_plan_is_exhausted():
    participants = {
        "ego": _recorded_vehicle("ego", start_y=0.0, speed=5.0),
        "near": _recorded_vehicle("near", start_y=2.0, speed=-3.0),
        "outer": _recorded_vehicle("outer", start_y=60.0, speed=2.0),
    }
    controller = InteractiveReplayController(
        model=_FakeLimSimModel(near_step=60.0),
        participants=participants,
        map_=None,
        ego_id="ego",
        roi_radius=20.0,
        roi_outer_radius=100.0,
    )

    result = controller.run(start_frame=0, steps=6)

    assert result.status == "completed"
    assert result.owned_vehicle_count == 2
    assert result.retired_vehicle_count == 1
    assert participants["near"].trajectory.has_state(500)
    assert not participants["near"].trajectory.has_state(600)


def test_interactive_replay_replans_every_five_simulation_steps():
    participants = {"ego": _recorded_vehicle("ego", start_y=0.0, speed=5.0)}
    model = _FakeLimSimModel()
    route_update_frames = []

    def update_routes(participants, map_, frame, ego_id, outer_radius, route_map):
        route_update_frames.append(frame)

    controller = InteractiveReplayController(
        model=model,
        participants=participants,
        map_=None,
        ego_id="ego",
        roi_radius=50.0,
        roi_outer_radius=100.0,
        route_updater=update_routes,
    )

    result = controller.run(start_frame=0, steps=12)

    assert result.status == "completed"
    assert result.num_cycles == 12
    assert model.call_frames == [0, 500, 1000]
    assert route_update_frames == model.call_frames
    assert len(result.plan_times) == 3
    assert sorted(result.ego_plans) == list(range(0, 1200, 100))
    for frame, planned_points in result.ego_plans.items():
        assert planned_points
        assert planned_points[0][1] > participants["ego"].trajectory.get_state(frame).y


def test_interactive_replay_forces_replan_when_cached_trajectory_runs_out():
    participants = {"ego": _recorded_vehicle("ego", start_y=0.0, speed=5.0)}
    model = _FakeLimSimModel(planning_interval=1.0, points=2)
    controller = InteractiveReplayController(
        model=model,
        participants=participants,
        map_=None,
        ego_id="ego",
        roi_radius=50.0,
        roi_outer_radius=100.0,
    )

    result = controller.run(start_frame=0, steps=7)

    assert result.status == "completed"
    assert model.call_frames == [0, 200, 400, 600]


def test_interactive_replay_rejects_non_positive_planning_interval():
    participants = {"ego": _recorded_vehicle("ego", start_y=0.0, speed=5.0)}

    with pytest.raises(ValueError, match="planning_interval"):
        InteractiveReplayController(
            model=_FakeLimSimModel(planning_interval=0.0),
            participants=participants,
            map_=None,
            ego_id="ego",
        )
