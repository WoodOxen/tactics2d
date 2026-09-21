# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the BITS receding-horizon replay."""

import numpy as np
import pytest

pytest.importorskip("torch", reason="BITS rolling tests require the tactics2d[behavior] extra.")

from tactics2d.behavior.bits.config import BitsConfig
from tactics2d.behavior.bits.rolling import BitsRollingRunner
from tactics2d.participant.element import Vehicle
from tactics2d.participant.trajectory import State, Trajectory

STEP_MS = 100
SPEED = 10.0
TAKE_OVER = 1000
HORIZON_MS = 500


def _config():
    """Build a BITS configuration on the 100 ms lattice."""
    return BitsConfig()


def _ego_trajectory(agent_id=0, heading=0.0):
    """Build a straight, constant-speed trajectory from 0 to 2000 ms."""
    trajectory = Trajectory(id_=agent_id, fps=1000.0 / STEP_MS, stable_freq=True)
    for frame in range(0, 2001, STEP_MS):
        trajectory.add_state(
            State(
                frame=frame,
                x=SPEED * frame / 1000.0,
                y=0.0,
                heading=heading,
                vx=SPEED * np.cos(heading),
                vy=SPEED * np.sin(heading),
            )
        )
    return trajectory


def _vehicle(agent_id=0, heading=0.0):
    """Build a vehicle carrying the straight trajectory."""
    return Vehicle(
        agent_id,
        "vehicle",
        trajectory=_ego_trajectory(agent_id, heading=heading),
        length=4.5,
        width=1.8,
    )


class _StubPlanner:
    """Stand-in for the BITS model that replans a straight, shifted track."""

    def __init__(self, config, shift=5.0, steps=5, period_ms=STEP_MS, speed=SPEED, plan_for=None):
        self.config = config
        self.shift = shift
        self.steps = steps
        self.period_ms = period_ms
        self.speed = speed
        self.plan_for = plan_for
        self.calls = []

    def predict(self, participants, map_, frame, agent_ids=None):
        """Return one plan per requested agent, issued at *frame*."""
        self.calls.append(frame)
        plans = {}
        for agent_id in agent_ids or ():
            if self.plan_for is not None and agent_id not in self.plan_for:
                continue
            plan = Trajectory(id_=agent_id, fps=1000.0 / self.config.step_ms, stable_freq=True)
            for index in range(1, self.steps + 1):
                plan_frame = frame + index * self.period_ms
                plan.add_state(
                    State(
                        frame=plan_frame,
                        x=self.shift + self.speed * plan_frame / 1000.0,
                        y=0.0,
                        heading=0.0,
                    )
                )
            plans[agent_id] = plan
        return plans


def _runner(horizon_ms=None, replan_interval=20, **stub_kwargs):
    """Build a runner driven by the stub planner."""
    config = _config()
    model = _StubPlanner(config, **stub_kwargs)
    return (
        BitsRollingRunner(model, config, horizon_ms=horizon_ms, replan_interval=replan_interval),
        model,
    )


@pytest.mark.integration
def test_runner_rejects_a_replan_interval_below_one():
    """A cycle has to commit at least one step."""
    config = _config()

    with pytest.raises(ValueError, match="replan_interval"):
        BitsRollingRunner(_StubPlanner(config), replan_interval=0)


@pytest.mark.integration
def test_runner_defaults_to_the_model_configuration_and_horizon():
    """The runner borrows the model's config and plans one full horizon per cycle."""
    config = _config()
    model = _StubPlanner(config)

    runner = BitsRollingRunner(model)

    assert runner.config is model.config
    assert runner.horizon_ms == config.planning_steps * STEP_MS
    assert runner.replan_interval == 20
    assert BitsRollingRunner(model, config, horizon_ms=250).horizon_ms == 250


@pytest.mark.integration
def test_run_rejects_controlled_ids_without_the_ego():
    """Re-simulating everyone but the ego is a caller error."""
    runner, _ = _runner()

    with pytest.raises(ValueError, match="must contain ego_id"):
        runner.run({0: _vehicle(0)}, None, ego_id=0, controlled_ids=[1])


@pytest.mark.integration
def test_run_rejects_a_vehicle_without_a_recorded_trajectory():
    """A vehicle with no frames cannot anchor a take-over."""
    runner, _ = _runner()
    participants = {0: Vehicle(0, "vehicle", trajectory=Trajectory(id_=0, fps=10.0))}

    with pytest.raises(ValueError, match="no recorded trajectory"):
        runner.run(participants, None, ego_id=0)


@pytest.mark.integration
def test_run_rejects_a_take_over_frame_past_the_track():
    """A take-over frame after the last recorded one leaves nothing to replay."""
    runner, _ = _runner()

    with pytest.raises(ValueError, match="no frame at or after"):
        runner.run({0: _vehicle(0)}, None, ego_id=0, frame_ms=9_999_999)


@pytest.mark.integration
def test_run_replays_the_ego_over_the_planning_horizon():
    """The replayed future lands on the recorded grid, shifted by the new plan."""
    runner, model = _runner(horizon_ms=HORIZON_MS)
    participants = {0: _vehicle(0)}

    result = runner.run(participants, None, ego_id=0)

    assert model.calls == [TAKE_OVER]
    assert result.cycles == 1
    assert result.controlled_ids == [0]
    assert result.frames == list(range(0, TAKE_OVER + HORIZON_MS + 1, STEP_MS))
    assert set(result.plans) == {TAKE_OVER}
    assert [frame for frame, _, _ in result.plans[TAKE_OVER]] == [1100, 1200, 1300, 1400, 1500]
    assert result.trajectory.get_state(TAKE_OVER).x == pytest.approx(10.0)
    assert result.trajectory.get_state(1100).x == pytest.approx(16.0)
    assert result.trajectory.get_state(1500).x == pytest.approx(20.0)


@pytest.mark.integration
def test_run_takes_over_at_the_requested_frame():
    """An explicit take-over frame bounds the history that is kept."""
    runner, model = _runner(horizon_ms=HORIZON_MS)

    result = runner.run({0: _vehicle(0)}, None, ego_id=0, frame_ms=1200)

    assert model.calls == [1200]
    assert result.frames == list(range(0, 1201, STEP_MS)) + list(range(1300, 1701, STEP_MS))


@pytest.mark.integration
def test_run_replans_until_the_horizon_is_covered():
    """A short cycle commits part of a plan and re-plans from its last frame."""
    runner, model = _runner(horizon_ms=HORIZON_MS, replan_interval=2)

    result = runner.run({0: _vehicle(0)}, None, ego_id=0)

    assert model.calls == [1000, 1200, 1400]
    assert result.cycles == 3
    assert list(result.plans) == [1000, 1200, 1400]
    assert result.trajectory.get_state(1500).x == pytest.approx(20.0)


@pytest.mark.integration
def test_run_interpolates_a_sparse_plan_onto_the_recorded_grid():
    """Plan waypoints further apart than the lattice are interpolated onto it."""
    runner, _ = _runner(horizon_ms=HORIZON_MS, period_ms=200)

    result = runner.run({0: _vehicle(0)}, None, ego_id=0)

    assert result.trajectory.get_state(1300).x == pytest.approx(18.0)
    assert result.trajectory.get_state(1500).x == pytest.approx(20.0)


@pytest.mark.integration
def test_run_keeps_the_recorded_heading_for_a_plan_that_stands_still():
    """A plan that does not move the ego leaves its heading untouched."""
    runner, _ = _runner(horizon_ms=HORIZON_MS, speed=0.0)

    result = runner.run({0: _vehicle(0, heading=1.0)}, None, ego_id=0)

    replayed = [
        state for frame, state in result.trajectory.history_states.items() if frame > TAKE_OVER
    ]
    assert len(replayed) == HORIZON_MS // STEP_MS
    assert all(state.heading == pytest.approx(1.0) for state in replayed)


@pytest.mark.integration
def test_run_stops_when_the_model_predicts_no_future_state():
    """A plan without a frame past the take-over ends the replay at the history."""
    runner, model = _runner(horizon_ms=HORIZON_MS, steps=0)

    result = runner.run({0: _vehicle(0)}, None, ego_id=0)

    assert model.calls == [TAKE_OVER]
    assert result.cycles == 0
    assert result.plans == {}
    assert sorted(result.trajectory.frames) == list(range(0, TAKE_OVER + 1, STEP_MS))


@pytest.mark.integration
def test_run_stops_when_the_model_omits_the_ego():
    """A prediction without the ego ends the replay before any cycle."""
    runner, model = _runner(horizon_ms=HORIZON_MS, plan_for=())

    result = runner.run({0: _vehicle(0)}, None, ego_id=0)

    assert model.calls == [TAKE_OVER]
    assert result.cycles == 0
    assert sorted(result.trajectory.frames) == list(range(0, TAKE_OVER + 1, STEP_MS))


@pytest.mark.integration
def test_run_restores_every_controlled_vehicle():
    """A controlled vehicle without a plan keeps its recording, as the ego does."""
    runner, _ = _runner(horizon_ms=HORIZON_MS, plan_for=(0,))
    participants = {0: _vehicle(0), 1: _vehicle(1)}

    result = runner.run(participants, None, ego_id=0, controlled_ids=[0, 1])

    assert result.controlled_ids == [0, 1]
    assert result.cycles == 1
    assert sorted(participants[0].trajectory.frames) == list(range(0, 2001, STEP_MS))
    assert sorted(participants[1].trajectory.frames) == list(range(0, 2001, STEP_MS))


@pytest.mark.integration
def test_run_with_a_zero_horizon_returns_the_recorded_history():
    """A horizon of no steps replays nothing and keeps the warm-up window."""
    runner, model = _runner(horizon_ms=0)

    result = runner.run({0: _vehicle(0)}, None, ego_id=0)

    assert model.calls == []
    assert result.cycles == 0
    assert result.frames == list(range(0, TAKE_OVER + 1, STEP_MS))
    assert sorted(result.trajectory.frames) == list(range(0, TAKE_OVER + 1, STEP_MS))
