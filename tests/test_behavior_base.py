# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the shared behavior-model batch interface."""

import inspect
import logging
import threading

import pytest

pytest.importorskip("torch", reason="Behavior-model tests require the tactics2d[behavior] extra.")

from tactics2d.behavior import BehaviorModelBase


class _StubModel(BehaviorModelBase):
    """Model that marks every requested agent and can fail on chosen frames."""

    def __init__(self, parallel_workers=0, fail_on=()):
        self.parallel_workers = parallel_workers
        self.fail_on = set(fail_on)
        self.calls = []

    def predict(self, participants, map_, frame, agent_ids=None):
        """Record the call and return a marker per requested agent."""
        self.calls.append((frame, None if agent_ids is None else tuple(agent_ids)))
        if frame in self.fail_on:
            raise RuntimeError(f"cannot predict frame {frame}")
        return {agent_id: (frame, agent_id) for agent_id in agent_ids or ()}


class _BarrierModel(_StubModel):
    """Model that blocks until every synchronised frame is running at once."""

    def __init__(self, barrier, frames_to_sync):
        super().__init__(parallel_workers=2)
        self.barrier = barrier
        self.frames_to_sync = set(frames_to_sync)

    def predict(self, participants, map_, frame, agent_ids=None):
        """Wait for the sibling frame before predicting."""
        if frame in self.frames_to_sync:
            self.barrier.wait(timeout=10.0)
        return super().predict(participants, map_, frame, agent_ids=agent_ids)


@pytest.mark.integration
def test_predict_batch_runs_sequentially_by_default():
    """A model that asks for no parallelism predicts every frame in order."""
    model = _StubModel()

    result = model.predict_batch({}, None, [10, 20, 30], agent_ids=[1, 2])

    assert list(result) == [10, 20, 30]
    assert result[20] == {1: (20, 1), 2: (20, 2)}
    assert model.calls == [(10, (1, 2)), (20, (1, 2)), (30, (1, 2))]


@pytest.mark.integration
def test_predict_batch_dispatches_across_workers():
    """A worker count above one still returns one entry per frame."""
    model = _StubModel(parallel_workers=4)

    result = model.predict_batch({}, None, [10, 20, 30, 40], agent_ids=[1])

    assert sorted(result) == [10, 20, 30, 40]
    assert all(entry == {1: (frame, 1)} for frame, entry in result.items())
    assert sorted(model.calls) == [(10, (1,)), (20, (1,)), (30, (1,)), (40, (1,))]


@pytest.mark.integration
def test_predict_batch_runs_sequentially_for_an_explicit_worker_count_of_one():
    """A worker count of one overrides a parallel default and predicts in order."""
    model = _StubModel(parallel_workers=4)

    result = model.predict_batch({}, None, [10, 20, 30], agent_ids=[1], max_workers=1)

    assert list(result) == [10, 20, 30]
    assert model.calls == [(10, (1,)), (20, (1,)), (30, (1,))]


@pytest.mark.integration
def test_predict_batch_overlaps_frames_when_parallel():
    """Two workers really run two frames at once, not one after the other."""
    barrier = threading.Barrier(2)
    model = _BarrierModel(barrier, frames_to_sync=[10, 20])

    result = model.predict_batch({}, None, [10, 20, 30], agent_ids=[1])

    assert result == {10: {1: (10, 1)}, 20: {1: (20, 1)}, 30: {1: (30, 1)}}


@pytest.mark.integration
def test_predict_batch_maps_a_failing_frame_to_an_empty_result(caplog):
    """One frame raising leaves the other frames in the batch intact."""
    model = _StubModel(fail_on=(20,))

    with caplog.at_level(logging.WARNING, logger="tactics2d.behavior.base"):
        result = model.predict_batch({}, None, [10, 20, 30], agent_ids=[1])

    assert result[10] == {1: (10, 1)}
    assert result[20] == {}
    assert result[30] == {1: (30, 1)}
    assert "frame 20" in caplog.text
    assert "RuntimeError" in caplog.text


@pytest.mark.integration
def test_predict_batch_logs_a_failure_from_a_worker_thread(caplog):
    """A frame that fails inside the executor is reported the same way."""
    model = _StubModel(parallel_workers=2, fail_on=(20,))

    with caplog.at_level(logging.WARNING, logger="tactics2d.behavior.base"):
        result = model.predict_batch({}, None, [10, 20], agent_ids=[1])

    assert result == {10: {1: (10, 1)}, 20: {}}
    assert "frame 20" in caplog.text


@pytest.mark.integration
def test_predict_batch_forwards_only_the_shared_arguments():
    """Model-specific options are not forwarded: ``predict`` takes them directly."""
    parameters = inspect.signature(BehaviorModelBase.predict_batch).parameters

    # ``route_map`` and ``decider`` are LimSim's and InterSim's own; forwarding
    # one of them raised ``TypeError`` in the other three models.
    assert list(parameters) == [
        "self",
        "participants",
        "map_",
        "frames",
        "agent_ids",
        "max_workers",
    ]
    assert parameters["agent_ids"].default is None
