# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the shared behavior-model batch interface."""

import inspect
import logging
import subprocess
import sys

import pytest

from tactics2d.behavior import BehaviorModelBase


def test_public_module_and_configs_do_not_import_torch():
    """The public facade and pure configs remain usable without model extras."""
    script = """
import sys

class BlockTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith(('torch.', 'torchvision')):
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, BlockTorch())
import tactics2d.behavior
from tactics2d.behavior import BehaviorModelBase, BitsConfig, SmartConfig
assert 'torch' not in sys.modules
assert BehaviorModelBase.__name__ == 'BehaviorModelBase'
assert BitsConfig().future_steps == 20
assert SmartConfig().future_steps == 80
"""

    subprocess.run([sys.executable, "-c", script], check=True)


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
