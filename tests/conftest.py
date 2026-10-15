# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Conftest implementation."""

import os
from pathlib import Path

import pytest

from tactics2d.dataset_parser import WOMDParser
from tactics2d.participant.element import Vehicle

# Set environment variables before test modules import model and rendering code.
os.environ["MPLBACKEND"] = "Agg"
os.environ["PYTHON_SKIP_TKINTER"] = "1"
# The default test suite is device-independent and must exercise the CPU path
# even on developer machines or self-hosted runners that expose a GPU.  Tests
# that explicitly validate CUDA can opt in from a separate process.
os.environ["CUDA_VISIBLE_DEVICES"] = ""


@pytest.fixture
def runtime_dir(request):
    """Return a module-scoped output directory under tests/runtime/.

    Each test module gets its own subdirectory named after the module file
    (e.g. tests/runtime/test_map_generator/). The directory is created if it
    does not exist.
    """
    module_name = Path(request.module.__file__).stem
    dir_path = Path(__file__).parent / "runtime" / module_name
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


@pytest.fixture(scope="session")
def womd_behavior_scene():
    """Load a small shared vehicle scene from the WOMD parser fixture."""

    sample_root = Path(__file__).parent.parent / "tactics2d/data/trajectory_sample/WOMD"
    shard = "uncompressed_scenario_validation_validation.tfrecord-00001-of-00150"
    if not (sample_root / shard).is_file():
        pytest.skip(f"WOMD behavior sample is not present under {sample_root}.")

    parser = WOMDParser()
    participants, _ = parser.parse_trajectory(0, file=shard, folder=str(sample_root))
    map_ = parser.parse_map(0, file=shard, folder=str(sample_root))
    controlled_ids = [
        agent_id
        for agent_id, participant in participants.items()
        if isinstance(participant, Vehicle)
        and participant.tracks_to_predict
        and len(participant.trajectory.frames) >= 91
    ][:3]
    if len(controlled_ids) < 3:
        pytest.fail("The WOMD behavior sample must contain three complete target vehicles.")

    return ({agent_id: participants[agent_id] for agent_id in controlled_ids}, map_, controlled_ids)


def pytest_addoption(parser):
    """Register optional local test-suite switches."""

    parser.addoption(
        "--run-bits-workflow",
        action="store_true",
        default=False,
        help="run BITS training/evaluation workflow tests",
    )


def pytest_configure(config):
    """Configure matplotlib backend before any tests run."""

    config.addinivalue_line(
        "markers", "bits_workflow: mark BITS training/evaluation workflow tests"
    )
    try:
        import matplotlib

        # Force Agg backend even if matplotlib was already imported
        matplotlib.use("Agg", force=True)
        # Also set rcParams to ensure no GUI backend is used
        matplotlib.rcParams["backend"] = "Agg"
    except ImportError:
        pass


def pytest_collection_modifyitems(config, items):
    """Skip resource-dependent BITS workflow tests unless explicitly requested."""

    if config.getoption("--run-bits-workflow"):
        return
    skip_workflow = pytest.mark.skip(reason="BITS workflow tests require --run-bits-workflow")
    for item in items:
        if "bits_workflow" in item.keywords:
            item.add_marker(skip_workflow)
