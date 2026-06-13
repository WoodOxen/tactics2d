# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Conftest implementation."""


import os
from pathlib import Path

import pytest

# Set environment variables before any imports
os.environ["MPLBACKEND"] = "Agg"
os.environ["PYTHON_SKIP_TKINTER"] = "1"


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
