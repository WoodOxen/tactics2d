# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: conftest.py
# @Description: Pytest configuration file to set matplotlib backend for tests
# @Author: Claude Code
# @Version: 0.1.8rc1

import os
import sys

# Set environment variables before any imports
os.environ["MPLBACKEND"] = "Agg"
os.environ["PYTHON_SKIP_TKINTER"] = "1"


def pytest_configure(config):
    """Configure matplotlib backend before any tests run."""
    try:
        import matplotlib

        # Force Agg backend even if matplotlib was already imported
        matplotlib.use("Agg", force=True)
        # Also set rcParams to ensure no GUI backend is used
        matplotlib.rcParams["backend"] = "Agg"
    except ImportError:
        pass
