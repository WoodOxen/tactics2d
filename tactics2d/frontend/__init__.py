# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Browser-based frontend utilities for Tactics2D."""

from .renderer import FrontendRenderer
from .server import run_server

__all__ = ["FrontendRenderer", "run_server"]
