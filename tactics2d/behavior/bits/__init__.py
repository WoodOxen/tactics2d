# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""BITS-style bi-level imitation behavior model (inference)."""

from .config import BitsConfig
from .model import BitsBehaviorModel

__all__ = ["BitsBehaviorModel", "BitsConfig"]
