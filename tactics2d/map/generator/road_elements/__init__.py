# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Local road element generators.

**Preferred entry points** for ramp generation are :func:`exit_ramp` and
:func:`entrance_ramp`, which accept a ``main_road_type`` argument
(``"freeway"`` or ``"urban"``). The six concrete wrappers
(:func:`freeway_exit_ramp`, :func:`freeway_entrance_ramp`,
:func:`urban_exit_ramp`, :func:`urban_entrance_ramp`) are thin aliases
exported for convenience; prefer the unified interface in new code.
"""

from .fork import fork
from .intersection import intersection
from .lane_adapter import lane_adapter
from .merge import merge
from .one_way import one_way
from .ramp import (
    entrance_ramp,
    exit_ramp,
    freeway_entrance_ramp,
    freeway_exit_ramp,
    urban_entrance_ramp,
    urban_exit_ramp,
)
from .roundabout import roundabout
from .two_way import two_way

__all__ = [
    "one_way",
    "two_way",
    "lane_adapter",
    "fork",
    "merge",
    "entrance_ramp",
    "exit_ramp",
    "freeway_entrance_ramp",
    "freeway_exit_ramp",
    "urban_entrance_ramp",
    "urban_exit_ramp",
    "intersection",
    "roundabout",
]
