# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Discrete behavior actions for the LimSim-style MCT planner."""

from enum import Enum


class LimSimAction(str, Enum):
    """High-level actions used by the interaction planner.

    The value strings follow the original LimSim implementation:
    ``KS`` keeps the current longitudinal speed, ``AC`` accelerates, ``DC``
    decelerates, and ``LCL``/``LCR`` request a lane change.
    """

    KS = "KS"
    AC = "AC"
    DC = "DC"
    LCL = "LCL"
    LCR = "LCR"

    #: Semantic alias — prefer ``KS`` for new code.
    KEEP = "KS"

    @property
    def acceleration(self) -> float:
        """Nominal longitudinal acceleration for this action (m/s²)."""

        if self == LimSimAction.AC:
            return 0.7
        if self == LimSimAction.DC:
            return -0.7
        return 0.0

    @property
    def is_lane_change(self) -> bool:
        """Whether this action requests a lateral lane transition."""

        return self in {LimSimAction.LCL, LimSimAction.LCR}
