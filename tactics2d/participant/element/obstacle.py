# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Obstacle implementation."""


import numpy as np

from tactics2d.participant.trajectory import State

from .other import Other


class Obstacle(Other):
    def get_state(self, frame: int = None) -> State:
        frames = np.array(self.trajectory.frames)
        closest_frame = frames[np.abs(frames - frame).argmin()]

        return self.trajectory.get_state(closest_frame)
