# Copyright (C) 2022, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Urban environment implementation."""

import gym

# from tactics2d.map.element import Map


class UrbanEnv(gym.Env):
    def __init__(self, render_mode: str = "human"):
        self.render_mode = render_mode
