# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the pygame renderer compatibility layer."""


import os

import numpy as np
import pygame


def test_pygame_render_manager_smoke(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")

    from tactics2d.renderer import RenderManager

    class DummySensor:
        id_ = 0
        off_screen = False
        window_size = (16, 16)

        def __init__(self):
            self.surface = pygame.Surface(self.window_size)
            self.bind_id = None

        def set_bind_id(self, bind_id):
            self.bind_id = bind_id

        def update(self, participants, participant_ids, frame=None, position=None, heading=None):
            self.surface.fill((255, 255, 255))

        def get_observation(self):
            return np.zeros((16, 16, 3), dtype=np.uint8)

    manager = RenderManager(fps=60, windows_size=(32, 32), off_screen=True)
    try:
        sensor = DummySensor()
        manager.add_sensor(sensor)
        manager.update({}, [], 0)
        manager.render()
        assert len(manager.get_observation()) == 1
        assert manager.graphic_driver == os.environ["SDL_VIDEODRIVER"]
    finally:
        manager.close()
