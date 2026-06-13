# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Renderer module."""


from .config import COLOR_PALETTE, DEFAULT_COLOR, DEFAULT_ORDER
from .matplotlib.renderer import MatplotlibRenderer
from .pygame.manager import RenderManager
from .pygame.renderer import PygameRenderer

__all__ = [
    "MatplotlibRenderer",
    "RenderManager",
    "PygameRenderer",
    "COLOR_PALETTE",
    "DEFAULT_COLOR",
    "DEFAULT_ORDER",
]
