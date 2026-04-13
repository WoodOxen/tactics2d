# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Renderer module."""


from .matplotlib_config import COLOR_PALETTE, DEFAULT_COLOR, DEFAULT_ORDER
from .matplotlib_renderer import MatplotlibRenderer

# from .webgl_renderer import WebGLRenderer

__all__ = ["MatplotlibRenderer", "COLOR_PALETTE", "DEFAULT_COLOR", "DEFAULT_ORDER"]
