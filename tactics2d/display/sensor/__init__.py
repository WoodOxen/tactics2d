# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Sensor module.

Provides sensor computation classes (BEVCamera, SingleLineLidar) that
produce structured geometry_data dicts consumed by display renderers.
"""

from .camera import BEVCamera
from .lidar import SingleLineLidar
from .sensor_base import SensorBase

__all__ = ["SensorBase", "BEVCamera", "SingleLineLidar"]
