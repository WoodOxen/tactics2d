##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: __init__.py
# @Description: Initialize the environment module.
# @Author: Yueyuan Li
# @Version: 1.0.0

from .parking import ParkingEnv
from .racing import RacingEnv

__all__ = ["RacingEnv", "ParkingEnv"]
