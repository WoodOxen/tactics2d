##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: __init__.py
# @Description: Initialize the environment module.
# @Author: Yueyuan Li
# @Version: 0.1.8rc1

from .parking import ParkingEnv
from .racing import RacingEnv

__all__ = ["RacingEnv", "ParkingEnv"]
