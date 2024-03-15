##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: __init__.py
# @Description: Initialize the map generator module.
# @Author: Yueyuan Li
# @Version: 1.0.0


from .generate_parking_lot import ParkingLotGenerator
from .generate_racing_track import RacingTrackGenerator

__all__ = ["RacingTrackGenerator", "ParkingLotGenerator"]
