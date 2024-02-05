##! python3
# -*- coding: utf-8 -*-
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: single_track_dynamics.py
# @Description: This file implements a dynamic single-track model for a vehicle.
# @Author: Yueyuan Li
# @Version: 1.0.0

import numpy as np

from .physics_model_base import PhysicsModelBase
from tactics2d.participant.trajectory import State


class SingleTrackDynamics(PhysicsModelBase):
    """This class implements a dynamic single-track model for a vehicle. The dynamic single-track model is a simplified model to simulate the vehicle dynamics. It combines the front and rear wheels into a single wheel, and the vehicle is assumed to be a point mass."""

    abbrev = "ST"

    def __init__(
        self, wheel_base: float, steer_range: list, speed_range: list, delta_t: float = 0.01
    ):
        self.wheel_base = wheel_base
        self.speed_range = speed_range
        self.steer_range = steer_range
        self.delta_t = delta_t
