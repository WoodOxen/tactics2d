import numpy as np

from .physics_model_base import PhysicsModelBase
from tactics2d.trajectory.element import State


class SingleTrackDynamics(PhysicsModelBase):
    """Implementation of the dynamic single-track (bicycle) Model."""

    abbrev = "ST"

    def __init__(
        self, wheel_base: float, steer_range: list, speed_range: list, delta_t: float = 0.01
    ):
        self.wheel_base = wheel_base
        self.speed_range = speed_range
        self.steer_range = steer_range
        self.delta_t = delta_t
