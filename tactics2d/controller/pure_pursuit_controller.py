# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pure pursuit controller implementation."""


import numpy as np
from scipy.interpolate import interp1d
from shapely.geometry import LineString, Point

from tactics2d.participant.trajectory.state import State

from .acceleration_controller import AccelerationController
from .controller_base import ControllerBase


class PurePursuitController(ControllerBase):
    """This class implements a pure pursuit controller to output steering and acceleration commands of the vehicle.

    The controller combines lateral control (pure pursuit algorithm) with longitudinal control
    (via an AccelerationController). It tracks a path defined by waypoints and computes steering
    angle to follow a look-ahead point.

    Attributes:
        interval (float, optional): The time interval between the current command and the next.
            The unit is second. The default value is 1.0. It can be adjusted by `update_driving_style`.
    """

    interval = 1.0

    def __init__(self, min_pre_aiming_distance: float = 10.0, target_speed: float = 5.0):
        if min_pre_aiming_distance <= 0:
            raise ValueError("min_pre_aiming_distance must be positive")
        if target_speed < 0:
            raise ValueError("target_speed must be non-negative")
        self.min_pre_aiming_distance = min_pre_aiming_distance

        self._interval_interpolator = self.create_style_interpolator(2.0, 1.0)

        self._longitudinal_control = AccelerationController(target_speed)

    def update_driving_style(self, style_id: float):
        """This method allows to adopt the controller's behavior by adjusting the internal parameters.

        Args:
            style_id (float): The driving style index, typically in range [-1.0, 1.0].
                Values outside this range will be clamped via extrapolation.
        """
        if not isinstance(style_id, (int, float)):
            raise TypeError("style_id must be int or float")
        self._longitudinal_control.update_driving_style(style_id)
        self.interval = self._interval_interpolator(style_id)

    def _lateral_control(self, ego_state: State, pre_aiming_point: Point, wheel_base: float):
        """Compute steering command using pure pursuit algorithm.

        Args:
            ego_state (State): Current state of the ego vehicle.
            pre_aiming_point (Point): Look-ahead point on the path.
            wheel_base (float): Wheelbase of the vehicle in meters.

        Returns:
            float: Steering angle in radians.
        """
        pre_aiming_angle = np.arctan2(
            pre_aiming_point.y - ego_state.y, pre_aiming_point.x - ego_state.x
        )
        distance = np.linalg.norm(
            (pre_aiming_point.y - ego_state.y, pre_aiming_point.x - ego_state.x)
        )
        steering = np.arctan(
            2.0 * wheel_base * np.sin(pre_aiming_angle - ego_state.heading) / distance
        )

        return steering

    def step(self, ego_state: State, waypoints: LineString, wheel_base: float = 2.637, **kwargs):
        """This method outputs the acceleration and steering command based on the current state of the ego vehicle.

        Args:
            ego_state (State): Current state of the ego vehicle.
            waypoints (LineString): Path to follow as a Shapely LineString.
            wheel_base (float, optional): The wheelbase of the ego vehicle in meters.
                Defaults to 2.637 (medium car).
            **kwargs: Additional inputs passed to the longitudinal controller.
                May include front_state (State) for adaptive cruise control.

        Returns:
            steering (float): The steering command for the ego vehicle in radians.
            accel (float): The acceleration command for the ego vehicle in m/s².
        """
        # TODO: set an automatic reference point catcher
        pre_aiming_distance = ego_state.speed * self.interval
        pre_aiming_distance = np.max([pre_aiming_distance, self.min_pre_aiming_distance])
        pre_aiming_point = waypoints.interpolate(pre_aiming_distance)

        _, accel = self._longitudinal_control.step(ego_state, **kwargs)
        steering = self._lateral_control(ego_state, pre_aiming_point, wheel_base)
        return steering, accel
