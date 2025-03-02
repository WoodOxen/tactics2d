##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: pure_pursuit_controller.py
# @Description:
# @Author: Tactics2D Team
# @Version: 0.18.0

import numpy as np
from scipy.interpolate import interp1d
from shapely.geometry import LineString, Point

from tactics2d.participant.trajectory.state import State

from .acceleration_controller import AccelerationController


class PurePursuitController:
    """This class implements a pure pursuit controller to output steeringing and acceleration commands of the vehicle.

    Attributes:
        kp (float): The proportional gain for speed error adjustment. The default value is 3.5. It can be adjusted by `update_driving_style`.
        accel_change_rate (float): The limitation to how quickly the acceleration can change over time to ensure smooth transitions. The unit is $m^2$/s. The default value is 3.0. It can be adjusted by `update_driving_style`.
        max_accel (float): The upper limit of the acceleration. The unit is $m^2$/s. The default value is 1.5. It can be adjusted by `update_driving_style`.
        min_accel (float): The lower limit of the acceleration. When negative, it describes the upper limit of the deceleration.The unit is $m^2$/s. The default value is -4.0. It can be adjusted by `update_driving_style`.
        interval (float, optional): The time interval between the current command and the next. The unit is second. The default value is 1.0. It can be adjusted by `update_driving_style`.
    """

    kp = 3.5
    accel_change_rate = 3.0
    max_accel = 1.5
    min_accel = -4.0
    interval = 1.0

    def __init__(self):
        self._kp_interpolator = interp1d(
            [-1.0, 1.0], [4.5, 2.5], kind="linear", bounds_error=False, fill_value=(4.5, 2.5)
        )
        self._accel_change_rate_interpolator = interp1d(
            [-1.0, 1.0], [2.0, 6.0], kind="linear", bounds_error=False, fill_value=(2.0, 6.0)
        )
        self._max_accel_interpolator = interp1d(
            [-1.0, 1.0], [1.5, 2.5], kind="linear", bounds_error=False, fill_value=(1.5, 2.5)
        )
        self._min_accel_interpolator = interp1d(
            [-1.0, 1.0], [-3.0, -5.0], kind="linear", bounds_error=False, fill_value=(-3.0, -5.0)
        )
        self._interval_interpolator = interp1d(
            [-1.0, 1.0], [2.0, 1.0], kind="linear", bounds_error=False, fill_value=(4.0, 2.0)
        )

        self._longitudinal_control = AccelerationController()

    def update_driving_style(self, style_id: int):
        """This method allows to adopt the controller's behavior by adjusting the internal parameters.

        Args:
            style_id (int): The index to seek for a new driving style.
        """
        self._longitudinal_control.update_driving_style(style_id)

        self.kp = self._kp_interpolator(style_id)
        self.accel_change_rate = self._accel_change_rate_interpolator(style_id)
        self.max_accel = self._max_accel_interpolator(style_id)
        self.min_accel = self._min_accel_interpolator(style_id)
        self.interval = self._interval_interpolator(style_id)

    def _lateral_control(self, ego_state: State, pre_aiming_point: Point, wheel_base: float):
        pre_aiming_angle = np.arctan2(
            pre_aiming_point.y - ego_state.y, pre_aiming_point.x - ego_state.x
        )
        # print("preview_angle_",preview_angle_)
        distance = np.linalg.norm(
            (pre_aiming_point.y - ego_state.y, pre_aiming_point.x - ego_state.x)
        )
        # print("dis",dis_2_pp)
        steering = np.arctan(
            2.0 * wheel_base * np.sin(pre_aiming_angle - ego_state.heading) / distance
        )

        return steering

    def step(self, ego_state: State, waypoints: LineString, wheel_base: float = 2.637):
        """This method outputs the acceleration and steering command based on the current state of the ego vehicle.

        Args:
            ego_state (State): _description_
            waypoints (LineString): _description_
            wheel_base (float, optional): The wheelbase of the ego vehicle. The default unit is meter. Defaults to 2.637 (medium_car).

        Returns:
            steering (float): The steering command for the ego vehicle.
            accel (float): The acceleration command for the ego vehicle.
        """
        # TODO: set an automatic reference point catcher
        # self.pre_aiming_distance = self.pre_aiming_interval * ego_state.speed
        # self.pre_aiming_distance = np.max([self.pre_aiming_distance, 5.0])
        pre_aiming_distance = ego_state.speed * self.interval
        pre_aiming_distance = np.max([pre_aiming_distance, 5.0])
        pre_aiming_point = waypoints.interpolate(pre_aiming_distance)

        # print(pre_aiming_point, pre_aiming_point.coords)

        _, accel = self._longitudinal_control.step(ego_state)
        steering = self._lateral_control(ego_state, pre_aiming_point, wheel_base)
        return steering, accel
