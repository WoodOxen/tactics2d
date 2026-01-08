##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: acceleration_controller.py
# @Description:
# @Author: Tactics2D Team
# @Version: 0.1.8rc1

import numpy as np
from scipy.interpolate import interp1d

from tactics2d.participant.trajectory.state import State


class AccelerationController:
    """This class defines a controller that outputs acceleration command of the vehicle. The acceleration is calculated based on the proportional controlled.

    Attributes:
        kp (float): The proportional gain for speed error adjustment. The default value is 3.5. It can be adjusted by `update_driving_style`.
        speed_factor (float): The factor to adjust the target speed based on the driving style. The default value is 1.0. It can be adjusted by `update_driving_style`.
        accel_change_rate (float): The limitation to how quickly the acceleration can change over time to ensure smooth transitions. The unit is $m^2$/s. The default value is 3.0. It can be adjusted by `update_driving_style`.
        max_accel (float): The upper limit of the acceleration. The unit is $m^2$/s. The default value is 1.5. It can be adjusted by `update_driving_style`.
        min_accel (float): The lower limit of the acceleration. When negative, it describes the upper limit of the deceleration.The unit is $m^2$/s. The default value is -4.0. It can be adjusted by `update_driving_style`.
        interval (float, optional): The time interval between the current command and the next. The unit is second. The default value is 2.0. It can be adjusted by `update_driving_style`.
        delta_t (float, optional): The discrete time step for the simulation. The unit is millisecond. The default value is 0.05.
    """

    kp = 3.5
    speed_factor = 1.0
    accel_change_rate = 3.0
    max_accel = 1.5
    min_accel = -4.0
    interval = 2.0
    delta_t = 0.05

    def __init__(self, target_speed: float = 5.0):
        self.target_speed = target_speed

        self._kp_interpolator = interp1d(
            [-1.0, 1.0], [4.5, 2.5], kind="linear", bounds_error=False, fill_value=(4.5, 2.5)
        )
        self._speed_factor_interpolator = interp1d(
            [-1.0, 1.0], [0.8, 1.2], kind="linear", bounds_error=False, fill_value=(0.8, 1.2)
        )
        self._accel_change_rate_interpolator = interp1d(
            [-1.0, 1.0], [3.0, 6.0], kind="linear", bounds_error=False, fill_value=(5.0, 6.0)
        )
        self._max_accel_interpolator = interp1d(
            [-1.0, 1.0], [1.5, 2.5], kind="linear", bounds_error=False, fill_value=(1.5, 2.5)
        )
        self._min_accel_interpolator = interp1d(
            [-1.0, 1.0], [-3.0, -5.0], kind="linear", bounds_error=False, fill_value=(-3.0, -5.0)
        )
        self._interval_interpolator = interp1d(
            [-1.0, 1.0], [3.5, 1.5], kind="linear", bounds_error=False, fill_value=(3.5, 1.5)
        )

    def update_driving_style(self, style_id: int):
        """This method allows to adopt the controller's behavior by adjusting the internal parameters.

        Args:
            style_id (int): The index to seek for a new driving style.
        """
        self.kp = self._kp_interpolator(style_id)
        self.speed_factor = self._speed_factor_interpolator(style_id)
        self.accel_change_rate = self._accel_change_rate_interpolator(style_id)
        self.max_accel = self._max_accel_interpolator(style_id)
        self.min_accel = self._min_accel_interpolator(style_id)
        self.interval = self._interval_interpolator(style_id)

    def _cruise_control(self, ego_state: State) -> float:
        speed = ego_state.speed
        accel_last = ego_state.accel

        accel = (self.target_speed - speed) / self.kp
        accel = np.clip(
            accel,
            accel_last - self.accel_change_rate * self.delta_t,
            accel_last + self.accel_change_rate * self.delta_t,
        )
        accel = np.clip(accel, self.min_accel, self.max_accel)

        return accel

    def _adaptive_cruise_control(self, ego_state: State, front_state: State) -> float:
        distance_front = np.hypot(ego_state.x - front_state.x, ego_state.y - front_state.y)
        distance_target = ego_state.speed * self.interval + 5.0
        distance_target = np.clip(distance_target, 7.0, 80.0)

        relative_speed = front_state.speed - ego_state.speed
        relative_target_speed = (distance_target - distance_front) / self.kp
        relative_accel = (relative_target_speed - relative_speed) / self.kp

        accel = front_state.accel - relative_accel
        accel_last = ego_state.accel
        accel = np.clip(
            accel,
            accel_last - self.accel_change_rate * self.delta_t,
            accel_last + self.accel_change_rate * self.delta_t,
        )
        accel = np.clip(accel, self.min_accel, self.max_accel)

        return accel

    def step(self, ego_state: State):
        """This method outputs the acceleration and steering command based on the current state of the ego vehicle.

        Args:
            ego_state (State): The current state of the vehicle.

        Returns:
            steer (float): The steering command for the ego vehicle. It is always zero for the acceleration controller.
            accel (float): The acceleration command for the ego vehicle.
        """
        # TODO: Currently we cannot detect the front vehicle, add this later
        # front_state = None
        # front_target_state = None

        accel_cc = self._cruise_control(ego_state)
        # accel_acc_current = self.adaptive_cruise_control(ego_state, front_state)
        # accel_acc_target = self.adaptive_cruise_control(ego_veh, front_target_state)

        # accel = np.min([accel_cc, accel_acc_cur, accel_acc_tar])
        accel = accel_cc
        return 0.0, accel
