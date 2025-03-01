##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: acceleration_controller.py
# @Description:
# @Author: Tactics2D Team
# @Version:

import numpy as np
from scipy.interpolate import interp1d

from tactics2d.participant.trajectory.state import State


class AccelerationController:
    """ """

    def __init__(
        self,
        kp: float = 3.5,
        speed_factor: float = 1.0,
        accel_change_rate: float = 3.0,
        max_accel: float = 1.5,
        min_accel: float = -4.0,
        interval: int = 200,
        delta_t: int = 5,
    ):
        """
        Args:
            interval (int, optional): The time interval between the current state and the new state. The unit is millisecond.
            delta_t (int, optional): The discrete time step for the simulation. The unit is millisecond.
        """
        self.kp = kp
        self.speed_factor = speed_factor
        self.accel_change_rate = accel_change_rate
        self.max_accel = max_accel
        self.min_accel = min_accel
        self.interval = interval
        self.delta_t = delta_t

        self.kp_interpolator = interp1d(
            [-1.0, 1.0], [4.5, 2.5], kind="linear", bounds_error=False, fill_value=(4.5, 2.5)
        )
        self.speed_factor_interpolator = interp1d(
            [-1.0, 1.0], [0.8, 1.2], kind="linear", bounds_error=False, fill_value=(0.8, 1.2)
        )
        self.accel_change_rate_interpolator = interp1d(
            [-1.0, 1.0], [3.0, 6.0], kind="linear", bounds_error=False, fill_value=(5.0, 6.0)
        )
        self.max_accel_interpolator = interp1d(
            [-1.0, 1.0], [1.5, 2.5], kind="linear", bounds_error=False, fill_value=(1.5, 2.5)
        )
        self.min_accel_interpolator = interp1d(
            [-1.0, 1.0], [-3.0, -5.0], kind="linear", bounds_error=False, fill_value=(-3.0, -5.0)
        )
        self.interval_interpolator = interp1d(
            [-1.0, 1.0], [3.5, 1.5], kind="linear", bounds_error=False, fill_value=(3.5, 1.5)
        )

    def update_driving_style(self, style_id: int):
        self.kp = self.kp_interpolator(style_id)
        self.speed_factor = self.speed_factor_interpolator(style_id)
        self.accel_change_rate = self.accel_change_rate_interpolator(style_id)
        self.max_accel = self.max_accel_interpolator(style_id)
        self.min_accel = self.min_accel_interpolator(style_id)
        self.interval = self.interval_interpolator(style_id)

    def cruise_control(self, ego_state: State):
        dt = float(self.delta_t) / 1000
        speed = ego_state.speed
        accel_last = ego_state.accel

        accel = (self.target_speed - speed) / self.kp
        accel = np.clip(
            accel,
            accel_last - self.accel_change_rate * dt,
            accel_last + self.accel_change_rate * dt,
        )
        accel = np.clip(accel, self.min_accel, self.max_accel)

        return accel

    def adaptive_cruise_control(self, ego_state: State, front_state: State):
        dt = float(self.delta_t) / 1000

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
            accel_last - self.accel_change_rate * dt,
            accel_last + self.accel_change_rate * dt,
        )
        accel = np.clip(accel, self.min_accel, self.max_accel)

        return accel

    def step(self, ego_state: State):
        # TODO: Currently we cannot detect the front vehicle, add this later
        front_state = None
        front_target_state = None

        accel_cc = self.cruise_control(ego_state)
        # accel_acc_current = self.adaptive_cruise_control(ego_state, front_state)
        # accel_acc_target = self.adaptive_cruise_control(ego_veh, front_target_state)

        # accel = np.min([accel_cc, accel_acc_cur, accel_acc_tar])
        accel = accel_cc
        return accel, 0.0
