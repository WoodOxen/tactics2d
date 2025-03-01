##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: pure_pursuit_controller.py
# @Description:
# @Author: Tactics2D Team
# @Version: 0.18.0

from typing import Tuple, Union

import numpy as np
from scipy.interpolate import interp1d

from tactics2d.participant.trajectory.state import State

from .acceleration_controller import AccelerationController


class PurePursuitController:
    def __init__(
        self,
        pre_aiming_distance: float = 15.0,
        pre_aiming_interval: float = 1.0,
        kp: float = 3.5,
        accel_change_rate: float = 3.0,
        max_accel: float = 1.5,
        min_accel: float = -4.0,
        interval: float = 2.0,
    ):
        self.pre_aiming_distance = pre_aiming_distance
        self.pre_aiming_interval = pre_aiming_interval
        self.kp = kp
        self.accel_change_rate = accel_change_rate
        self.max_accel = max_accel
        self.min_accel = min_accel
        self.interval = interval

        self.pre_aiming_interval_interpolator = interp1d(
            [-1.0, 1.0], [2.0, 1.0], kind="linear", bounds_error=False, fill_value=(4.0, 2.0)
        )
        self.kp_interpolator = interp1d(
            [-1.0, 1.0], [4.5, 2.5], kind="linear", bounds_error=False, fill_value=(4.5, 2.5)
        )
        self.accel_change_rate_interpolator = interp1d(
            [-1.0, 1.0], [2.0, 6.0], kind="linear", bounds_error=False, fill_value=(2.0, 6.0)
        )
        self.max_accel_interpolator = interp1d(
            [-1.0, 1.0], [1.5, 2.5], kind="linear", bounds_error=False, fill_value=(1.5, 2.5)
        )
        self.min_accel_interpolator = interp1d(
            [-1.0, 1.0], [-3.0, -5.0], kind="linear", bounds_error=False, fill_value=(-3.0, -5.0)
        )

        self.longitudinal_control = AccelerationController()

    def update_driving_style(self, style_id: int):
        self.longitudinal_control.update_driving_style(style_id)

        self.pre_aiming_interval = self.pre_aiming_interval_interpolator(style_id)
        self.kp = self.kp_interpolator(style_id)
        self.accel_change_rate = self.accel_change_rate_interpolator(style_id)
        self.max_accel = self.max_accel_interpolator(style_id)
        self.min_accel = self.min_accel_interpolator(style_id)

    def lateral_control(self, ego_state: State, wheel_base: float):
        position = np.array(ego_state.location)

        # TODO: Convert the vehicle location from global coordinates to local coordinates
        # proj_point, proj_s, _ ,_= veh.plan_traj_glb.frenet_projection(position)

        # f_x = interp1d(veh.plan_traj_glb.s, veh.plan_traj_glb.x, kind='linear', fill_value="extrapolate")
        # f_y = interp1d(veh.plan_traj_glb.s, veh.plan_traj_glb.y, kind='linear', fill_value="extrapolate")

        ref_point_s = proj_s + self.pre_aiming_distance
        ref_point_x = f_x(ref_point_s)
        ref_point_y = f_y(ref_point_s)

        pre_aiming_angle = np.arctan2(ref_point_y - ego_state.y, ref_point_x - ego_state.x)
        # print("preview_angle_",preview_angle_)
        distance = np.linalg.norm((ref_point_y - ego_state.y, ref_point_x - ego_state.x))
        # print("dis",dis_2_pp)
        steer = np.arctan(
            2.0 * wheel_base * np.sin(pre_aiming_angle - ego_state.heading) / distance
        )

        return steer

    def step(self, ego_state: State):
        accel = self.longitudinal_control.step()
        steer = self.lateral_control()

        self.pre_aiming_distance = self.pre_aiming_interval * ego_state.speed
        self.pre_aiming_distance = np.max([self.pre_aiming_distance, 5.0])

        accel, _ = self.longitudinal_control.step(ego_state)
        steer = self.lateral_control
        return accel, steer
