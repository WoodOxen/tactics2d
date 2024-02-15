##! python3
# -*- coding: utf-8 -*-
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: single_track_dynamics.py
# @Description: This file implements a dynamic single-track model for a vehicle.
# @Author: Yueyuan Li
# @Version: 1.0.0

from typing import Tuple, Union

import numpy as np

from .physics_model_base import PhysicsModelBase
from tactics2d.participant.trajectory import State


class SingleTrackDynamics(PhysicsModelBase):
    """This class implements a dynamic single-track model for a vehicle.

    The dynamic single-track model is a simplified model to simulate the vehicle dynamics. It combines the front and rear wheels into a single wheel, and the vehicle is assumed to be a point mass.

    !!! quote "Reference"
        The dynamic single-track model is based on Chapter 7 of the following reference:
        [CommonRoad: Vehicle Models (2020a)](https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/-/blob/master/vehicleModels_commonRoad.pdf)
    """

    g = 9.81  # gravitational acceleration, m/s^2

    def __init__(
        self,
        lf: float,
        lr: float,
        mass: float,
        mass_height: float,
        mu: float = 0.7,
        iz: float = 1500,
        cf: float = 20.89,
        cr: float = 20.89,
        steer_range: Union[float, Tuple[float, float]] = None,
        speed_range: Union[float, Tuple[float, float]] = None,
        accel_range: Union[float, Tuple[float, float]] = None,
        interval: int = None,
        delta_t: int = None,
    ):
        """Initializes the single-track dynamics model.

        Args:
            lf (float): The distance from the center of mass to the front axle center. The unit is meter.
            lr (float): The distance from the center of mass to the rear axle center. The unit is meter.
            mass (float): The mass of the vehicle. The unit is kilogram. You can use the curb weight of the vehicle as an approximation.
            mass_height (float): The height of the center of mass from the ground. The unit is meter. You can use half of the vehicle height as an approximation.
            mu (float): The friction coefficient. It is a dimensionless quantity.
            iz (float): The moment of inertia of the vehicle. The unit is kilogram per meter squared (kg/m$^2$).
            cf (float): The cornering stiffness of the front wheel. The unit is 1/rad.
            cr (float): The cornering stiffness of the rear wheel. The unit is 1/rad.
        """
        self.lf = lf
        self.lr = lr
        self.mu = mu
        self.iz = iz
        self.cf = cf
        self.cr = cr
        self.whl_base = lf + lr
        self.mass = mass
        self.mass_height = mass_height

        if isinstance(steer_range, float):
            self.steer_range = None if steer_range < 0 else [-steer_range, steer_range]
        elif hasattr(steer_range, "__len__") and len(steer_range) == 2:
            if steer_range[0] >= steer_range[1]:
                self.steer_range = None
            else:
                self.steer_range = self.steer_range
        else:
            self.speed_range = None

        if isinstance(speed_range, float):
            self.speed_range = None if speed_range < 0 else [-speed_range, speed_range]
        elif hasattr(speed_range, "__len__") and len(speed_range) == 2:
            if speed_range[0] >= speed_range[1]:
                self.speed_range = None
            else:
                self.speed_range = self.speed_range
        else:
            self.speed_range = None

        if isinstance(accel_range, float):
            self.accel_range = None if accel_range < 0 else [-accel_range, accel_range]
        elif hasattr(accel_range, "__len__") and len(accel_range) == 2:
            if accel_range[0] >= accel_range[1]:
                self.accel_range = None
            else:
                self.accel_range = accel_range
        else:
            self.speed_range = None

        self.interval = interval

        if delta_t is None:
            self.delta_t = self._DELTA_T
        else:
            self.delta_t = max(delta_t, self._MIN_DELTA_T)
            if self.interval is not None:
                self.delta_t = min(self.delta_t, self.interval)

    def _step(self, state: State, accel: Tuple[float, float], delta: float, interval: int) -> State:
        dts = [float(self.delta_t) / 1000] * (interval // self.delta_t)
        dts.append(float(interval % self.delta_t) / 1000)

        factor_f = (self.g * self.lr - accel * self.mass_height) / self.wheel_base
        factor_r = (self.g * self.lf + accel * self.mass_height) / self.wheel_base

        x, y = state.location
        phi = state.heading
        v = state.speed
        dphi = v / self.wheel_base * np.tan(delta)
        beta = np.arctan(self.lr / self.lf * np.tan(delta))  # slip angle

        for dt in dts:
            dx = v * np.cos(phi + beta)
            dy = v * np.sin(phi + beta)
            dv = accel
            dbeta = (
                self.mu
                / v
                * (
                    self.cf * factor_f * delta
                    - (self.cr * factor_r + self.cf * factor_f) * beta
                    + (self.cr * factor_r * self.lr - self.cf * factor_f * self.lf) * dphi / v
                )
                - dphi
            )
            ddphi = (
                self.mu
                * self.mass
                / self.iz
                * (
                    self.lf * self.cf * factor_f * delta
                    + (self.lr * self.cr * factor_r - self.lf * self.cf * factor_f) * beta
                    - (self.lr**2 * self.cr * factor_r + self.lf**2 * self.cf * factor_f) * dphi / v
                )
            )

            x += dx * dt
            y += dy * dt
            v += dv * dt
            phi += dphi * dt
            beta += dbeta * dt
            dphi += ddphi * dt

            v = np.clip(v, *self.speed_range) if not self.speed_range is None else v

        state = State(
            frame=state.frame + interval,
            x=x,
            y=y,
            heading=np.mod(phi, 2 * np.pi),
            speed=v,
            accel=accel,
        )

        return state

    def step(self, state: State, accel: float, delta: float, interval: int = None) -> State:
        """This function updates the state of the vehicle based on the single-track dynamics model.

        Args:
            state (State): The current state of the traffic participant.
            accel (float): The acceleration of the traffic participant. The unit is meter per second squared (m/s$^2$).
            delta (float): The steering angle of the traffic participant. The unit is radian.
            interval (int): The time interval between the current state and the new state. The unit is millisecond.

        Returns:
            next_state (State): The new state of the traffic participant.
        """
        accel = np.clip(accel, *self.accel_range) if not self.accel_range is None else accel
        delta = np.clip(delta, *self.steer_range) if not self.steer_range is None else delta
        interval = interval if interval is not None else self.interval

        next_state = self._step(state, accel, delta, interval)

        return next_state

    def verify_state(self, state: State, last_state: State, interval: int = None) -> bool:
        return
