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

    Attributes:
        lf (float): The distance from the geometry center to the front axle center. The unit is meter.
        lr (float): The distance from the geometry center to the rear axle center. The unit is meter.
        steer_range (Union[float, Tuple[float, float]], optional): The steering angle range. The valid input is a float or a tuple of two floats represents (min steering angle, max steering angle). The unit is radian.

            - When the steer_range is a non-negative float, the steering angle is constrained to be within the range [-steer_range, steer_range].
            - When the steer_range is a tuple, the steering angle is constrained to be within the range [min steering angle, max steering angle].
            - When the steer_range is negative or the min steering angle is not less than the max steering angle, the steer_range is set to None.

        mass (float): The mass of the vehicle. The unit is kilogram.
        mass_height (float): The height of the center of mass from the ground. The unit is meter.
        mu (float): The friction coefficient. It is a dimensionless quantity. Defaults to 0.7.
        iz (float): The moment of inertia of the vehicle. The unit is kilogram per meter squared (kg/m$^2$). Defaults to 1500.
        cf (float): The cornering stiffness of the front wheel. The unit is 1/rad. Defaults to 20.89.
        cr (float): The cornering stiffness of the rear wheel. The unit is 1/rad. Defaults to 20.89.
        steer_range (Union[float, Tuple[float, float]], optional): The steering angle range. The valid input is a float or a tuple of two floats represents (min steering angle, max steering angle). The unit is radian.

            - When the steer_range is a non-negative float, the steering angle is constrained to be within the range [-steer_range, steer_range].
            - When the steer_range is a tuple, the steering angle is constrained to be within the range [min steering angle, max steering angle].
            - When the steer_range is negative or the min steering angle is not less than the max steering angle, the steer_range is set to None.

        speed_range (Union[float, Tuple[float, float]], optional): The speed range. The valid input is a float or a tuple of two floats represents (min speed, max speed). The unit is meter per second (m/s).
            - When the speed_range is a non-negative float, the speed is constrained to be within the range [-speed_range, speed_range].
            - When the speed_range is a tuple, the speed is constrained to be within the range [min speed, max speed].
            - When the speed_range is negative or the min speed is not less than the max speed, the speed_range is set to None.

        accel_range (Union[float, Tuple[float, float]], optional): The acceleration range. The valid input is a float or a tuple of two floats represents (min acceleration, max acceleration). The unit is meter per second squared (m/s$^2$).

            - When the accel_range is a non-negative float, the acceleration is constrained to be within the range [-accel_range, accel_range].
            - When the accel_range is a tuple, the acceleration is constrained to be within the range [min acceleration, max acceleration].
            - When the accel_range is negative or the min acceleration is not less than the max acceleration, the accel_range is set to None.

        interval (int, optional): The time interval between the current state and the new state. The unit is millisecond. Defaults to None.
        delta_t (int, optional): The time step for the simulation. The unit is millisecond. Defaults to `_DELTA_T`(5 ms). The expected value is between `_MIN_DELTA_T`(1 ms) and `interval`. It is recommended to keep delta_t smaller than 5 ms.
    """

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
            steer_range (Union[float, Tuple[float, float]], optional): The range of steering angle. The valid input is a positive float or a tuple of two floats represents (min steering angle, max steering angle). The unit is radian.
            speed_range (Union[float, Tuple[float, float]], optional): The range of speed. The valid input is a positive float or a tuple of two floats represents (min speed, max speed). The unit is meter per second (m/s).
            accel_range (Union[float, Tuple[float, float]], optional): The range of acceleration. The valid input is a positive float or a tuple of two floats represents (min acceleration, max acceleration). The unit is meter per second squared (m/s$^2$).
            interval (int, optional): The time interval between the current state and the new state. The unit is millisecond.
            delta_t (int, optional): The discrete time step for the simulation. The unit is millisecond.
        """
        self.lf = lf
        self.lr = lr
        self.whl_base = lf + lr
        self.mass = mass
        self.mass_height = mass_height
        self.mu = mu
        self.iz = iz
        self.cf = cf
        self.cr = cr

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

        factor_f = (self._G * self.lr - accel * self.mass_height) / self.wheel_base
        factor_r = (self._G * self.lf + accel * self.mass_height) / self.wheel_base

        x, y = state.location
        phi = state.heading
        v = state.speed
        d_phi = v / self.wheel_base * np.tan(delta)
        beta = np.arctan(self.lr / self.lf * np.tan(delta))  # slip angle

        for dt in dts:
            dx = v * np.cos(phi + beta)
            dy = v * np.sin(phi + beta)
            dv = accel
            d_beta = (
                self.mu
                / v
                * (
                    self.cf * factor_f * delta
                    - (self.cr * factor_r + self.cf * factor_f) * beta
                    + (self.cr * factor_r * self.lr - self.cf * factor_f * self.lf) * d_phi / v
                )
                - d_phi
            )
            dd_phi = (
                self.mu
                * self.mass
                / self.iz
                * (
                    self.lf * self.cf * factor_f * delta
                    + (self.lr * self.cr * factor_r - self.lf * self.cf * factor_f) * beta
                    - (self.lr**2 * self.cr * factor_r + self.lf**2 * self.cf * factor_f)
                    * d_phi
                    / v
                )
            )

            x += dx * dt
            y += dy * dt
            v += dv * dt
            phi += d_phi * dt
            beta += d_beta * dt
            d_phi += dd_phi * dt

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
        """This function updates the state of the vehicle based on the dynamics single-track model.

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
