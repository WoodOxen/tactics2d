##! python3
# -*- coding: utf-8 -*-
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: point_mass.py
# @Description: This file implements a point mass model for a traffic participant.
# @Author: Yueyuan Li
# @Version: 1.0.0

from typing import Tuple, Union
import logging

import numpy as np

from .physics_model_base import PhysicsModelBase
from tactics2d.participant.trajectory import State


class PointMass(PhysicsModelBase):
    """This class implements a point mass model for a traffic participant. The point mass model supposes that the mass of the object is concentrated at the center of the object. The state of the object is represented by its center position, velocity, and heading. The object is assumed to be operating in a 2D plane (x-y).

    !!! warning
        This model is recommended to be used for pedestrians. Because the point mass model ignores that vehicles have a minimum turning circle, if this model is used for bicycle and vehicles, the results will not be accurate.

    Attributes:
        speed_range (Union[float, Tuple[float, float]]: The range of speed. The valid input is a float or a tuple of two floats represents (min speed, max speed). The unit is meter per second (m/s). The default value is None, which means no constraint on the speed. When the speed_range is negative or the min speed is not less than the max speed, the speed_range is set to None.
        accel_range (Union[float, Tuple[float, float]]: The range of acceleration. The valid input is a float or a tuple of two floats represents (min acceleration, max acceleration). The unit is meter per second squared (m/s$^2$). The default value is None, which means no constraint on the acceleration. When the accel_range is negative or the min acceleration is not less than the max acceleration, the accel_range is set to None.
        interval (int): The time interval between the current state and the new state. The unit is millisecond. Defaults to None.
        delta_t (int): The discrete time step for the simulation. The unit is millisecond. Defaults to `_DELTA_T`(5 ms). The expected value is between `_MIN_DELTA_T`(1 ms) and `interval`. It is recommended to keep delta_t smaller than 5 ms.
        backend (str): The backend for the simulation. The default value is `newton`. The available choices are `newton` and `euler`. The `newton` backend is recommended because it is faster. The `euler` backend is used for comparison and testing purposes at currently. We plan to improve the `euler` backend in the future (maybe in version 1.1.0)
    """

    backends = ["newton", "euler"]

    def __init__(
        self,
        speed_range: Union[float, Tuple[float, float]] = None,
        accel_range: Union[float, Tuple[float, float]] = None,
        interval: int = 100,
        delta_t: int = None,
        backend: str = "newton",
    ):
        """Initialize the point mass model.

        Args:
            speed_range (Union[float, Tuple[float, float]], optional): The range of speed. The valid input is a positive float or a tuple of two floats represents (min speed, max speed). The unit is meter per second (m/s).
            accel_range (Union[float, Tuple[float, float]], optional): The range of acceleration. The valid input is a positive float or a tuple of two floats represents (min acceleration, max acceleration). The unit is meter per second squared (m/s$^2$).
            interval (int, optional): The time interval between the current state and the new state. The unit is millisecond.
            delta_t (int, optional): The discrete time step for the simulation. The unit is millisecond.
            backend (str, optional): The backend for the simulation. The available choices are `newton` and `euler`.
        """
        if isinstance(speed_range, float):
            self.speed_range = None if speed_range < 0 else [0, speed_range]
        elif hasattr(speed_range, "__len__") and len(speed_range) == 2:
            self.speed_range = [max(0, speed_range[0]), max(0, speed_range[1])]
            if self.speed_range[0] >= self.speed_range[1]:
                self.speed_range = None
        else:
            self.speed_range = None

        if isinstance(accel_range, float):
            self.accel_range = None if accel_range < 0 else [0, speed_range]
        elif hasattr(accel_range, "__len__") and len(accel_range) == 2:
            self.accel_range = [max(0, accel_range[0]), max(0, accel_range[1])]
            if self.accel_range[0] >= self.accel_range[1]:
                self.accel_range = None
        else:
            self.accel_range = None

        self.interval = interval

        if delta_t is None:
            self.delta_t = self._DELTA_T
        else:
            self.delta_t = max(delta_t, self._MIN_DELTA_T)
            if self.interval is not None:
                self.delta_t = min(self.delta_t, self.interval)

        if backend not in self.backends:
            logging.warning(f"Unsupported backend {backend}. Using `newton` instead.")
            self.backend = "newton"
        else:
            self.backend = backend

    def _step_newton(self, state: State, accel: Tuple[float, float], interval: int) -> State:
        ax, ay = accel
        vx, vy = state.velocity
        dt = float(interval) / 1000

        next_vx = vx + ax * dt
        next_vy = vy + ay * dt
        next_speed = np.linalg.norm([next_vx, next_vy])

        # When the speed has not exceeded the constraint, the next state can be calculated directly.
        if self.speed_range is None or self.speed_range[0] <= next_speed <= self.speed_range[1]:
            next_state = State(
                frame=state.frame + interval,
                x=state.x + vx * dt + 0.5 * ax * dt**2,
                y=state.y + vy * dt + 0.5 * ay * dt**2,
                heading=np.arctan2(next_vy, next_vx),
                vx=next_vx,
                vy=next_vy,
            )
        # When the speed has exceeded the constraint, the next state needs to be calculated in two steps.
        # 1. from current velocity to the constraint velocity
        # 2. the velocity is constant, but the heading is changing
        elif next_speed < self.speed_range[0]:
            a_ = ax**2 + ay**2
            b_ = 2 * (ax * vx + ay * vy)
            c_ = vx**2 + vy**2 - self.speed_range[0] ** 2
            t1 = (-b_ - np.sqrt(b_**2 - 4 * a_ * c_)) / (
                2 * a_
            )  # assume the minimal speed is positive
            t2 = dt - t1
            vx_min = vx + ax * t1
            vy_min = vy + ay * t1

            next_state = State(
                frame=state.frame + interval,
                x=state.x + vx * t1 + 0.5 * ax * t1**2 + vx_min * t2,
                y=state.y + vy * t1 + 0.5 * ay * t1**2 + vy_min * t2,
                heading=np.arctan2(vy_min, vx_min),
                vx=vx_min,
                vy=vy_min,
            )
        else:
            a_ = ax**2 + ay**2
            b_ = 2 * (ax * vx + ay * vy)
            c_ = vx**2 + vy**2 - self.speed_range[1] ** 2
            t1 = (-b_ + np.sqrt(b_**2 - 4 * a_ * c_)) / (2 * a_)
            t2 = dt - t1
            vx_max = vx + ax * t1
            vy_max = vy + ay * t1

            next_state = State(
                frame=state.frame + interval,
                x=state.x + vx * t1 + 0.5 * ax * t1**2 + vx_max * t2,
                y=state.y + vy * t1 + 0.5 * ay * t1**2 + vy_max * t2,
                heading=np.arctan2(vy_max, vx_max),
                vx=vx_max,
                vy=vy_max,
            )
        return next_state

    def _step_euler(self, state: State, accel: Tuple[float, float], interval: int = None) -> State:
        ax, ay = accel
        vx, vy = state.velocity
        x, y = state.location
        heading = state.heading
        dts = [float(self.delta_t) / 1000] * (interval // self.delta_t)
        dts.append(float(interval % self.delta_t) / 1000)

        for dt in dts:
            vx += ax * dt
            vy += ay * dt
            speed = np.linalg.norm([vx, vy])
            speed_clipped = (
                np.clip(speed, *self.speed_range) if not self.speed_range is None else speed
            )
            if speed != speed_clipped:
                vx = speed_clipped * np.cos(heading)
                vy = speed_clipped * np.sin(heading)

            x += vx * dt
            y += vy * dt
            heading = np.arctan2(vy, vx)

        next_state = State(
            frame=state.frame + interval, x=x, y=y, heading=heading, vx=vx, vy=vy, ax=ax, ay=ay
        )

        return next_state

    def step(self, state: State, accel: Tuple[float, float], interval: int = None) -> State:
        """This function updates the state of the traffic participant based on the point mass model.

        Args:
            state (State): The current state of the traffic participant.
            accel (Tuple[float, float]): The acceleration vector ($a_x$, $a_y$). The unit of the acceleration is meter per second squared (m/s$^2$).
            interval (int): The time interval between the current state and the new state. The unit is millisecond.

        Returns:
            next_state (State): A new state of the traffic participant.
        """
        interval = interval if interval is not None else self.interval

        accel_value = np.linalg.norm(accel)
        accel_value = (
            np.clip(accel_value, *self.accel_range) if not self.accel_range is None else accel_value
        )

        if self.backend == "newton":
            next_state = self._step_newton(state, accel, interval)
        elif self.backend == "euler":
            next_state = self._step_euler(state, accel, interval)

        return next_state

    def verify_state(self, state: State, last_state: State, interval: int = None) -> bool:
        """This function provides a very rough check for the state transition. It checks whether the acceleration and the steering angle are within the range.

        Args:
            state (State): The new state of the traffic participant.
            last_state (State): The last state of the traffic participant.
            interval (int): The time interval between the last state and the new state. The unit is millisecond.

        Returns:
            True if the new state is valid, False otherwise.
        """
        interval = interval if not interval is None else state.frame - last_state.frame
        dt = interval / 1000  # convert to second
        denominator = 2 / dt**2
        ax = (state.x - last_state.x - last_state.vx * dt) * denominator
        ay = (state.y - last_state.y - last_state.vy * dt) * denominator

        if not self.accel_range is None:
            accel = np.linalg.norm([ax, ay])
            if not self.accel_range[0] <= accel <= self.accel_range[1]:
                return False

        return True
