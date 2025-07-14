##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: pid_controller.py
# @Description:
# @Author: Tactics2D Team
# @Version: 0.1.9


import math
from collections import deque

import numpy as np
from numpy.typing import ArrayLike


class PIDController:
    def __init__(self, k_p=1.0, k_i=0.0, k_d=0.0, dt=0.02):
        """Initialize the PID controller with given parameters.

        Args:
            k_p (float, optional): The proportional gain. Defaults to 1.0.
            k_i (float, optional): The integral gain. Defaults to 0.0.
            k_d (float, optional): The derivative gain. Defaults to 0.0.
            dt (float, optional): The time step for the controller. Defaults to 0.02 s.
        """
        self._k_p = k_p
        self._k_i = k_i
        self._k_d = k_d
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def step(self, error: float) -> float:
        """Step the PID controller with the given error.

        Args:
            error (float): The error value to be processed by the PID controller.

        Returns:
            float: The control output computed by the PID controller.
        """
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return (self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie)


class PIDLongitudinalController(PIDController):
    def __init__(self, k_p=1.0, k_i=0.0, k_d=0.0, dt=0.02):
        """Initialize the PID controller with given parameters.

        Args:
            k_p (float, optional): The proportional gain. Defaults to 1.0.
            k_i (float, optional): The integral gain. Defaults to 0.0.
            k_d (float, optional): The derivative gain. Defaults to 0.0.
            dt (float, optional): The time step for the controller. Defaults to 0.02 s.
        """
        super().__init__(k_p, k_i, k_d, dt)

    def step(self, current_speed: float, target_speed: float) -> float:
        """Step the longitudinal PID controller with the current and target speeds.

        Args:
            current_speed (float): The current speed of the vehicle.
            target_speed (float): The target speed to be achieved.

        Returns:
            float: The control output computed by the PID controller, clipped to the range [-1.0, 1.0].
        """
        error = target_speed - current_speed
        return np.clip(super().step(error), -1.0, 1.0)


class PIDLateralController(PIDController):
    def __init__(self, k_p=1.0, k_i=0.0, k_d=0.0, dt=0.02):
        """Initialize the PID controller with given parameters.

        Args:
            k_p (float, optional): The proportional gain. Defaults to 1.0.
            k_i (float, optional): The integral gain. Defaults to 0.0.
            k_d (float, optional): The derivative gain. Defaults to 0.0.
            dt (float, optional): The time step for the controller. Defaults to 0.02 s.
        """
        super().__init__(k_p, k_i, k_d, dt)

    def step(
        self, current_position: ArrayLike, target_position: ArrayLike, current_heading: float
    ) -> float:
        """Step the lateral PID controller with the current position, target position, and current heading.

        Args:
            current_position (ArrayLike): The current position of the vehicle as a 2D array-like structure (e.g., [x, y]).
            target_position (ArrayLike): The target position of the vehicle as a 2D array-like structure (e.g., [x, y]).
            current_heading (float): The current heading of the vehicle in radians.

        Returns:
            float: The control output computed by the PID controller, representing the steering angle, clipped to the range [-1.0, 1.0].
        """
        direction_vector = np.array([np.cos(current_heading), np.sin(current_heading), 0])
        displacement_vector = np.array(
            [target_position[0] - current_position[0], target_position[1] - current_position[1], 0]
        )

        norm_displacement = np.linalg.norm(displacement_vector)
        if norm_displacement < 1e-6:
            return 0.0

        dot_product = math.acos(
            np.clip(
                np.dot(direction_vector, displacement_vector)
                / (np.linalg.norm(direction_vector) * np.linalg.norm(displacement_vector)),
                -1.0,
                1.0,
            )
        )
        cross_product = np.cross(direction_vector, displacement_vector)

        if cross_product[2] < 0:
            dot_product = -dot_product

        return np.clip(super().step(dot_product), -1.0, 1.0)
