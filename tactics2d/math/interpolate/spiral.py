##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: spiral.py
# @Description: This file implements a spiral interpolation.
# @Author: Yueyuan Li
# @Version: 1.0.0

import numpy as np
from scipy.special import fresnel


class Spiral:
    """This class implements a spiral interpolation."""

    @staticmethod
    def get_spiral(
        length: float, start_point: np.ndarray, heading: float, start_curvature: float, gamma: float
    ) -> np.ndarray:
        """This function gets the points on a spiral curve line.

        Args:
            length (float): The length of the spiral.
            start_point (np.ndarray): The start point of the spiral. The shape is (2,).
            heading (float): The heading of the start point. The unit is radian.
            start_curvature (float): The curvature of the start point.
            gamma (float): The rate of change of curvature

        Returns:
            np.ndarray: The points on the spiral curve line. The shape is (n_interpolate, 2).
        """

        # Start
        x_start, y_start = start_point
        # Interpolation
        interpolations = np.linspace(0, length, int(length / 0.01) if length < 100 else 10000)

        if gamma == 0 and start_curvature == 0:
            # Straight line
            x_interpolated = x_start + interpolations * np.cos(heading)
            y_interpolated = y_start + interpolations * np.sin(heading)

        elif gamma == 0 and start_curvature != 0:
            # Arc
            x_interpolated = x_start + (1 / start_curvature) * (
                np.sin(heading + start_curvature * interpolations) - np.sin(heading)
            )
            y_interpolated = y_start + (1 / start_curvature) * (
                np.cos(heading) - np.cos(heading + start_curvature * interpolations)
            )

        else:
            scaler = np.sqrt(np.pi * np.abs(gamma))
            # Fresnel integrals
            S_interpolated, C_interpolated = fresnel(
                (start_curvature + gamma * interpolations) / scaler
            )
            S_start, C_start = fresnel(start_curvature / scaler)

            # Euler Spiral
            Cs1 = np.sqrt(np.pi / np.abs(gamma)) * np.exp(
                1j * (heading - start_curvature**2 / 2 / gamma)
            )
            Cs2 = np.sign(gamma) * (C_interpolated - C_start) + 1j * S_interpolated - 1j * S_start
            delta_C = Cs1 * Cs2
            x_interpolated = x_start + delta_C.real
            y_interpolated = y_start + delta_C.imag

        return np.array([x_interpolated, y_interpolated]).T
