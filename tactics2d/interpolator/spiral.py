##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: spiral.py
# @Description: This file implements a spiral interpolation.
# @Author: Tactics2D Team
# @Version: 0.1.8rc1


import numpy as np
from scipy.special import fresnel


class Spiral:
    """This class implements a spiral interpolation."""

    @staticmethod
    def get_curve(
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
            # Proper Euler spiral using arc-length param
            a = np.sqrt(np.pi / np.abs(gamma))
            scaled_s = interpolations * np.sqrt(np.abs(gamma) / np.pi)

            S, C = fresnel(scaled_s)

            # Compute delta from s=0
            delta_C = C
            delta_S = S

            # Complex offset
            complex_offset = a * (delta_C + 1j * delta_S)

            # Rotation
            theta_0 = heading - start_curvature**2 / (2 * gamma)
            rotation = np.exp(1j * theta_0)

            spiral_points = complex_offset * rotation
            x_interpolated = x_start + spiral_points.real
            y_interpolated = y_start + spiral_points.imag

        return np.array([x_interpolated, y_interpolated]).T
