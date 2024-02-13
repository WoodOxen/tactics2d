##! python3
# -*- coding: utf-8 -*-
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: cubic_spline.py
# @Description: This file implements a cubic spline interpolator.
# @Author: Yueyuan Li
# @Version: 1.0.0

from enum import Enum

import numpy as np


class CubicSpline:
    """This class implement a cubic spline interpolator.

    Attributes:
        boundary_type (int): Boundary condition type. The cubic spline interpolator offers three distinct boundary condition options: natural (1), clamped (2), and not-a-knot (3). By default, the not-a-knot boundary condition is applied, serving as a wise choice when specific boundary condition information is unavailable.
    """

    class BoundaryType(Enum):
        Natural = 1
        Clamped = 2
        NotAKnot = 3

    def __init__(self, boundary_type: int = 3):
        """Initialize the cubic spline interpolator.

        Args:
            boundary_type (int, optional): Boundary condition type. Defaults to 3. The available options are natural (1), clamped (2), and not-a-knot (3).

        Raises:
            ValueError: The boundary type is not valid. Please choose from 1 (CubicSpline.BoundaryType.Natural), 2 (CubicSpline.BoundaryType.Clamped), and 3 (CubicSpline.BoundaryType.NotAKnot).
        """
        self.boundary_type = boundary_type

        if self.boundary_type not in self.BoundaryType.__members__.values():
            raise ValueError(
                "The boundary type is not valid. Please choose from 1 (CubicSpline.BoundaryType.Natural), 2 (CubicSpline.BoundaryType.Clamped), and 3 (CubicSpline.BoundaryType.NotAKnot)."
            )

    def _check_validity(self, control_points: np.ndarray):
        if len(control_points.shape) != 2 or control_points.shape[1] != 2:
            raise ValueError("The shape of control_points is expected to be (n, 2).")

        if len(control_points) < 3:
            raise ValueError(
                "There is not enough control points to interpolate a cubic spline curve."
            )

        if np.any((control_points[1:, 0] - control_points[:-1, 0]) < 0):
            raise ValueError("The x coordinates of the control points must be non-decreasing.")

    def get_parameters(self, control_points: np.ndarray, xx: tuple = (0, 0)):
        """Get the parameters of the cubic functions

        Args:
            control_points (np.ndarray): The control points of the curve. The shape is (n + 1, 2).
            xx (float): The first derivative of the curve at the first and the last control points. Defaults to (0, 0).

        Returns:
            a (np.ndarray): The constant parameters of the cubic functions. The shape is (n, 1).
            b (np.ndarray): The linear parameters of the cubic functions. The shape is (n, 1).
            c (np.ndarray): The quadratic parameters of the cubic functions. The shape is (n, 1).
            d (np.ndarray): The cubic parameters of the cubic functions. The shape is (n, 1).
        """
        self._check_validity(control_points)

        n = control_points.shape[0] - 1
        x = control_points[:, 0]
        y = control_points[:, 1]

        h = x[1:] - x[:-1]  # shape=(n, 1), h_i = x_i+1 - x_i
        b = (y[1:] - y[:-1]) / h  # shape=(n, 1), b_i = (y_i+1 - y_i) / h_i

        # Construct the matrix A
        A = np.zeros((n + 1, n + 1))
        for i in range(1, n):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]

        # construct the vector B
        B = 6 * np.array([0] + list(b[1:] - b[:-1]) + [0])  # shape=(n+1, 1)

        # Set the boundary conditions
        if self.boundary_type == self.BoundaryType.Natural:
            A[0, 0] = 1
            A[-1, -1] = 1
        elif self.boundary_type == self.BoundaryType.Clamped:
            A[0, 0] = 2 * h[0]
            A[0, 1] = h[0]
            A[-1, -1] = 2 * h[-1]
            A[-1, -2] = h[-1]
            B[0] = 6 * (b[0] - xx[0])
            B[-1] = 6 * (xx[1] - b[-1])
        elif self.boundary_type == self.BoundaryType.NotAKnot:
            A[0, 0] = -h[1]
            A[0, 1] = h[0] + h[1]
            A[0, 2] = -h[0]
            A[-1, -3] = -h[-1]
            A[-1, -2] = h[-2] + h[-1]
            A[-1, -1] = -h[-2]

        m = np.linalg.solve(A, B)  # shape=(n+1, 1)

        a = y[:-1]
        b = b - h * m[:-1] / 2 - h * (m[1:] - m[:-1]) / 6
        c = m[:-1] / 2
        d = (m[1:] - m[:-1]) / (6 * h)

        return a, b, c, d

    def get_curve(
        self, control_points: np.ndarray, xx: tuple = (0, 0), n_interpolation: int = 100
    ) -> np.ndarray:
        """Get the interpolation points of a cubic spline curve.

        Args:
            control_points (np.ndarray): The control points of the curve. The shape is (n + 1, 2).
            xx (float): The first derivative of the curve at the first and the last control points. These conditions will be used when the boundary condition is "clamped". Defaults to (0, 0).
            n_interpolation (int): The number of interpolations between every two control points. Defaults to 100.

        Returns:
            curve_points (np.ndarray): The interpolation points of the curve. The shape is (n_interpolation * n + 1, 2).
        """
        self._check_validity(control_points)
        a, b, c, d = self.get_parameters(control_points, xx)
        n = control_points.shape[0] - 1

        curve_points = []

        for i in range(n):
            x = np.linspace(control_points[i, 0], control_points[i + 1, 0], n_interpolation)
            y = (
                a[i]
                + b[i] * (x - control_points[i, 0])
                + c[i] * (x - control_points[i, 0]) ** 2
                + d[i] * (x - control_points[i, 0]) ** 3
            )
            curve_points += list(zip(x, y))

        curve_points.append(control_points[-1])

        return np.array(curve_points)
