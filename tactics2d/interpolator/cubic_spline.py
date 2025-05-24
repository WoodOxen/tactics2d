##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: cubic_spline.py
# @Description: This file implements a cubic spline interpolator.
# @Author: Tactics2D Team
# @Version: 0.1.9
from enum import Enum

import numpy as np
from cpp_interpolator import CubicSpline as cpp_CubicSpline


class CubicSpline:
    """This class implement a cubic spline interpolator.

    Attributes:
        boundary_type (int): Boundary condition type. The cubic spline interpolator offers three distinct boundary condition options: Natural (1), Clamped (2), and NotAKnot (3). By default, the not-a-knot boundary condition is applied, serving as a wise choice when specific boundary condition information is unavailable.
    """

    class BoundaryType(Enum):
        """The boundary condition type of the cubic spline interpolator.

        Attributes:
            Natural (int): Natural boundary condition. The second derivative of the curve at the first and the last control points is set to 0.
            Clamped (int): Clamped boundary condition. The first derivative of the curve at the first and the last control points is set to the given values.
            NotAKnot (int): Not-a-knot boundary condition. The first and the second cubic functions are connected at the second and the third control points, and the last and the second-to-last cubic functions are connected at the last and the second-to-last control points.
        """

        Natural = 1
        Clamped = 2
        NotAKnot = 3

    def __init__(self, boundary_type: BoundaryType = BoundaryType.NotAKnot):
        """Initialize the cubic spline interpolator.

        Args:
            boundary_type (BoundaryType, optional): Boundary condition type. Defaults to BoundaryType.NotAKnot. The available options are CubicSpline.BoundaryType.Natural, CubicSpline.BoundaryType.Clamped, and CubicSpline.BoundaryType.NotAKnot.
        """
        self.boundary_type = boundary_type
        if self.boundary_type not in self.BoundaryType.__members__.values():
            raise ValueError(
                "The boundary type is not valid. Please choose from 1 (CubicSpline.BoundaryType.Natural), 2 (CubicSpline.BoundaryType.Clamped), and 3 (CubicSpline.BoundaryType.NotAKnot)."
            )

        if self.boundary_type == CubicSpline.BoundaryType.Natural:
            self.cpp_cubic_spline = cpp_CubicSpline(cpp_CubicSpline.BoundaryType.Natural)
        elif boundary_type == CubicSpline.BoundaryType.Clamped:
            self.cpp_cubic_spline = cpp_CubicSpline(cpp_CubicSpline.BoundaryType.Clamped)
        elif boundary_type == CubicSpline.BoundaryType.NotAKnot:
            self.cpp_cubic_spline = cpp_CubicSpline(cpp_CubicSpline.BoundaryType.NotAKnot)

    def _check_validity(self, control_points: np.ndarray):
        if control_points.ndim != 2 or control_points.shape[1] != 2:
            raise ValueError("Cubic interpolator: Control points should have shape (n, 2).")
        if len(control_points) < 3:
            raise ValueError("Cubic interpolator: Need at least 3 control points.")
        if np.any(np.diff(control_points[:, 0]) <= 0):
            raise ValueError("Cubic interpolator: x-values must be strictly increasing.")

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
        curve_points = self.cpp_cubic_spline.get_curve(control_points.tolist(), xx, n_interpolation)
        return np.array(curve_points)
