# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Cubic spline implementation."""


from enum import Enum
from typing import Union

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

    @staticmethod
    def get_curve(
        control_points: np.ndarray,
        xx: tuple = (0, 0),
        n_interpolation: int = 100,
        boundary_type: Union[int, BoundaryType] = BoundaryType.NotAKnot,
    ) -> np.ndarray:
        """Get the interpolation points of a cubic spline curve.

        Args:
            control_points (np.ndarray): The control points of the curve. The shape is (n + 1, 2).
            xx (float): The first derivative of the curve at the first and the last control points. These conditions will be used when the boundary condition is "clamped". Defaults to (0, 0).
            n_interpolation (int): The number of interpolations between every two control points. Defaults to 100.
            boundary_type (Union[int, BoundaryType]): The boundary condition type. It can be an integer value from 1 to 3 or a BoundaryType enum value. Defaults to NotAKnot.

        Returns:
            curve_points (np.ndarray): The interpolation points of the curve. The shape is (n_interpolation * n + 1, 2).
        """

        # Check the validity of the inputs
        if isinstance(boundary_type, int):
            try:
                boundary_type = CubicSpline.BoundaryType(boundary_type)
            except ValueError:
                raise ValueError(
                    f"Invalid boundary type: {boundary_type}. Available options are: {', '.join(CubicSpline.BoundaryType.__members__.keys())} or an integer value from 1 to {len(CubicSpline.BoundaryType.__members__)}."
                )
        if boundary_type not in CubicSpline.BoundaryType.__members__.values():
            raise ValueError(
                f"Invalid boundary type: {boundary_type}. Available options are: {', '.join(CubicSpline.BoundaryType.__members__.keys())} or an integer value from 1 to {len(CubicSpline.BoundaryType.__members__)}."
            )

        if control_points.ndim != 2 or control_points.shape[1] != 2:
            raise ValueError("Control points should have shape (n, 2).")

        if len(control_points) < 3:
            raise ValueError("Need at least 3 control points.")

        if np.any(np.diff(control_points[:, 0]) <= 0):
            raise ValueError("x-values must be strictly increasing.")

        # Compute the cubic spline curve points using the C++ implementation
        curve_points = cpp_CubicSpline.get_curve(
            control_points.tolist(),
            xx,
            n_interpolation,
            cpp_CubicSpline.BoundaryType(boundary_type.value),
        )
        return np.array(curve_points)
