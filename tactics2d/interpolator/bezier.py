##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: bezier.py
# @Description: This file implements a Bezier curve interpolator.
# @Author: Tactics2D Team
# @Version: 0.1.9

import numpy as np
from cpp_interpolator import Bezier as cpp_Bezier


class Bezier:
    """This class implements a Bezier curve interpolator.

    Attributes:
        order (int): The order of the Bezier curve. The order of a Bezier curve is equal to the number of control points minus one.
    """

    def __init__(self, order: int):
        """Initialize the Bezier curve interpolator.

        Args:
            order (int): The order of the Bezier curve.
        """
        self.order = order
        if self.order < 1:
            raise ValueError("Bezier interpolator: Order must be greater than or equal to one.")

    def _check_validity(self, control_points: np.ndarray):
        if len(control_points.shape) != 2 or control_points.shape[1] != 2:
            raise ValueError("Bezier interpolator: Control points should have shape (n, 2).")
        if len(control_points) != self.order + 1:
            raise ValueError(
                "Bezier interpolator: Number of control points must be equal to order plus one."
            )

    def get_curve(self, control_points: np.ndarray, n_interpolation: int) -> np.ndarray:
        """
        Args:
            control_points (np.ndarray): The control points of the curve. The shape is (order + 1, 2).
            n_interpolation (int): The number of interpolations.

        Returns:
            curve_points (np.ndarray): The interpolated points of the curve. The shape is (n_interpolation, 2).
        """
        self._check_validity(control_points)

        # Directly call the optimized C++ function for Bezier curve calculation
        curve_points = cpp_Bezier.get_curve(control_points.tolist(), n_interpolation)
        return np.array(curve_points)
