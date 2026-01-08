##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: bezier.py
# @Description: This file implements a Bezier curve interpolator.
# @Author: Tactics2D Team
# @Version: 0.1.8rc1

import numpy as np

from cpp_interpolator import Bezier as cpp_Bezier


class Bezier:
    """This class implements a Bezier curve interpolator."""

    @staticmethod
    def get_curve(
        control_points: np.ndarray, n_interpolation: int, order: int = None
    ) -> np.ndarray:
        """
        Args:
            control_points (np.ndarray): The control points of the curve. The shape is (order + 1, 2).
            n_interpolation (int): The number of interpolations.
            order (int, optional): The order of the Bezier curve. If not provided, it will be set to the number of control points minus one.

        Returns:
            curve_points (np.ndarray): The interpolated points of the curve. The shape is (n_interpolation, 2).
        """
        # Check the validity of the inputs
        if len(control_points.shape) != 2 or control_points.shape[1] != 2:
            raise ValueError("Control points should have shape (n, 2).")

        if order is None:
            order = len(control_points) - 1
        order = int(order)
        if order < 1:
            raise ValueError("Order must be greater than or equal to one.")

        if len(control_points) != order + 1:
            raise ValueError("Number of control points must be equal to order plus one.")

        # Compute the Bezier curve points using the C++ implementation
        curve_points = cpp_Bezier.get_curve(control_points.tolist(), int(n_interpolation))

        return np.array(curve_points)
