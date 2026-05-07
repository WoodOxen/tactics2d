# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""B spline implementation."""


import numpy as np

from cpp_interpolator import BSpline as cpp_BSpline


class BSpline:
    """This class implements a B-spline curve interpolator."""

    @staticmethod
    def get_curve(
        control_points: np.ndarray,
        knot_vectors: np.ndarray = None,
        degree: int = 0,
        n_interpolation: int = 100,
    ) -> np.ndarray:
        r"""Get the interpolation points of a b-spline curve.

        Args:
            control_points (np.ndarray): The control points of the curve. Usually denoted as $P_0, P_1, \dots, P_n$ in literature. The shape is $(n + 1, 2)$.
            knot_vectors (np.ndarray, optional): The knots of the curve. Usually denoted as $u_0, u_1, \dots, u_t$ in literature. The shape is $(t + 1, )$. Defaults to None.
            degree (int, optional): The degree of the B-spline curve. Usually denoted as p in the literature. Defaults to 0.
            n_interpolation (int, optional): The number of interpolation points. Defaults to 100.

        Returns:
            curve_points (np.ndarray): The interpolation points of the curve. The shape is (n_interpolation, 2).
        """

        n = len(control_points) - 1
        if knot_vectors is None:
            knot_vectors = np.linspace(0, 1, n + degree + 2)

        # Check the validity of the inputs
        degree = int(degree)
        if degree < 0:
            raise ValueError("Degree must be non-negative.")

        if control_points.ndim != 2 or control_points.shape[1] != 2:
            raise ValueError("Control points should have shape (n, 2).")

        expected_knots = len(control_points) + degree + 1
        if knot_vectors.ndim != 1 or len(knot_vectors) != expected_knots:
            raise ValueError(f"Expected {expected_knots} knots, got {len(knot_vectors)}.")

        if np.any(np.diff(knot_vectors) < 0):
            raise ValueError("Knot vector must be non-decreasing.")

        # Compute the B-spline curve points using the C++ implementation
        curve_points = cpp_BSpline.get_curve(
            control_points.tolist(), knot_vectors.tolist(), degree, int(n_interpolation)
        )
        return np.array(curve_points)
