##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: b_spline.py
# @Description: This file implements a B-spline curve interpolator.
# @Author: Tactics2D Team
# @Version: 0.1.9

import numpy as np
from cpp_interpolator import BSpline as cpp_BSpline


class BSpline:
    """This class implements a B-spline curve interpolator.

    Attributes:
        degree (int): The degree of the B-spline curve. Usually denoted as $p$ in the literature. The degree of a B-spline curve is equal to the degree of the highest degree basis function. The order of a B-spline curve is equal to $p + 1$.
    """

    def __init__(self, degree: int):
        """Initialize the B-spline curve interpolator.

        Args:
            degree (int): The degree of the B-spline curve. Usually denoted as p in the literature.

        Raises:
            ValueError: The degree of a B-spline curve must be non-negative.
        """
        if degree < 0:
            raise ValueError("BSpline interpolator: Degree must be non-negative.")
        self.degree = degree

    def _check_validity(self, control_points: np.ndarray, knot_vectors: np.ndarray):
        if control_points.ndim != 2 or control_points.shape[1] != 2:
            raise ValueError("BSpline interpolator: Control points should have shape (n, 2).")

        expected_knots = len(control_points) + self.degree + 1
        if knot_vectors.ndim != 1 or len(knot_vectors) != expected_knots:
            raise ValueError(
                f"BSpline interpolator: Expected {expected_knots} knots, got {len(knot_vectors)}."
            )

        if np.any(np.diff(knot_vectors) < 0):
            raise ValueError("BSpline interpolator: Knot vector must be non-decreasing.")

    def get_curve(
        self,
        control_points: np.ndarray,
        knot_vectors: np.ndarray = None,
        n_interpolation: int = 100,
    ) -> np.ndarray:
        r"""Get the interpolation points of a b-spline curve.

        Args:
            control_points (np.ndarray): The control points of the curve. Usually denoted as $P_0, P_1, \dots, P_n$ in literature. The shape is $(n + 1, 2)$.
            knot_vectors (np.ndarray): The knots of the curve. Usually denoted as $u_0, u_1, \dots, u_t$ in literature. The shape is $(t + 1, )$.
            n_interpolation (int): The number of interpolation points.

        Returns:
            curve_points (np.ndarray): The interpolation points of the curve. The shape is (n_interpolation, 2).
        """
        n = len(control_points) - 1
        if knot_vectors is None:
            knot_vectors = np.linspace(0, 1, n + self.degree + 2)

        self._check_validity(control_points, knot_vectors)

        curve_points = cpp_BSpline.get_curve(
            control_points.tolist(), knot_vectors.tolist(), self.degree, n_interpolation
        )
        return np.array(curve_points)
