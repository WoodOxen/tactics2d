##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: b_spline.py
# @Description: This file implements a B-spline curve interpolator.
# @Author: Yueyuan Li
# @Version: 1.0.0

import numpy as np


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
        self.degree = degree
        if degree < 0:
            raise ValueError("The degree of a B-spline curve must be non-negative.")

    def _check_validity(self, control_points: np.ndarray, knot_vectors: np.ndarray):
        if len(control_points.shape) != 2 or control_points.shape[1] != 2:
            raise ValueError("The shape of control_points is expected to be (n, 2).")

        if len(knot_vectors.shape) != 1:
            raise ValueError("The shape of knots is expected to be (t, ).")

        if len(knot_vectors) != len(control_points) + self.degree + 1:  # t + 1 = (n + 1) + p + 1
            raise ValueError(
                "The number of knots must be equal to the number of control points plus the degree of the B-spline curve plus one."
            )

        if np.any((knot_vectors[1:] - knot_vectors[:-1]) < 0):
            raise ValueError("The knot vectors must be non-decreasing.")

    def cox_deBoor(self, knot_vectors, degree: int, u: float) -> float:
        r"""Get the value of the basis function N<sub>i</sub>, p(u) at u.

        Args:
            knot_vectors (np.ndarray): The subset of knot vectors $\{u_i, u_{i+1}, \dots, u_{i+p+1}\}. The shape is $(p + 2, )$.
            degree (int): The degree of the basis function. Usually denoted as $p$ in the literature.
            u (float): The value at which the basis function is evaluated.
        """
        if degree == 0:
            return 1 if knot_vectors[0] <= u < knot_vectors[1] else 0

        N_i_p = 0
        if knot_vectors[degree] - knot_vectors[0] != 0:
            N_i_p += (
                (u - knot_vectors[0])
                / (knot_vectors[degree] - knot_vectors[0])
                * self.cox_deBoor(knot_vectors[:-1], degree - 1, u)
            )

        if knot_vectors[degree + 1] - knot_vectors[1] != 0:
            N_i_p += (
                (knot_vectors[degree + 1] - u)
                / (knot_vectors[degree + 1] - knot_vectors[1])
                * self.cox_deBoor(knot_vectors[1:], degree - 1, u)
            )

        return N_i_p

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
        knot_vectors = (
            np.linspace(0, 1, n + 1 + self.degree + 1) if knot_vectors is None else knot_vectors
        )

        self._check_validity(control_points, knot_vectors)

        curve_points = []

        us = np.linspace(
            knot_vectors[self.degree],
            knot_vectors[-self.degree - 1],
            n_interpolation,
            endpoint=False,
        )
        for u in us:
            basis_functions = np.zeros((n + 1, 1))
            for i in range(n + 1):
                N_i_p = self.cox_deBoor(knot_vectors[i : i + self.degree + 2], self.degree, u)
                basis_functions[i] = N_i_p

            curve_point = np.sum(np.multiply(basis_functions, control_points), axis=0)
            curve_points.append(curve_point)

        return np.array(curve_points)
