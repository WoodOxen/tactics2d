from copy import deepcopy

import numpy as np


class Bezier:
    """This class implement a Bezier curve interpolator."""

    def __init__(self, order: int):
        self.order = order
        if self.order < 1:
            raise ValueError("The order of a Bezier curve must be greater than or equal to 1.")

    def _check_validity(self, control_points: np.ndarray):
        if len(control_points.shape) != 2 and control_points.shape[1] != 2:
            raise ValueError("The shape of control_points is expected to be (n, 2).")

        if len(control_points) != self.order + 1:
            raise ValueError(
                "The number of control points must be equal to the order of the Bezier curve plus one."
            )

    def de_casteljau(self, points: np.ndarray, t: float, order: int) -> np.ndarray:
        """The de Casteljau algorithm for Bezier curves.

        Args:
            points (np.ndarray): The interpolation points of the curve. The shape is (order + 1, 2).
            t (float): The binomial coefficient.
            order (int): The order of the Bezier curve.
        """
        if order == 1:
            return points[0] * (1 - t) + points[1] * t
        else:
            new_points = points[:-1] * (1 - t) + points[1:] * t
            return self.de_casteljau(new_points, t, order - 1)

    def get_curve(self, control_points: np.ndarray, n_interpolation: int):
        """
        Args:
            control_points (np.ndarray): The control points of the curve. The shape is (order + 1, 2).
            n_interpolation (int): The number of interpolations.

        Returns:
            np.ndarray: The interpolation points of the curve. The shape is (n_interpolation, 2).
        """
        self._check_validity(control_points)

        curve_points = []

        interpolates = np.linspace(control_points[:-1], control_points[1:], n_interpolation)
        if self.order == 1:
            return np.squeeze(interpolates)

        for n in range(n_interpolation):
            t = n / (n_interpolation - 1)
            point = self.de_casteljau(interpolates[n], t, self.order - 1)
            curve_points.append(point)

        return np.array(curve_points)
