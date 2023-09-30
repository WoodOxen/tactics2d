from copy import deepcopy

import numpy as np


class Bezier:
    """This class implement a Bezier curve interpolator."""

    def __init__(self, order: int):
        self.order = order
        if self.order < 1:
            raise ValueError("The order of a Bezier curve must be greater than or equal to 1.")

    def _check_validity(self, control_points: np.ndarray):
        if control_points.shape[1] != 2:
            raise ValueError("The control points must be 2D.")

        if len(control_points) != self.order + 1:
            raise ValueError(
                "The number of control points must be equal to the order of the Bezier curve plus one."
            )

    def get_curve(self, control_points: np.ndarray, n_interpolation: int):
        """
        Args:
            control_points (np.ndarray): The control points of the curve. The shape is (n, 2).
            n_interpolation (int): The number of interpolations.

        Returns:
            np.ndarray: The interpolation points of the curve. The shape is (n_interpolation, 2).
        """
        self._check_validity(control_points)

        points = []

        for n in range(n_interpolation):
            t = n / (n_interpolation - 1)
            if self.order == 1:
                point = (1 - t) * control_points[0] + t * control_points[1]
            elif self.order == 2:
                point = (
                    (1 - t) ** 2 * control_points[0]
                    + 2 * t * (1 - t) * control_points[1]
                    + t**2 * control_points[2]
                )
            else:
                iter_points = deepcopy(control_points)
                for i in range(self.order):
                    for j in range(self.order - i):
                        iter_points[j] = (1 - t) * iter_points[j] + t * iter_points[j + 1]
                point = iter_points[0]

            points.append(point)

        return np.array(points)
