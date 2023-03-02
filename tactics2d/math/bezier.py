from copy import deepcopy
from typing import List

import numpy as np


class Bezier:
    def __init__(self, order: int, n_interpolation: int):
        self.order = order
        self.n_interpolation = n_interpolation

    def _second_order(self, control_points: np.ndarray) -> List[np.ndarray]:
        points = []
        for n in range(self.n_interpolation + 1):
            t = n / self.n_interpolation
            point = (
                (1 - t) ** 2 * control_points[0]
                + 2 * t * (1 - t) * control_points[1]
                + t**2 * control_points[2]
            )
            points.append(point)
        return points

    def _higher_order(self, control_points: np.ndarray) -> List[np.ndarray]:
        """Get the interpolations of a Bezier curve by De Casteljau's algorithm."""
        points = []
        for idx in range(self.n_interpolation + 1):
            Ps = deepcopy(control_points)
            t = idx / self.n_interpolation
            for i in range(self.order):
                for j in range(self.order - i):
                    Ps[j] = (1 - t) * Ps[j] + t * Ps[j + 1]
            points.append(Ps[0])
        return points

    def get_points(self, control_points: np.ndarray) -> List[np.ndarray]:
        if control_points.shape[0] - 1 != self.order:
            raise ValueError

        if self.order == 2:
            return self._second_order(control_points)

        return self._higher_order(self)
