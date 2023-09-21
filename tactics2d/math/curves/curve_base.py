from abc import ABC, abstractmethod

import numpy as np


class CurveBase(ABC):
    @abstractmethod
    def _check_validity(self, control_points: np.ndarray):
        """Check if the control points are valid.
        """

    @abstractmethod
    def get_curve(self, control_points: np.ndarray, n_interpolation: int):
        """Get the interpolation points of a curve.

        Args:
            control_points (np.ndarray): The control points of the curve. The shape is (n, 2).
            n_interpolation (int): The number of interpolations.
        """
