import numpy as np

from .curve_base import CurveBase


class CubicSpline(CurveBase):
    """This class implement a cubic spline interpolator.
    
    Attributes:

    """

    def __init__(self, boundary_type: str = "not-a-knot"):
        self.boundary_type = boundary_type
        if self.boundary_type not in ["natural", "clamped", "not-a-knot"]:
            raise ValueError("The boundary type must be .")
        
    def _check_validity(self, control_points: np.ndarray):
        if control_points.shape[1] != 2:
            raise ValueError("The control points must be 2D.")

        if len(control_points) < 3:
            raise ValueError(
                "There is not enough control points to interpolate a cubic spline curve."
            )

    def get_expression(self, control_points: np.ndarray) -> np.ndarray:
        """Get the interpolation expressions of a curve.

        Args:
            control_points (np.ndarray): The control points of the curve. The shape is (n, 2).

        Returns:
            np.ndarray: The parameters of the interpolation expressions. The shape is (n - 1, 4). The first column is the coefficients of t^3, the second column is the coefficients of t^2, the third column is the coefficients of t, and the last column is the constant.
        """
        self._check_validity(control_points)

        n = len(control_points)
        A = np.zeros((n, n))
        b = np.zeros((n, 2))

        # Set the boundary conditions.
        if self.boundary_type == "natural":
            A[0, 0] = 1
            A[-1, -1] = 1
        elif self.boundary_type == "clamped":
            A[0, 0] = 2
            A[0, 1] = 1
            A[-1, -1] = 2
            A[-1, -2] = 1

    def get_curve(self, control_points: np.ndarray, n_interpolation: int):
        """Get the interpolation points of a curve.

        Args:
            n_interpolation (int): The number of interpolations.
        """
        self._check_validity(control_points)
