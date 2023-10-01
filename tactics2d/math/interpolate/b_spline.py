from enum import Enum

import numpy as np


class BSpline:
    """This class implements a B-spline curve interpolator.

    Attributes:
        order (int): The order of the B-spline curve.
        boundary_type (int, optional): Boundary condition type. The B-spline curve interpolator offers three distinct boundary condition options: uniform (1), open uniform (2), and non-uniform (3). By default, the uniform boundary condition is applied, serving as a wise choice when specific boundary condition information is unavailable.
    """

    class BoundaryType(Enum):
        Uniform = 1
        OpenUniform = 2
        NonUniform = 3

    def __init__(self, order: int, boundary_type=BoundaryType.Uniform):
        """
        Args:
            order (int): _description_
            boundary_type (_type_, optional): _description_. Defaults to BoundaryType.Uniform.

        Raises:
            ValueError: _description_
        """
        self.order = order
        self.boundary_type = boundary_type
        if self.boundary_type not in self.BoundaryType.__members__.values():
            raise ValueError(
                "The boundary type is not valid. Please choose from BSpline.BoundaryType.Uniform, BSpline.BoundaryType.OpenUniform, and BSpline.BoundaryType.NonUniform."
            )

    def _check_validity(self, control_points: np.ndarray, knots: np.ndarray):
        if len(control_points.shape) != 2 and control_points.shape[1] != 2:
            raise ValueError("The shape of control_points is expected to be (n, 2).")

        if knots is None:
            pass
        else:
            if len(knots.shape) != 1:
                raise ValueError("The shape of knots is expected to be (t, ).")
            if len(knots) != len(control_points) + self.order + 1:
                raise ValueError(
                    "The number of knots must be equal to the number of control points plus the order of the B-spline curve plus one."
                )

    def cox_deBoor(
        self,
    ):
        return

    def get_curve(
        self, control_points: np.ndarray, knots: np.ndarray = None, n_interpolation: int = 100
    ) -> np.ndarray:
        """Get the interpolation points of a b-spline curve.

        Args:
            control_points (np.ndarray): control_points (np.ndarray): The control points of the curve. The shape is (n + 1, 2).
            knots (np.ndarray): The knots of the curve. The shape is (t + 1, ).

        """
        self._check_validity(control_points, knots)

        if knots is None:
            knots = np.linspace(0, 1, len(control_points) + self.order + 1)

        curve_points = []

        return
