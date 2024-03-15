import numpy as np
from scipy.special import fresnel


class Spiral:
    """This class implements a spiral interpolation."""

    def __init__(self, gamma: float):
        """Initialize the spiral interpolation.

        Args:
            gamma (float): The curvature of the spiral.
        """
        self.gamma = gamma

    @staticmethod
    def get_spiral(
        s: np.ndarray,
        start_point: np.ndarray,
        heading: float,
        curv_start: float,
        gamma: float,
    ) -> np.ndarray:
        """This function gets the points on a spiral curve line.

        Args:
            s (np.ndarray): The distance from the start point. The shape is (n_interpolate,).
            start_point (np.ndarray): The start point of the spiral. The shape is (2,).
            heading (float): The heading of the start point. The unit is radian.
            curv_start (float): The curvature of the start point.
            gamma (float): The rate of change of curvature

        Returns:
            np.ndarray: The points on the spiral curve line. The shape is (n_interpolate, 2).
        """

        # # Start
        x_start, y_start = start_point

        if gamma == 0 and curv_start == 0:
            # Straight line
            x_interpolated = x_start + s * np.cos(heading)
            y_interpolated = y_start + s * np.sin(heading)

        elif gamma == 0 and curv_start != 0:
            # Arc
            x_interpolated = x_start + (1 / curv_start) * (np.sin(heading + curv_start * s) - np.sin(heading))
            y_interpolated = y_start + (1 / curv_start) * (np.cos(heading) - np.cos(heading + curv_start * s))

        else:
            scaler = np.sqrt(np.pi * np.abs(gamma))
            # Fresnel integrals
            S_interpolated, C_interpolated = fresnel(
                (curv_start + gamma * s) / scaler
            )
            S_start, C_start = fresnel(curv_start / scaler)

            # Euler Spiral
            Cs1 = np.sqrt(np.pi / np.abs(gamma)) * np.exp(
                1j * (heading - curv_start ** 2 / 2 / gamma)
            )
            Cs2 = np.sign(gamma) * (C_interpolated - C_start) + 1j * S_interpolated - 1j * S_start
            delta_C = Cs1 * Cs2
            x_interpolated = x_start + delta_C.real
            y_interpolated = y_start + delta_C.imag


        return np.array([x_interpolated, y_interpolated]).T
    