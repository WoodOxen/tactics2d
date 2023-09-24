import math

import numpy as np

from .curve_base import CurveBase


class Dubins(CurveBase):
    """
    This class implements a Dubins curve interpolator. The curve comprises a sequence of three segments: RSR, RSL, LSL, LSR, RLR, and LRL. R stands for the right turn, L for the left turn, and S for the straight line. A Dubins path planner operates within the constraints of forward actions exclusively.

    Attribute:
        radius (float):
    """

    def __init__(self, radius: float) -> None:
        self.radius = radius
        if self.radius <= 0:
            raise ValueError("The minimum radius must be greater than 0.")

    def _RSR(self, alpha, beta, dist):
        discriminant = (
            2 + dist**2 - 2 * np.cos(alpha - beta) + 2 * dist * (-np.sin(alpha) + np.sin(beta))
        )

        if discriminant < 0:
            return None
        else:
            tmp = np.arctan2(np.cos(alpha) - np.cos(beta), dist - np.sin(alpha) + np.sin(beta))

            p = np.sqrt(discriminant)
            t = np.mod(alpha - tmp, 2 * np.pi)
            q = np.mod(-(beta - tmp), 2 * np.pi)
        return t, p, q

    def _RSL(self, alpha, beta, dist):
        discriminant = (
            -2 + dist**2 + 2 * np.cos(alpha - beta) - 2 * dist * (np.sin(alpha) + np.sin(beta))
        )

        if discriminant < 0:
            return None
        else:
            tmp0 = np.arctan2(np.cos(alpha) + np.cos(beta), dist - np.sin(alpha) - np.sin(beta))

            p = np.sqrt(discriminant)

            tmp1 = np.arctan2(2, p)
            t = np.mod(alpha - tmp0 + tmp1, 2 * np.pi)
            q = np.mod(beta - tmp0 + tmp1, 2 * np.pi)
        return t, p, q

    def _LSL(self, alpha, beta, dist):
        discriminant = (
            2 + dist**2 - 2 * np.cos(alpha - beta) + 2 * dist * (np.sin(alpha) - np.sin(beta))
        )

        if discriminant < 0:
            return None
        else:
            tmp = np.arctan2(-np.cos(alpha) + np.cos(beta), dist + np.sin(alpha) - np.sin(beta))

            p = np.sqrt(discriminant)
            t = np.mod(-(alpha - tmp), 2 * np.pi)
            q = np.mod(beta - tmp, 2 * np.pi)
        return t, p, q

    def _LSR(self, alpha, beta, dist):
        discriminant = (
            -2 + dist**2 + 2 * np.cos(alpha - beta) + 2 * dist * (np.sin(alpha) + np.sin(beta))
        )

        if discriminant < 0:
            return None
        else:
            tmp0 = np.arctan2(-np.cos(alpha) - np.cos(beta), dist + np.sin(alpha) + np.sin(beta))

            p = np.sqrt(discriminant)

            tmp1 = np.arctan2(-2, p)
            t = np.mod(-alpha + tmp0 - tmp1, 2 * np.pi)
            q = np.mod(-beta + tmp0 - tmp1, 2 * np.pi)
        return t, p, q

    def _RLR(self, alpha, beta, dist):
        discriminant = (
            6 - dist**2 + 2 * np.cos(alpha - beta) + 2 * dist * (np.sin(alpha) - np.sin(beta))
        ) / 8

        if abs(discriminant) > 1:
            return None
        else:
            p = np.mod(2 * np.pi - np.arccos(discriminant), 2 * np.pi)
            t = np.mod(
                alpha
                - np.arctan2(np.cos(alpha) - np.cos(beta), dist - np.sin(alpha) + np.sin(beta))
                + p / 2,
                2 * np.pi,
            )
            q = np.mod(alpha - beta - t + p, 2 * np.pi)

        return t, p, q

    def _LRL(self, alpha, beta, dist):
        discriminant = (
            6 - dist**2 + 2 * np.cos(alpha - beta) + 2 * dist * (-np.sin(alpha) + np.sin(beta))
        ) / 8

        if abs(discriminant) > 1:
            return None
        else:
            p = np.mod(2 * np.pi - np.arccos(discriminant), 2 * np.pi)
            t = np.mod(
                -alpha
                + np.arctan2(-np.cos(alpha) + np.cos(beta), dist + np.sin(alpha) - np.sin(beta))
                + p / 2,
                2 * np.pi,
            )
            q = np.mod(beta - alpha - t + p, 2 * np.pi)

        return t, p, q

    def _local_coordinates(self, control_points, headings):
        return

    def get_curve(self, control_points, headings):
        return
