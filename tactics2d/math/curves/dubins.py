import numpy as np

from .curve_base import CurveBase


class Dubins(CurveBase):
    """This class implement a Dubins curve interpolator. The curve is composed of a sequence of three segments: RSR, RSL, LSL, LSR, RLR, LRL. R stands for right turn, L stands for left turn, S stands for straight line.

    Args:
        CurveBase (_type_): _description_
    """

    def __init__(self, min_radius: float) -> None:
        self.min_radius = min_radius
        if self.min_radius <= 0:
            raise ValueError("The minimum radius must be greater than 0.")

    def _RSR(self, alpha, beta, dist):
        p_squared = (
            2 + dist**2 - 2 * np.cos(alpha - beta) + 2 * dist * (np.sin(alpha) - np.sin(beta))
        )

        if p_squared < 0:
            return None
        else:
            tmp = np.arctan2(np.cos(alpha) - np.cos(beta), dist - np.sin(alpha) + np.sin(beta))

            p = np.sqrt(p_squared)
            t = np.mod(alpha - tmp, 2 * np.pi)
            q = np.mod(-(beta - tmp), 2 * np.pi)
        return t, p, q

    def _RSL(self, alpha, beta, dist):
        p_squared = (
            -2 + dist**2 + 2 * np.cos(alpha - beta) - 2 * dist * (np.sin(alpha) + np.sin(beta))
        )

        if p_squared < 0:
            return None
        else:
            tmp0 = np.arctan2(np.cos(alpha) + np.cos(beta), dist - np.sin(alpha) - np.sin(beta))

            p = np.sqrt(p_squared)

            tmp1 = np.arctan2(2, p)
            t = np.mod(alpha - tmp0 + tmp1, 2 * np.pi)
            q = np.mod(beta - tmp0 + tmp1, 2 * np.pi)
        return t, p, q

    def _LSL(self, alpha, beta, dist):
        p_squared = (
            2 + dist**2 - 2 * np.cos(alpha - beta) + 2 * dist * (np.sin(alpha) - np.sin(beta))
        )

        if p_squared < 0:
            return None
        else:
            tmp = np.mod(
                np.arctan2(-np.cos(alpha) + np.cos(beta), dist + np.sin(alpha) - np.sin(beta)),
                2 * np.pi,
            )

            p = np.sqrt(p_squared)
            t = -(alpha - tmp)
            q = beta - tmp
        return t, p, q

    def _LSR(self, alpha, beta, dist):
        p_squared = (
            -2 + dist**2 + 2 * np.cos(alpha - beta) + 2 * dist * (np.sin(alpha) + np.sin(beta))
        )

        if p_squared < 0:
            return None
        else:
            tmp0 = np.arctan2(-np.cos(alpha) - np.cos(beta), dist + np.sin(alpha) + np.sin(beta))

            p = np.sqrt(p_squared)

            tmp1 = np.arctan2(-2, p)
            t = np.mod(-(alpha - tmp0 - tmp1), 2 * np.pi)
            q = np.mod(-(beta + tmp0 - tmp1), 2 * np.pi)
        return t, p, q

    def _RLR(self, alpha, beta, dist):
        return

    def _LRL(self, alpha, beta, dist):
        return

    def get_curve(self, control_points, headings):
        return
