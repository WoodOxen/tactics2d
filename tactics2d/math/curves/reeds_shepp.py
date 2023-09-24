import numpy as np

from .curve_base import CurveBase


class ReedsShepp(CurveBase):
    """This class implements a Reeds Shepp curve interpolator. The implementation follows the paper "Optimal paths for a car that goes both forwards and backwards" by Reeds and Shepp.

    Args:
        CurveBase (_type_): _description_
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _R(x, y):
        """Convert cartesian coordinates to polar coordinates."""
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return r, theta

    @staticmethod
    def _M(theta):
        """Regulate a given angle to the range of [-pi, pi]."""
        phi = np.mod(theta, 2 * np.pi)

        if phi > np.pi:
            phi -= 2 * np.pi
        if phi < -np.pi:
            phi += 2 * np.pi

        return phi

    def _CSC(self):
        def LpSpLp(x, y, phi):
            """This function follows Equation 8.1 in the paper. It implements the L+S+L+ path, which can be converted to L-S-L-, R+S+R+, and R-S-R- by proper transformation."""
            u, t = self._R(x - np.sin(phi), y - 1 + np.cos(phi))

            if t < 0:
                return None

            v = self._M(phi - t)
            if v < 0:
                return None

            return t, u, v

        def LpSpRp(x, y, phi):
            """This function follows Equation 8.2 in the paper. It implements the L+S+R+ path, which can be converted to L-S-R-, R+S+L+, and R-S-L+ by proper transformation."""
            u1, t1 = self._R(x + np.sin(phi), y - 1 - np.cos(phi))

            if u1**2 < 4:
                return None

            u = np.sqrt(u1**2 - 4)
            _, theta = self._R(u, 2)
            t = self._M(t1 + theta)
            v = self._M(t - phi)

            if t < 0 or v < 0:
                return None

            return t, u, v

        return

    def _CCC(self):
        def LRL(x, y, phi):
            """This function follows Equation 8.3. It implements the L+R-L+ path, which can be converted to L-R+L-, R+L+R-, and R-L-R+ by proper transformation. Since Equation 8.4 is the same as Equation 8.3, by this function we can obtain L+R-L-, L-R+L+, R+L-R-, R-L+R+, L-R-L+, L+R+L-, R-L-R+, and R+L+R-."""
            xi = x - np.sin(phi)
            eta = y - 1 + np.cos(phi)
            u1, theta = self._R(xi, eta)

            if u1**2 > 4:
                return None

            A = np.arcsin(u1**2 / 4)
            u = self._M(A + theta)
            _, v = self._R(2 - xi * np.sin(u) + eta * np.cos(u), xi * np.cos(u) + eta * np.sin(u))

            return

        return

    def _CCCC(self):
        def LpRpLmRm(x, y, phi):
            """This function follows Equation 8.7. It implements the L+R+L-R- path, which can be converted to L-R-L+R+, R+L+R-L-, and R-L-R+L+ by proper transformation."""
            xi = x + np.sin(phi)
            eta = y - 1 - np.cos(phi)
            rho = (2 + np.sqrt(xi**2 + eta**2)) / 4

            return

        def LpRmLmRp(x, y, phi):
            """This function follows Equation 8.8. It implements the L+R-L-R+ path, which can be converted to L-R+L+R-, R+L-R-L+, and R-L+R-L+ by proper transformation."""
            xi = x + np.sin(phi)
            eta = y - 1 - np.cos(phi)
            rho = (20 - xi**2 - eta**2) / 16

            return

        return

    def _CCSC(self):
        def LpRmSmLm(x, y, phi):
            """This function follows Equation 8.9. It implements the L+R-S-L- path, which can be converted to L-R+S+L+, R+L-S-R-, R-L+S+R+, L-S-R-L+, L+S+R+L-, R-S-L-R+, and R+S+L+R- by proper transformation."""
            xi = x + np.sin(phi)
            eta = y - 1 - np.cos(phi)
            rho, theta = self._R(-eta, xi)

            if rho < 2:
                return None

            T, theta1 = self._R(np.sqrt(rho**2 - 4), -2)
            t = self._M(theta - theta1)
            u = 2 - theta1
            v = self._M(phi - np.pi / 2 - t)

            return t, u, v

        def LpRmSmRm(x, y, phi):
            """This function follows Equation 8.10. It implements the L+R-S-R- path, which can be converted to L-R+S+R+, R+L-S-L-, R-L+S+L+, R-S-R-L+, R+S+R+L-, L-S-L-R+, and L+S+L+R- by proper transformation."""
            xi = x + np.sin(phi)
            eta = y - 1 - np.cos(phi)
            rho, theta = self._R(-eta, xi)

            if rho < 2:
                return None

            t = theta
            u = 2 - rho
            v = self._M(t + np.pi / 2 - phi)

            return t, u, v

        return

    def _CCSCC(self):
        def LpRmSmLmRp(x, y, phi):
            """This function follows Equation 8.11. It implements the L+R-S-L-R+ path, which can be converted to L-R+S+L+R-, R+L-S-R-L+, and R-L+S+R+L- by proper transformation."""
            xi = x + np.sin(phi)
            eta = y - 1 - np.cos(phi)
            rho, theta = self._R(xi, eta)

            if rho < 2:
                return None

            t = self.M(theta - np.arccos(-2 / rho))
            if t < 0:
                return None

            u = 4 - (xi + 2 * np.cos(t)) / np.sin(t)

            return

        return

    def _time_flip(self):
        return

    def _reflect(self):
        return

    def _backwards(self):
        return

    def get_curve(self, control_points, headings):
        return
