import numpy as np


class ReedsShepp:
    """This class implements a Reeds Shepp curve interpolator. The implementation follows the paper "Optimal paths for a car that goes both forwards and backwards" by Reeds and Shepp."""

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

    def _tau_omega(self, u, v, xi, eta, phi):
        """This function follows Equation 8.5 and 8.6 in the paper."""
        delta = self._M(u - v)
        A = np.sin(u) - np.sin(delta)
        B = np.cos(u) - np.cos(delta) - 1

        _, t1 = self._R(xi * A + eta * B, eta * A - xi * B)
        t2 = 2 * (np.cos(delta) - np.cos(v) - np.cos(u)) + 3

        if t2 < 0:
            tau = self._M(t1 + np.pi)
        else:
            tau = self._M(t1)

        omega = self._M(tau - u + v - phi)

        return tau, omega

    def _time_flip(self, x, y, phi):
        return (-x, y, -phi)

    def _reflect(self, x, y, phi):
        return (x, -y, -phi)

    def _backward(self, x, y, phi):
        x_ = x * np.cos(phi) + y * np.sin(phi)
        y_ = x * np.sin(phi) - y * np.cos(phi)
        return (x_, y_, phi)

    class Path:
        def __init__(self, lengths, actions, curve_type):
            self.lengths = lengths
            self.actions = actions
            self.curve_type = curve_type

        @property
        def length(self):
            return sum(np.abs(self.lengths))

    def _get_path(self, segment_lengths, matrix, actions, curve_type):
        if segment_lengths is None:
            return None

        lengths = np.dot(segment_lengths, matrix)
        return self.Path(lengths, actions, curve_type)

    def _CSC(self, x, y, phi):
        def LpSpLp(x, y, phi):
            """This function follows Equation 8.1 in the paper. It implements the L+S+L+ path, which can be converted to L-S-L-, R+S+R+, and R-S-R- by proper transformation."""
            u, t = self._R(x - np.sin(phi), y - 1 + np.cos(phi))

            if t < 0:
                return None

            v = self._M(phi - t)
            if v < 0:
                return None

            return np.array([t, u, v])

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

            return np.array([t, u, v])

        inputs = [
            (x, y, phi),
            self._time_flip(*(x, y, phi)),
            self._reflect(*(x, y, phi)),
            self._time_flip(*(self._reflect(*(x, y, phi)))),
        ]

        # L+S+L+, L-S-L-, R+S+R+, R-S-R-, L+S+R+, L-S-R-, R+S+L+, R-S-L-
        paths = [
            self._get_path(LpSpLp(*inputs[0]), np.diag([1, 1, 1]), ["L", "S", "L"], "CSC"),
            self._get_path(LpSpLp(*inputs[1]), np.diag([-1, -1, -1]), ["L", "S", "L"], "CSC"),
            self._get_path(LpSpLp(*inputs[2]), np.diag([1, 1, 1]), ["R", "S", "R"], "CSC"),
            self._get_path(LpSpLp(*inputs[3]), np.diag([-1, -1, -1]), ["R", "S", "R"], "CSC"),
            self._get_path(LpSpRp(*inputs[0]), np.diag([1, 1, 1]), ["L", "S", "R"], "CSC"),
            self._get_path(LpSpRp(*inputs[1]), np.diag([-1, -1, -1]), ["L", "S", "R"], "CSC"),
            self._get_path(LpSpRp(*inputs[2]), np.diag([1, 1, 1]), ["R", "S", "L"], "CSC"),
            self._get_path(LpSpRp(*inputs[3]), np.diag([-1, -1, -1]), ["R", "S", "R"], "CSC"),
        ]

        return paths

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

    def _CCCC(self, x, y, phi):
        def LpRpLnRn(x, y, phi):
            """This function follows Equation 8.7. It implements the L+R+L-R- path, which can be converted to L-R-L+R+, R+L+R-L-, and R-L-R+L+ by proper transformation."""
            xi = x + np.sin(phi)
            eta = y - 1 - np.cos(phi)
            rho = (2 + np.sqrt(xi**2 + eta**2)) / 4

            if rho > 1 or rho < 0:
                return None

            u = np.arccos(rho)
            t, v = self._tau_omega(u, -u, xi, eta, phi)

            if t < 0 or v > 0:
                return None

            return np.array([t, u, v])

        def LpRnLnRp(x, y, phi):
            """This function follows Equation 8.8. It implements the L+R-L-R+ path, which can be converted to L-R+L+R-, R+L-R-L+, and R-L+R-L+ by proper transformation."""
            xi = x + np.sin(phi)
            eta = y - 1 - np.cos(phi)
            rho = (20 - xi**2 - eta**2) / 16

            if rho > 1 or rho < 0:
                return None

            u = -np.arccos(rho)
            t, v = self._tau_omega(u, u, xi, eta, phi)

            if t < 0 or v < 0:
                return None

            return np.array([t, u, v])

        inputs = [
            (x, y, phi),
            self._time_flip(*(x, y, phi)),
            self._reflect(*(x, y, phi)),
            self._time_flip(*(self._reflect(*(x, y, phi)))),
            self._backward(*(x, y, phi)),
            self._time_flip(*(self._backward(*(x, y, phi)))),
            self._reflect(*(self._backward(*(x, y, phi)))),
            self._time_flip(*(self._reflect(*(self._backward(*(x, y, phi)))))),
        ]

        matrix1 = np.array([[1, 0, 0, 0], [0, 1, -1, 0], [0, 0, 0, 1]])
        matrix2 = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]])

        # L+R+L-R-, L-R-L+R+, R+L+R-L-, R-L-R+L+, L+R-L+R-, L-R+L-R+, R+L-R-L-, R-L+R-L+
        paths = [
            self._get_path(LpRpLnRn(*inputs[0]), matrix1, ["L", "R", "L", "R"], "CCCC"),
            self._get_path(LpRpLnRn(*inputs[1]), -matrix1, ["L", "R", "L", "R"], "CCCC"),
            self._get_path(LpRpLnRn(*inputs[2]), matrix1, ["R", "L", "R", "L"], "CCCC"),
            self._get_path(LpRpLnRn(*inputs[3]), -matrix1, ["R", "L", "R", "L"], "CCCC"),
            self._get_path(LpRnLnRp(*inputs[4]), matrix2, ["L", "R", "L", "R"], "CCCC"),
            self._get_path(LpRnLnRp(*inputs[5]), -matrix2, ["L", "R", "L", "R"], "CCCC"),
            self._get_path(LpRnLnRp(*inputs[6]), matrix2, ["R", "L", "R", "L"], "CCCC"),
            self._get_path(LpRnLnRp(*inputs[7]), -matrix2, ["R", "L", "R", "L"], "CCCC"),
        ]

        return paths

    def _CCSC(self, x, y, phi):
        def LpRnSnLn(x, y, phi):
            """This function follows Equation 8.9. It implements the L+R-S-L- path, which can be converted to L-R+S+L+, R+L-S-R-, R-L+S+R+, L-S-R-L+, L+S+R+L-, R-S-L-R+, and R+S+L+R- by proper transformation."""
            xi = x + np.sin(phi)
            eta = y - 1 - np.cos(phi)
            rho, theta = self._R(-eta, xi)

            if rho < 2:
                return None

            _, theta1 = self._R(np.sqrt(rho**2 - 4), -2)
            t = self._M(theta - theta1)
            u = 2 - theta1
            v = self._M(phi - np.pi / 2 - t)

            if t < 0 or u > 0 or v > 0:
                return None

            return np.array([t, u, v, 1])

        def LpRnSnRn(x, y, phi):
            """This function follows Equation 8.10. It implements the L+R-S-R- path, which can be converted to L-R+S+R+, R+L-S-L-, R-L+S+L+, R-S-R-L+, R+S+R+L-, L-S-L-R+, and L+S+L+R- by proper transformation."""
            xi = x + np.sin(phi)
            eta = y - 1 - np.cos(phi)
            rho, theta = self._R(-eta, xi)

            if rho < 2:
                return None

            t = theta
            u = 2 - rho
            v = self._M(t + np.pi / 2 - phi)

            if t < 0 or u > 0 or v > 0:
                return None

            return np.array([t, u, v, 1])

        inputs = [
            (x, y, phi),
            self._time_flip(*(x, y, phi)),
            self._reflect(*(x, y, phi)),
            self._time_flip(*(self._reflect(*(x, y, phi)))),
            self._backward(*(x, y, phi)),
            self._time_flip(*(self._backward(*(x, y, phi)))),
            self._reflect(*(self._backward(*(x, y, phi)))),
            self._time_flip(*(self._reflect(*(self._backward(*(x, y, phi)))))),
        ]

        matrix1 = np.array([[1, 0, 0, 0], [0, 0, 0, -np.pi / 2], [0, 1, 0, 0], [0, 0, 1, 0]])
        matrix2 = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, -np.pi / 2], [1, 0, 0, 0]])

        # L+R-S-L-, L-R+S+L+, R+L-S-R-, R-L+S+R+, L-S-R-L+, L+S+R+L-, R-S-L-R+, R+S+L+R-
        # L+R-S-R-, L-R+S+R+, R+L-S-L-, R-L+S+L+, R-S-R-L+, R+S+R+L-, L-S-L-R-, L+S+L+R-,
        paths = [
            self._get_path(LpRnSnLn(*inputs[0]), matrix1, ["L", "R", "S", "L"], "CCSC"),
            self._get_path(LpRnSnLn(*inputs[1]), -matrix1, ["L", "R", "S", "L"], "CCSC"),
            self._get_path(LpRnSnLn(*inputs[2]), matrix1, ["R", "L", "S", "R"], "CCSC"),
            self._get_path(LpRnSnLn(*inputs[3]), -matrix1, ["R", "L", "S", "R"], "CCSC"),
            self._get_path(LpRnSnRn(*inputs[4]), matrix2, ["L", "S", "R", "L"], "CCSC"),
            self._get_path(LpRnSnRn(*inputs[5]), -matrix2, ["L", "S", "R", "L"], "CCSC"),
            self._get_path(LpRnSnRn(*inputs[6]), matrix2, ["R", "S", "L", "R"], "CCSC"),
            self._get_path(LpRnSnRn(*inputs[7]), -matrix2, ["R", "S", "L", "R"], "CCSC"),
            self._get_path(LpRnSnRn(*inputs[0]), matrix1, ["L", "R", "S", "R"], "CCSC"),
            self._get_path(LpRnSnRn(*inputs[1]), -matrix1, ["L", "R", "S", "R"], "CCSC"),
            self._get_path(LpRnSnRn(*inputs[2]), matrix1, ["R", "L", "S", "L"], "CCSC"),
            self._get_path(LpRnSnRn(*inputs[3]), -matrix1, ["R", "L", "S", "L"], "CCSC"),
            self._get_path(LpRnSnLn(*inputs[4]), matrix2, ["R", "S", "R", "L"], "CCSC"),
            self._get_path(LpRnSnLn(*inputs[5]), -matrix2, ["R", "S", "R", "L"], "CCSC"),
            self._get_path(LpRnSnLn(*inputs[6]), matrix2, ["L", "S", "L", "R"], "CCSC"),
            self._get_path(LpRnSnLn(*inputs[7]), -matrix2, ["L", "S", "L", "R"], "CCSC"),
        ]

        return paths

    def _CCSCC(self, x, y, phi):
        def LpRnSnLnRp(x, y, phi):
            """This function follows Equation 8.11. It implements the L+R-S-L-R+ path, which can be converted to L-R+S+L+R-, R+L-S-R-L+, and R-L+S+R+L- by proper transformation."""
            xi = x + np.sin(phi)
            eta = y - 1 - np.cos(phi)
            rho, theta = self._R(xi, eta)

            if rho < 2:
                return None

            t = self._M(theta - np.arccos(-2 / rho))

            if t <= 0:
                return None

            u = 4 - (xi + 2 * np.cos(t)) / np.sin(t)
            v = self._M(t - phi)

            if u > 0 or v < 0:
                return None

            return np.array([t, u, v, 1])

        inputs = [
            (x, y, phi),
            self._time_flip(*(x, y, phi)),
            self._reflect(*(x, y, phi)),
            self._time_flip(*(self._reflect(*(x, y, phi)))),
        ]

        matrix = np.array(
            [1, 0, 0, 0, 0],
            [0, 0, 0, -np.pi / 2, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, -np.pi / 2, 0],
            [0, 0, 1, 0, 0],
        )

        # L+R-S-L-R+, L-R+S+L+R-, R+L-S-R-L+, R-L+S+R+L-
        paths = [
            self._get_path(LpRnSnLnRp(*inputs[0]), matrix, ["L", "R", "S", "L", "R"], "CCSCC"),
            self._get_path(LpRnSnLnRp(*inputs[1]), -matrix, ["L", "R", "S", "L", "R"], "CCSCC"),
            self._get_path(LpRnSnLnRp(*inputs[2]), matrix, ["R", "L", "S", "R", "L"], "CCSCC"),
            self._get_path(LpRnSnLnRp(*inputs[3]), -matrix, ["R", "L", "S", "R", "L"], "CCSCC"),
        ]

        return paths

    def get_curve(self, control_points, headings, type):
        return
