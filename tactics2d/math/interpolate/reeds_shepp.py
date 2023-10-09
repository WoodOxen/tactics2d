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

        tau = self._M(t1 + np.pi) if t2 < 0 else self._M(t1)
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

    def _get_segment(self, segments, matrix, actions, curve_type):
        if segments is None:
            return None

        t, u, v = segments
        if curve_type in ["CCSC", "CCSCC"]:
            segments_ = np.array([t, u, v, 1])
        else:
            segments_ = np.array([t, u, v])

        return self.Path(np.dot(segments_, matrix), actions, curve_type)

    def _CSC(self, x, y, phi):
        def LpSpLp(x, y, phi):
            """This function follows Equation 8.1 in the paper. It implements the L+S+L+ path, which can be converted to L-S-L-, R+S+R+, and R-S-R- by proper transformation."""
            u, t = self._R(x - np.sin(phi), y - 1 + np.cos(phi))

            if t < 0:
                return None

            v = self._M(phi - t)
            if v < 0:
                return None

            return (t, u, v)

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

            return (t, u, v)

        inputs = [
            (x, y, phi),
            self._time_flip(*(x, y, phi)),
            self._reflect(*(x, y, phi)),
            self._time_flip(*(self._reflect(*(x, y, phi)))),
        ]

        # L+S+L+, L-S-L-, R+S+R+, R-S-R-, L+S+R+, L-S-R-, R+S+L+, R-S-L-
        paths = [
            self._get_segment(LpSpLp(*inputs[0]), np.diag([1, 1, 1]), ["L", "S", "L"], "CSC"),
            self._get_segment(LpSpLp(*inputs[1]), np.diag([-1, -1, -1]), ["L", "S", "L"], "CSC"),
            self._get_segment(LpSpLp(*inputs[2]), np.diag([1, 1, 1]), ["R", "S", "R"], "CSC"),
            self._get_segment(LpSpLp(*inputs[3]), np.diag([-1, -1, -1]), ["R", "S", "R"], "CSC"),
            self._get_segment(LpSpRp(*inputs[0]), np.diag([1, 1, 1]), ["L", "S", "R"], "CSC"),
            self._get_segment(LpSpRp(*inputs[1]), np.diag([-1, -1, -1]), ["L", "S", "R"], "CSC"),
            self._get_segment(LpSpRp(*inputs[2]), np.diag([1, 1, 1]), ["R", "S", "L"], "CSC"),
            self._get_segment(LpSpRp(*inputs[3]), np.diag([-1, -1, -1]), ["R", "S", "R"], "CSC"),
        ]

        return paths

    def _CCC(self, x, y, phi):
        def LpRnLp(x, y, phi):
            """This function follows Equation 8.3. It implements the L+R-L+ path, which can be converted to L-R+L-, R+L+R-, and R-L-R+ by proper transformation.

            The Equation 8.3 in the original paper is wrong. Refer to the corrected version.
            """
            xi = x - np.sin(phi)
            eta = y - 1 + np.cos(phi)
            u1, theta = self._R(xi, eta)

            if u1 > 4:
                return None

            A = np.arccos(u1 / 4)
            u = self._M(np.pi - 2 * A)
            t = self._M(theta + A + np.pi / 2)
            v = self._M(phi - t - u)

            return (t, u, v)

        def LpRnLn(x, y, phi):
            """This function follows Equation 8.4. It implements the L+R-L- path, which can be converted to L-R+L+, R+L-R-, R-L+R+, L-R-L+, L+R+L-, R-L-R+, and R+L+R- by proper transformation.

            The Equation 8.4 in the original paper is wrong. Refer to the corrected version.
            """
            xi = x - np.sin(phi)
            eta = y - 1 + np.cos(phi)
            u1, theta = self._R(xi, eta)

            if u1 > 4:
                return None

            A = np.arccos(u1 / 4)
            u = self._M(np.pi - 2 * A)
            t = self._M(theta + A + np.pi / 2)
            v = self._M(phi - t + u)

            return (t, u, v)

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

        matrix1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        matrix2 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        # L+R-L+, L-R+L-, R+L-R+, R-L+R-, L+R-L-, L-R+L+, R+L-R-, R-L+R+, L-R-L+, L+R+L-, R-L-R+, R+L+R-
        paths = [
            self._get_segment(LpRnLp(*inputs[0]), matrix1, ["L", "R", "L"], "CCC"),
            self._get_segment(LpRnLp(*inputs[1]), -matrix1, ["L", "R", "L"], "CCC"),
            self._get_segment(LpRnLp(*inputs[2]), matrix1, ["R", "L", "R"], "CCC"),
            self._get_segment(LpRnLp(*inputs[3]), -matrix1, ["R", "L", "R"], "CCC"),
            self._get_segment(LpRnLn(*inputs[0]), matrix2, ["L", "R", "L"], "CCC"),
            self._get_segment(LpRnLn(*inputs[1]), -matrix2, ["L", "R", "L"], "CCC"),
            self._get_segment(LpRnLn(*inputs[2]), matrix2, ["R", "L", "R"], "CCC"),
            self._get_segment(LpRnLn(*inputs[3]), -matrix2, ["R", "L", "R"], "CCC"),
            self._get_segment(LpRnLn(*inputs[4]), matrix2, ["L", "R", "L"], "CCC"),
            self._get_segment(LpRnLn(*inputs[5]), -matrix2, ["L", "R", "L"], "CCC"),
            self._get_segment(LpRnLn(*inputs[6]), matrix2, ["R", "L", "R"], "CCC"),
            self._get_segment(LpRnLn(*inputs[7]), -matrix2, ["R", "L", "R"], "CCC"),
        ]

        return paths

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

            return (t, u, v)

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

            return (t, u, v)

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
            self._get_segment(LpRpLnRn(*inputs[0]), matrix1, ["L", "R", "L", "R"], "CCCC"),
            self._get_segment(LpRpLnRn(*inputs[1]), -matrix1, ["L", "R", "L", "R"], "CCCC"),
            self._get_segment(LpRpLnRn(*inputs[2]), matrix1, ["R", "L", "R", "L"], "CCCC"),
            self._get_segment(LpRpLnRn(*inputs[3]), -matrix1, ["R", "L", "R", "L"], "CCCC"),
            self._get_segment(LpRnLnRp(*inputs[4]), matrix2, ["L", "R", "L", "R"], "CCCC"),
            self._get_segment(LpRnLnRp(*inputs[5]), -matrix2, ["L", "R", "L", "R"], "CCCC"),
            self._get_segment(LpRnLnRp(*inputs[6]), matrix2, ["R", "L", "R", "L"], "CCCC"),
            self._get_segment(LpRnLnRp(*inputs[7]), -matrix2, ["R", "L", "R", "L"], "CCCC"),
        ]

        return paths

    def _CCSC(self, x, y, phi):
        def LpRnSnLn(x, y, phi):
            """This function follows Equation 8.9. It implements the L+R-S-L- path, which can be converted to L-R+S+L+, R+L-S-R-, R-L+S+R+, L-S-R-L+, L+S+R+L-, R-S-L-R+, and R+S+L+R- by proper transformation."""
            xi = x - np.sin(phi)
            eta = y - 1 + np.cos(phi)
            rho, theta = self._R(xi, eta)

            if rho < 2:
                return None

            _, theta1 = self._R(np.sqrt(rho**2 - 4), -2)
            t = self._M(theta - theta1)
            u = 2 - theta1
            v = self._M(phi - np.pi / 2 - t)

            if t < 0 or u > 0 or v > 0:
                return None

            return (t, u, v)

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

            return (t, u, v)

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
            self._get_segment(LpRnSnLn(*inputs[0]), matrix1, ["L", "R", "S", "L"], "CCSC"),
            self._get_segment(LpRnSnLn(*inputs[1]), -matrix1, ["L", "R", "S", "L"], "CCSC"),
            self._get_segment(LpRnSnLn(*inputs[2]), matrix1, ["R", "L", "S", "R"], "CCSC"),
            self._get_segment(LpRnSnLn(*inputs[3]), -matrix1, ["R", "L", "S", "R"], "CCSC"),
            self._get_segment(LpRnSnRn(*inputs[4]), matrix2, ["L", "S", "R", "L"], "CCSC"),
            self._get_segment(LpRnSnRn(*inputs[5]), -matrix2, ["L", "S", "R", "L"], "CCSC"),
            self._get_segment(LpRnSnRn(*inputs[6]), matrix2, ["R", "S", "L", "R"], "CCSC"),
            self._get_segment(LpRnSnRn(*inputs[7]), -matrix2, ["R", "S", "L", "R"], "CCSC"),
            self._get_segment(LpRnSnRn(*inputs[0]), matrix1, ["L", "R", "S", "R"], "CCSC"),
            self._get_segment(LpRnSnRn(*inputs[1]), -matrix1, ["L", "R", "S", "R"], "CCSC"),
            self._get_segment(LpRnSnRn(*inputs[2]), matrix1, ["R", "L", "S", "L"], "CCSC"),
            self._get_segment(LpRnSnRn(*inputs[3]), -matrix1, ["R", "L", "S", "L"], "CCSC"),
            self._get_segment(LpRnSnLn(*inputs[4]), matrix2, ["R", "S", "R", "L"], "CCSC"),
            self._get_segment(LpRnSnLn(*inputs[5]), -matrix2, ["R", "S", "R", "L"], "CCSC"),
            self._get_segment(LpRnSnLn(*inputs[6]), matrix2, ["L", "S", "L", "R"], "CCSC"),
            self._get_segment(LpRnSnLn(*inputs[7]), -matrix2, ["L", "S", "L", "R"], "CCSC"),
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

            return (t, u, v)

        inputs = [
            (x, y, phi),
            self._time_flip(*(x, y, phi)),
            self._reflect(*(x, y, phi)),
            self._time_flip(*(self._reflect(*(x, y, phi)))),
        ]

        matrix = np.array(
            [[1, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 1], [0, -np.pi / 2, 0, -np.pi / 2, 0]]
        )

        # L+R-S-L-R+, L-R+S+L+R-, R+L-S-R-L+, R-L+S+R+L-
        paths = [
            self._get_segment(LpRnSnLnRp(*inputs[0]), matrix, ["L", "R", "S", "L", "R"], "CCSCC"),
            self._get_segment(LpRnSnLnRp(*inputs[1]), -matrix, ["L", "R", "S", "L", "R"], "CCSCC"),
            self._get_segment(LpRnSnLnRp(*inputs[2]), matrix, ["R", "L", "S", "R", "L"], "CCSCC"),
            self._get_segment(LpRnSnLnRp(*inputs[3]), -matrix, ["R", "L", "S", "R", "L"], "CCSCC"),
        ]

        return paths

    def get_all_path(
        self,
        start_point: np.ndarray,
        start_heading: float,
        end_point: np.ndarray,
        end_heading: float,
    ) -> list:
        """Get all the Reeds-Shepp paths connecting two points.

        Args:
            start_point (np.ndarray): The start point of the curve. The shape is (2,).
            start_heading (float): The start heading of the curve.
            end_point (np.ndarray): The end point of the curve. The shape is (2,).
            end_heading (float): The end heading of the curve.
        """
        x = end_point[0] - start_point[0]
        y = end_point[1] - start_point[1]
        phi = end_heading - start_heading

        paths = (
            self._CCC(x, y, phi)
            + self._CSC(x, y, phi)
            + self._CCSC(x, y, phi)
            + self._CCCC(x, y, phi)
            + self._CCSCC(x, y, phi)
        )

        return paths
    
    def get_path(
        self,
        start_point: np.ndarray,
        start_heading: float,
        end_point: np.ndarray,
        end_heading: float,
    ):
        """Get the shortest Reeds-Shepp path connecting two points.

        Args:
            start_point (np.ndarray): The start point of the curve. The shape is (2,).
            start_heading (float): The start heading of the curve.
            end_point (np.ndarray): The end point of the curve. The shape is (2,).
            end_heading (float): The end heading of the curve.
        """
        candidate_paths = self.get_all_path(start_point, start_heading, end_point, end_heading)

        shortest_path = None
        shortest_length = np.inf
        for path in candidate_paths:
            if path is not None and path.length < shortest_length:
                shortest_path = path
                shortest_length = path.length

        return shortest_path

    def get_curve_line(self, path: Path):
        return

    def get_curve(
        self,
        start_point: np.ndarray,
        start_heading: float,
        end_point: np.ndarray,
        end_heading: float,
    ):
        """Get the shortest Reeds-Shepp curve connecting two points.

        Args:
            start_point (np.ndarray): The start point of the curve. The shape is (2,).
            start_heading (float): The start heading of the curve.
            end_point (np.ndarray): The end point of the curve. The shape is (2,).
            end_heading (float): The end heading of the curve.
        """
        shortest_path = self.get_path(start_point, start_heading, end_point, end_heading)
        

        return
