import numpy as np

from tactics2d.math.geometry import Circle


class Dubins:
    """
    This class implements a Dubins curve interpolator.

    The curve comprises a sequence of three segments: RSR, RSL, LSL, LSR, RLR, and LRL. R stands for the right turn, L for the left turn, and S for the straight line. A Dubins path planner operates within the constraints of forward actions exclusively.

    Attributes:
        radius (float): The minimum turning radius of the vehicle.
    """

    def __init__(self, radius: float) -> None:
        self.radius = radius
        if self.radius <= 0:
            raise ValueError("The minimum turning radius must be positive.")

    def _RSR(self, alpha, beta, dist):
        discriminant = (
            2 + dist**2 - 2 * np.cos(alpha - beta) + 2 * dist * (-np.sin(alpha) + np.sin(beta))
        )

        if discriminant < 0:
            return None

        tmp = np.arctan2(np.cos(alpha) - np.cos(beta), dist - np.sin(alpha) + np.sin(beta))

        t = np.mod(alpha - tmp, 2 * np.pi)
        p = np.sqrt(discriminant)
        q = np.mod(-(beta - tmp), 2 * np.pi)

        return t, p, q

    def _RSL(self, alpha, beta, dist):
        discriminant = (
            -2 + dist**2 + 2 * np.cos(alpha - beta) - 2 * dist * (np.sin(alpha) + np.sin(beta))
        )

        if discriminant < 0:
            return None

        p = np.sqrt(discriminant)

        tmp = np.arctan2(
            np.cos(alpha) + np.cos(beta), dist - np.sin(alpha) - np.sin(beta)
        ) - np.arctan2(2, p)

        t = np.mod(alpha - tmp, 2 * np.pi)
        q = np.mod(beta - tmp, 2 * np.pi)

        return t, p, q

    def _LSL(self, alpha, beta, dist):
        discriminant = (
            2 + dist**2 - 2 * np.cos(alpha - beta) + 2 * dist * (np.sin(alpha) - np.sin(beta))
        )

        if discriminant < 0:
            return None

        tmp = np.arctan2(-np.cos(alpha) + np.cos(beta), dist + np.sin(alpha) - np.sin(beta))

        t = np.mod(-(alpha - tmp), 2 * np.pi)
        p = np.sqrt(discriminant)
        q = np.mod(beta - tmp, 2 * np.pi)

        return t, p, q

    def _LSR(self, alpha, beta, dist):
        discriminant = (
            -2 + dist**2 + 2 * np.cos(alpha - beta) + 2 * dist * (np.sin(alpha) + np.sin(beta))
        )

        if discriminant < 0:
            return None

        p = np.sqrt(discriminant)

        tmp = -np.arctan2(
            np.cos(alpha) + np.cos(beta), dist + np.sin(alpha) + np.sin(beta)
        ) + np.arctan2(2, p)

        t = np.mod(-alpha + tmp, 2 * np.pi)
        q = np.mod(-beta + tmp, 2 * np.pi)

        return t, p, q

    def _RLR(self, alpha, beta, dist):
        discriminant = (
            6 - dist**2 + 2 * np.cos(alpha - beta) + 2 * dist * (np.sin(alpha) - np.sin(beta))
        ) / 8

        if abs(discriminant) > 1:
            return None

        tmp = np.arctan2(np.cos(alpha) - np.cos(beta), dist - np.sin(alpha) + np.sin(beta))

        p = np.mod(2 * np.pi - np.arccos(discriminant), 2 * np.pi)
        t = np.mod(alpha - tmp + p / 2, 2 * np.pi)
        q = np.mod(alpha - beta - t + p, 2 * np.pi)

        return t, p, q

    def _LRL(self, alpha, beta, dist):
        discriminant = (
            6 - dist**2 + 2 * np.cos(alpha - beta) + 2 * dist * (-np.sin(alpha) + np.sin(beta))
        ) / 8

        if abs(discriminant) > 1:
            return None

        p = np.mod(2 * np.pi - np.arccos(discriminant), 2 * np.pi)

        tmp = np.arctan2(np.cos(alpha) - np.cos(beta), dist + np.sin(alpha) - np.sin(beta))

        t = np.mod(-alpha + tmp + p / 2, 2 * np.pi)
        q = np.mod(beta - alpha - t + p, 2 * np.pi)

        return t, p, q

    class Path:
        def __init__(
            self, segments, curve_type, radius, start_point, start_heading, end_point, end_heading
        ):
            self.start_point = start_point
            self.start_heading = start_heading
            self.end_point = end_point
            self.end_heading = end_heading
            self.segments = segments
            self.curve_type = curve_type
            self.length = (abs(segments[0]) + abs(segments[1]) + abs(segments[2])) * radius

    def _set_path(self, segments, curve_type, start_point, start_heading, end_point, end_heading):
        if segments is None:
            return None

        path = self.Path(
            segments, curve_type, self.radius, start_point, start_heading, end_point, end_heading
        )
        return path

    def get_all_path(
        self,
        start_point: np.ndarray,
        start_heading: float,
        end_point: np.ndarray,
        end_heading: float,
    ):
        """Get all the Dubins paths connecting two points.

        Args:
            start_point (np.ndarray): The start point of the curve. The shape is (2,).
            start_heading (float): The heading of the start point. The unit is radian.
            end_point (np.ndarray): The end point of the curve. The shape is (2,).
            end_heading (float): The heading of the end point. The unit is radian.

        Returns:
            _type_: _description_
        """
        # create a new coordinate system with the start point as the origin
        theta = np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
        d = np.linalg.norm(end_point - start_point) / self.radius
        alpha = np.mod(start_heading - theta, 2 * np.pi)
        beta = np.mod(end_heading - theta, 2 * np.pi)

        states = (start_point, start_heading, end_point, end_heading)

        paths = [
            self._set_path(self._LRL(alpha, beta, d), "LRL", *states),
            self._set_path(self._RLR(alpha, beta, d), "RLR", *states),
            self._set_path(self._LSL(alpha, beta, d), "LSL", *states),
            self._set_path(self._RSL(alpha, beta, d), "RSL", *states),
            self._set_path(self._RSR(alpha, beta, d), "RSR", *states),
            self._set_path(self._LSR(alpha, beta, d), "LSR", *states),
        ]

        return paths

    def get_path(
        self,
        start_point: np.ndarray,
        start_heading: float,
        end_point: np.ndarray,
        end_heading: float,
    ) -> Path:
        """
        Get the shortest Dubins path connecting two points.

        Args:
            start_point (np.ndarray): The start point of the curve. The shape is (2,).
            start_heading (float): The heading of the start point. The unit is radian.
            end_point (np.ndarray): The end point of the curve. The shape is (2,).
            end_heading (float): The heading of the end point. The unit is radian.
        """
        candidate_paths = self.get_all_path(start_point, start_heading, end_point, end_heading)

        # filter out the invalid paths
        shortest_length = np.inf
        shortest_path = None
        for path in candidate_paths:
            if path is None or path.length > shortest_length:
                continue
            else:
                shortest_length = path.length
                shortest_path = path

        return shortest_path

    def get_curve_line(self, path: Path, step_size: float = 0.1):
        actions = path.curve_type
        # get the first segment
        start_circle = Circle.get_circle(
            Circle.ConstructBy.TangentVector,
            path.start_point,
            path.start_heading,
            self.radius,
            actions[0],
        )
        first_seg = Circle.get_arc(
            start_circle[0],
            self.radius,
            abs(path.segments[0]),
            (path.start_heading + np.pi / 2)
            if actions[0] == "R"
            else (path.start_heading - np.pi / 2),
            actions[0] == "R",
            step_size,
        )

        # get the third segment
        end_circle = Circle.get_circle(
            Circle.ConstructBy.TangentVector,
            path.end_point,
            path.end_heading,
            self.radius,
            actions[2],
        )
        third_seg = Circle.get_arc(
            end_circle[0],
            self.radius,
            abs(path.segments[2]),
            (path.end_heading + np.pi / 2) if actions[2] == "R" else (path.end_heading - np.pi / 2),
            actions[2] == "L",
            step_size,
        )

        # get the second segment
        if actions[0] == "R":
            theta1 = path.start_heading + np.pi / 2 - path.segments[0]
        else:
            theta1 = path.start_heading - np.pi / 2 + path.segments[0]
        point1 = start_circle[0] + np.array([np.cos(theta1), np.sin(theta1)]) * self.radius

        if actions[2] == "R":
            theta2 = path.end_heading + np.pi / 2 + path.segments[2]
        else:
            theta2 = path.end_heading - np.pi / 2 - path.segments[2]
        point2 = end_circle[0] + np.array([np.cos(theta2), np.sin(theta2)]) * self.radius

        if actions[1] == "S":
            n_points = np.linalg.norm(point2 - point1) / step_size
            second_seg = np.linspace(point1, point2, int(n_points))
        else:
            middle_heading = (
                (path.start_heading - path.segments[0])
                if actions[0] == "R"
                else (path.start_heading + path.segments[0])
            )
            middle_circle = Circle.get_circle(
                Circle.ConstructBy.TangentVector,
                point1,
                middle_heading,
                self.radius,
                actions[1],
            )
            second_seg = Circle.get_arc(
                middle_circle[0],
                self.radius,
                path.segments[1],
                (middle_heading + np.pi / 2) if actions[1] == "R" else (middle_heading - np.pi / 2),
                actions[1] == "R",
                step_size,
            )

        curve = np.concatenate((first_seg, second_seg, np.flip(third_seg, axis=0)))

        return curve

    def get_curve(
        self,
        start_point: np.ndarray,
        start_heading: float,
        end_point: np.ndarray,
        end_heading: float,
        step_size: float = 0.1,
    ):
        """Get the shortest Dubins curve connecting two points.

        Args:
            start_point (np.ndarray): The start point of the curve. The shape is (2,).
            start_heading (float): The heading of the start point. The unit is radian.
            end_point (np.ndarray): The end point of the curve. The shape is (2,).
            end_heading (float): The heading of the end point. The unit is radian.
            step_size (float, optional): The step size of the curve. Defaults to 0.1.

        Returns:
            shortest_path (Path): The shortest Dubins path connecting two points.
            curve (np.ndarray): The shortest Dubins curve connecting two points.
        """

        shortest_path = self.get_path(start_point, start_heading, end_point, end_heading)
        curve = self.get_curve_line(shortest_path, step_size)

        return shortest_path, curve
