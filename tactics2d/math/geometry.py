import numpy as np


class Circle:
    @staticmethod
    def from_three_points(pt1, pt2, pt3):
        """Derive a circle by three points.

        Returns:
            center:
            radius (float): The radius of the circle
        """
        d = -np.linalg.det(
            [
                [pt1[0] ** 2 + pt1[1] ** 2, pt1[1], 1],
                [pt2[0] ** 2 + pt2[1] ** 2, pt2[1], 1],
                [pt3[0] ** 2 + pt3[1] ** 2, pt3[1], 1],
            ]
        )
        e = np.linalg.det(
            [
                [pt1[0] ** 2 + pt1[1] ** 2, pt1[0], 1],
                [pt2[0] ** 2 + pt2[1] ** 2, pt2[0], 1],
                [pt3[0] ** 2 + pt3[1] ** 2, pt3[0], 1],
            ]
        )
        det = np.linalg.det(
            [[pt1[0], pt1[1], 1], [pt2[0], pt2[1], 1], [pt3[0], pt3[1], 1]]
        )

        D = d / det
        E = e / det

        center = [-D / 2, -E / 2]
        radius = np.linalg.norm(pt1 - center)

        return center, radius
