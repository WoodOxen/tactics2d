#! python3
# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# @File: param_poly3.py
# @Description: This file implements a parametric cubic polynomial curve interpolation.
# @Author: Zexi Chen
# @Version: 1.0.0

import numpy as np


class ParamPoly3:
    """This class implements a parametric cubic polynomial curve interpolation."""

    @staticmethod
    def get_curve(
        length: float,
        start_point: tuple,
        heading: float,
        aU: float, bU: float, cU: float, dU: float,
        aV: float, bV: float, cV: float, dV: float,
        p_range_type: str = "normalized",
    ) -> np.ndarray:
        """Get the points on a paramPoly3 curve defined in the OpenDRIVE standard.

        The curve is parameterised by p, where the local U and V coordinates are
        computed as cubic polynomials in p.  The result is then rotated by the
        road heading and translated to the start point to yield world coordinates.

        Args:
            length (float): The arc length of the curve segment.
            start_point (tuple): The (x, y) world coordinates of the curve start.
            heading (float): The heading angle at the start point in radians.
            aU (float): Constant coefficient of the U polynomial.
            bU (float): Linear coefficient of the U polynomial.
            cU (float): Quadratic coefficient of the U polynomial.
            dU (float): Cubic coefficient of the U polynomial.
            aV (float): Constant coefficient of the V polynomial.
            bV (float): Linear coefficient of the V polynomial.
            cV (float): Quadratic coefficient of the V polynomial.
            dV (float): Cubic coefficient of the V polynomial.
            p_range_type (str): Parameter range type as specified in OpenDRIVE.
                "normalized" maps p in [0, 1]; "arcLength" maps p in [0, length].
                Defaults to "normalized".

        Returns:
            np.ndarray: World-coordinate points on the curve. Shape is (n, 2).
        """
        p_max = length if p_range_type == "arcLength" else 1.0
        n_interpolate = max(2, int(length / 0.1))
        p = np.linspace(0, p_max, n_interpolate)

        u = aU + bU * p + cU * p**2 + dU * p**3
        v = aV + bV * p + cV * p**2 + dV * p**3
        coords_uv = np.array([u, v]).T

        transform = np.array([
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading),  np.cos(heading)],
        ])

        coords_xy = np.dot(coords_uv, transform.T) + np.array(start_point)
        return coords_xy