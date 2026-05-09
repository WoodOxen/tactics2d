#! python3
# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# @File: param_poly3.py
# @Description: This file implements a parametric cubic polynomial curve interpolation.
# @Author: Zexi Chen
# @Version: 1.0.0

from __future__ import annotations

import numpy as np


class ParamPoly3:
    """This class implements a parametric cubic polynomial curve interpolation."""

    @staticmethod
    def get_curve(
        length: float,
        start_point: tuple,
        heading: float,
        aU: float,
        bU: float,
        cU: float,
        dU: float,
        aV: float,
        bV: float,
        cV: float,
        dV: float,
        p_range_type: str = "normalized",
    ) -> np.ndarray:
        """Get the points on a paramPoly3 curve defined in the OpenDRIVE standard.

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

        transform = np.array(
            [[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]]
        )

        coords_xy = np.dot(coords_uv, transform.T) + np.array(start_point)
        return coords_xy

    @staticmethod
    def fit(pts: np.ndarray) -> tuple | None:
        """Fit a paramPoly3 curve to a polyline segment.

        Transforms the segment to a local frame (origin at start, x-axis along
        initial heading), then fits cubic polynomials U(p) and V(p) where p is
        the normalised arc-length parameter in [0, 1].

        Args:
            pts (np.ndarray): Polyline points in world coordinates. Shape (N, 2).

        Returns:
            tuple | None: A tuple of
                (x, y, hdg, length, aU, bU, cU, dU, aV, bV, cV, dV).
                Returns None if the segment is degenerate.

        Example:
            >>> import numpy as np
            >>> from tactics2d.interpolator.param_poly3 import ParamPoly3
            >>> pts = np.array([[0., 0.], [10., 1.], [20., 0.]])
            >>> result = ParamPoly3.fit(pts)
            >>> x, y, hdg, length, aU, bU, cU, dU, aV, bV, cV, dV = result
            >>> length > 0
            True
        """
        if len(pts) < 2:
            return None

        x0, y0 = float(pts[0, 0]), float(pts[0, 1])
        dx = float(pts[-1, 0] - pts[0, 0])
        dy = float(pts[-1, 1] - pts[0, 1])
        hdg = float(np.arctan2(dy, dx)) if (abs(dx) + abs(dy)) > 1e-9 else 0.0

        cos_h, sin_h = np.cos(hdg), np.sin(hdg)
        diffs = np.diff(pts, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        total = float(seg_lengths.sum())
        if total < 1e-6:
            return None

        s_cum = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        p = s_cum / total

        local = pts - pts[0]
        u = local[:, 0] * cos_h + local[:, 1] * sin_h
        v = -local[:, 0] * sin_h + local[:, 1] * cos_h

        coeffs_u = np.polyfit(p, u, 3)[::-1]
        coeffs_v = np.polyfit(p, v, 3)[::-1]

        return (
            x0,
            y0,
            hdg,
            total,
            float(coeffs_u[0]),
            float(coeffs_u[1]),
            float(coeffs_u[2]),
            float(coeffs_u[3]),
            float(coeffs_v[0]),
            float(coeffs_v[1]),
            float(coeffs_v[2]),
            float(coeffs_v[3]),
        )


def split_polyline(pts: list | np.ndarray, max_seg_length: float = 20.0) -> list[np.ndarray]:
    """Split a polyline into segments no longer than *max_seg_length*.

    Args:
        pts (list | np.ndarray): Polyline points. Shape (N, 2).
        max_seg_length (float): Maximum segment length in metres. Defaults to 20.0.

    Returns:
        list[np.ndarray]: List of segment arrays, each shape (M, 2).

    Example:
        >>> import numpy as np
        >>> from tactics2d.interpolator.param_poly3 import split_polyline
        >>> pts = [(0, 0), (30, 0), (60, 0)]
        >>> segs = split_polyline(pts, max_seg_length=25)
        >>> len(segs)
        3
    """
    arr = np.asarray(pts, dtype=float)
    if len(arr) < 2:
        return [arr]

    diffs = np.diff(arr, axis=0)
    lens = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(lens)])
    total = cum[-1]

    if total <= max_seg_length:
        return [arr]

    n_segs = max(1, int(np.ceil(total / max_seg_length)))
    breaks = np.linspace(0, total, n_segs + 1)

    segments = []
    for i in range(n_segs):
        s0, s1 = breaks[i], breaks[i + 1]
        mask = (cum >= s0 - 1e-9) & (cum <= s1 + 1e-9)
        seg = arr[mask]
        if len(seg) < 2:
            idx0 = np.searchsorted(cum, s0)
            idx1 = np.searchsorted(cum, s1)
            seg = arr[max(0, idx0) : min(len(arr), idx1 + 1)]
        if len(seg) >= 2:
            segments.append(seg)

    return segments if segments else [arr]
