import logging
import sys
import time

import numpy as np
import pytest
from scipy.interpolate import BSpline as SciBSpline
from scipy.spatial.distance import directed_hausdorff

sys.path.append(".")
sys.path.append("..")

from tactics2d.interpolator import BSpline


def compare_similarity(curve1: np.ndarray, curve2: np.ndarray, diff: float = 0.001) -> bool:
    len1 = np.linalg.norm(curve1[1:] - curve1[:-1], axis=1).sum()
    len2 = np.linalg.norm(curve2[1:] - curve2[:-1], axis=1).sum()
    len_diff = abs(len1 - len2)
    ratio_len_diff = len_diff / max(min(len1, len2), 1e-8)

    hausdorff_dist = max(
        directed_hausdorff(curve1, curve2)[0], directed_hausdorff(curve2, curve1)[0]
    )
    ratio_shape_diff = hausdorff_dist / max(min(len1, len2), 1e-8)

    if ratio_len_diff >= diff or ratio_shape_diff >= diff:
        logging.warning(f"Hausdorff dist: {hausdorff_dist}, Length diff: {len_diff}")
    return ratio_len_diff < diff and ratio_shape_diff < diff


@pytest.mark.math
@pytest.mark.parametrize(
    "degree, control_points, knots, n_interpolation",
    [
        (0, None, None, 100),
        (1, None, None, 1000),
        (
            2,
            np.array([[-1, 0], [-0.5, 0.5], [0.5, -0.5], [1, 0]]),
            np.array([0, 0, 0, 1, 2, 2, 2]),
            100,
        ),
        (
            2,
            np.array(
                [
                    [-1, 0],
                    [-0.5, 0.5],
                    [0.5, -0.5],
                    [1, 0],
                    [-1, 0],
                    [-0.5, 0.5],
                    [0.5, -0.5],
                ]
            ),
            np.array(np.arange(0, 10)),
            100,
        ),
        (2, np.random.uniform(-10, 10, (3, 3)), None, 100),
        (2, np.random.uniform(-10, 10, (3, 2)), np.array(np.arange(0, 4)), 100),
        (2, np.random.uniform(-10, 10, (3, 2)), np.array(np.arange(0, 10)), 100),
        (3, None, None, 1000),
        (4, None, None, 1000),
    ],
)
def test_b_spline(degree, control_points, knots, n_interpolation):
    if control_points is None:
        n_control_point = np.random.randint(5, 100)
        control_points = np.zeros((n_control_point, 2))
        for i in range(1, n_control_point):
            control_points[i, 0] = control_points[i - 1, 0] + np.random.uniform(0.1, 1)
            control_points[i, 1] = np.random.uniform(-1, 1)

    if knots is None and np.random.rand() < 0.3:
        knots = np.zeros((len(control_points) + degree + 1,))
        for i in range(1, len(knots) - 1):
            knots[i] = knots[i - 1] + np.random.randint(0, 3)

    # Construct B-spline instance
    t1 = time.time()
    try:
        my_bspline = BSpline(degree)
    except ValueError as err:
        if degree < 1:
            assert (
                str(err) == "BSpline interpolator: Degree must be non-negative."
            ), "Incorrect error message for invalid degree"
        return

    # Call our BSpline implementation
    try:
        my_curve = my_bspline.get_curve(control_points, knots, n_interpolation=n_interpolation)
    except ValueError as err:
        msg = str(err)
        if control_points.shape[-1] != 2:
            assert (
                msg == "BSpline interpolator: Control points should have shape (n, 2)."
            ), "Unexpected error message for invalid control points shape"
        elif knots is not None and len(knots) != len(control_points) + degree + 1:
            expected = len(control_points) + degree + 1
            assert (
                msg == f"BSpline interpolator: Expected {expected} knots, got {len(knots)}."
            ), "Unexpected error message for invalid knot count"
        elif knots is not None and np.any(np.diff(knots) < 0):
            assert (
                msg == "BSpline interpolator: Knot vector must be non-decreasing."
            ), "Unexpected error for non-monotonic knots"
        else:
            raise
        return

    t2 = time.time()

    # Create scipy BSpline for ground truth
    if knots is None:
        knots = np.linspace(0, 1, len(control_points) + degree + 1)

    t3 = time.time()
    scipy_bspline = SciBSpline(knots, control_points, degree)
    us = np.linspace(knots[degree], knots[-degree - 1], n_interpolation, endpoint=False)
    scipy_curve = np.array([scipy_bspline(u) for u in us])
    t4 = time.time()

    assert compare_similarity(
        my_curve, scipy_curve
    ), "The curve output differs from SciPy's BSpline"

    t_custom = t2 - t1
    t_scipy = t4 - t3
    if t_custom > t_scipy:
        logging.warning(f"Our B-spline is ~{t_custom / (t_scipy + 1e-8):.2f}x slower than SciPy's")
