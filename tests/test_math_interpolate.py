##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: test_math_interpolate.py
# @Description: This script is used to test the interpolators in the math module.
# @Author: Yueyuan Li
# @Version: 1.0.0


import sys

sys.path.append(".")
sys.path.append("..")

import logging
import time

# import dubins
import numpy as np
import pytest
from scipy.interpolate import BSpline as SciBSpline
from scipy.interpolate import CubicSpline as SciCubic
from scipy.spatial.distance import directed_hausdorff

from tactics2d.math.interpolate import *


def compare_similarity(curve1: np.ndarray, curve2: np.ndarray, diff: float = 0.001) -> bool:
    # compute difference in length
    len1 = np.linalg.norm(curve1[1:] - curve1[:-1], axis=1).sum()
    len2 = np.linalg.norm(curve2[1:] - curve2[:-1], axis=1).sum()
    len_diff = abs(len1 - len2)
    ratio_len_diff = len_diff / min(len1, len2)

    # compute difference in shape
    hausdorff_dist = max(
        directed_hausdorff(curve1, curve2)[0], directed_hausdorff(curve2, curve1)[0]
    )
    ratio_shape_diff = hausdorff_dist / min(len1, len2)

    return ratio_len_diff < diff and ratio_shape_diff < diff


@pytest.mark.math
@pytest.mark.parametrize(
    "order, control_points, n_interpolation",
    [
        (1, None, 100),
        (2, None, 1000),
        (3, np.random.uniform(1, 5, (5, 2)), 100),
        (3, np.random.uniform(1, 5, (4, 3)), 100),
        (4, None, 1000),
        (4, np.random.uniform(1, 5, (4, 2)), 100),
        (5, None, 1000),
    ],
)
def test_bezier(order: int, control_points: np.ndarray, n_interpolation: int):
    if control_points is None:
        control_points = np.zeros((order + 1, 2))
        for i in range(1, order + 1):
            control_points[i, 0] = control_points[i - 1, 0] + np.random.uniform(0, 1)
            control_points[i, 1] = np.random.uniform(-1, 1)

    t1 = time.time()
    try:
        my_bezier = Bezier(order)
    except ValueError as err:
        if order < 1:
            assert (
                err.args[0] == "The order of a Bezier curve must be greater than or equal to one."
            ), "Test failed: error handling for invalid order."
        return

    try:
        my_curve = my_bezier.get_curve(control_points, n_interpolation)
    except ValueError as err:
        if len(control_points.shape) != 2 or control_points.shape[1] != 2:
            assert (
                err.args[0] == "The shape of control_points is expected to be (n, 2)."
            ), "Test failed: error handling for invalid shape of control points."
        elif len(control_points) != order + 1:
            assert (
                err.args[0]
                == "The number of control points must be equal to the order of the Bezier curve plus one."
            ), "Test failed: error handling for invalid number of control points."
        else:
            raise err
        return

    t2 = time.time()

    try:
        import bezier
    except:
        return

    curve = bezier.Curve(control_points.T, degree=order).evaluate_multi(
        np.linspace(0, 1, n_interpolation)
    )
    t3 = time.time()

    assert compare_similarity(
        my_curve, curve.T
    ), "The curve output of the Bezier interpolator is incorrect."

    if t2 - t1 > t3 - t2:
        logging.warning(
            "The implemented Bezier interpolator is %.2f times slower than the Python library bezier. The efficiency needs further improvement."
            % ((t2 - t1) / (t3 + 1e-6 - t2) * 100)
        )


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
                [[-1, 0], [-0.5, 0.5], [0.5, -0.5], [1, 0], [-1, 0], [-0.5, 0.5], [0.5, -0.5]]
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
def test_b_spline(degree: int, control_points: np.ndarray, knots: np.ndarray, n_interpolation: int):
    if control_points is None:

        n_control_point = np.random.randint(5, 100)
        control_points = np.zeros((n_control_point, 2))
        for i in range(1, n_control_point):
            control_points[i, 0] = control_points[i - 1, 0] + np.random.uniform(0, 1)
            control_points[i, 1] = np.random.uniform(-1, 1)

    if knots is None and np.random.rand() < 0.3:
        knots = np.zeros((len(control_points) + degree + 1,))
        for i in range(1, len(knots) - 1):
            knots[i] = knots[i - 1] + np.random.randint(0, 3)

    t1 = time.time()
    try:
        my_bspline = BSpline(degree)
    except ValueError as err:
        if degree < 1:
            assert (
                err.args[0] == "The degree of a B-spline curve must be non-negative."
            ), "Test failed: error handling for invalid degree."
        return

    try:
        my_curve = my_bspline.get_curve(control_points, knots, n_interpolation=n_interpolation)
    except ValueError as err:
        if len(control_points.shape) != 2 or control_points.shape[1] != 2:
            assert (
                err.args[0] == "The shape of control_points is expected to be (n, 2)."
            ), "Test failed: error handling for invalid shape of control points."
        elif len(knots.shape) != 1:
            assert (
                err.args[0] == "The shape of knots is expected to be (t, )."
            ), "Test failed: error handling for invalid shape of knots."
        elif len(knots) != len(control_points) + degree + 1:
            assert (
                err.args[0]
                == "The number of knots must be equal to the number of control points plus the degree of the B-spline curve plus one."
            ), "Test failed: error handling for invalid number of knots."
        elif np.any((knots[1:] - knots[:-1]) < 0):
            assert (
                err.args[0] == "The knot vectors must be non-decreasing."
            ), "Test failed: error handling for invalid shape of control points."
        else:
            raise err
        return

    t2 = time.time()
    if knots is None:
        knots = np.linspace(0, 1, len(control_points) + degree + 1)

    if len(control_points) < degree + 1:
        assert len(knots) == 2 * degree + 2, "Test failed: error handling for insufficient number of knots."

    bspline = SciBSpline(knots, control_points, degree)
    us = np.linspace(knots[degree], knots[-degree - 1], n_interpolation, endpoint=False)
    curve = np.array([list(bspline(u)) for u in us])
    t3 = time.time()

    assert compare_similarity(
        my_curve, curve
    ), "The curve output of the B-Spline interpolator is incorrect."

    if t2 - t1 > t3 - t2:
        logging.warning(
            "The implemented B-Spline interpolator is %.2f times slower than Scipy's implementation. The efficiency needs further improvement."
            % ((t2 - t1) / (t3 + 1e-6 - t2) * 100)
        )


@pytest.mark.math
@pytest.mark.parametrize(
    "boundary_type, n, control_points, n_interpolation",
    [
        ("hello-world", 2, None, 100),
        ("natural", 2, None, 100),
        ("clamped", None, np.random.uniform(-10, 10, (3, 3)), 100),
        ("not-a-knot", None, np.random.uniform(-10, 10, (3, 1)), 100),
        ("natural", None, None, 10),
        ("natural", None, None, 100),
        ("clamped", None, None, 10),
        ("clamped", None, None, 100),
        ("not-a-knot", None, None, 10),
        ("not-a-knot", None, None, 100),
    ],
)
def test_cubic_spline(boundary_type: str, n: int, control_points: np.ndarray, n_interpolation: int):
    if boundary_type == "natural":
        cubic_spline = CubicSpline(CubicSpline.BoundaryType.Natural)
    elif boundary_type == "clamped":
        cubic_spline = CubicSpline(CubicSpline.BoundaryType.Clamped)
    elif boundary_type == "not-a-knot":
        cubic_spline = CubicSpline(CubicSpline.BoundaryType.NotAKnot)
    else:
        try:
            cubic_spline = CubicSpline(boundary_type)
        except ValueError:
            return

    if control_points is None:
        n = np.random.randint(3, 1000) if n is None else n
        control_points = np.zeros((n, 2))
        for i in range(1, n):
            control_points[i, 0] = control_points[i - 1, 0] + np.random.uniform(0, 1)
            control_points[i, 1] = np.random.uniform(-1, 1)

    t1 = time.time()
    try:
        curve = cubic_spline.get_curve(control_points, n_interpolation=n_interpolation)
    except ValueError as err:
        if len(control_points.shape) != 2 or control_points.shape[1] != 2:
            assert (
                err.args[0] == "The shape of control_points is expected to be (n, 2)."
            ), "Test failed: error handling for invalid shape of control points."
        elif len(control_points) < 3:
            assert (
                err.args[0]
                == "There is not enough control points to interpolate a cubic spline curve."
            ), "Test failed: error handling for insufficient number of control points."
        else:
            raise err
        return

    t2 = time.time()

    sci_cubic = SciCubic(control_points[:, 0], control_points[:, 1], bc_type=boundary_type)
    t3 = time.time()
    xs = np.linspace(control_points[:-1, 0], control_points[1:, 0], n_interpolation)
    xs = np.concatenate(xs.T, axis=None)
    ys = sci_cubic(xs)
    t4 = time.time()

    assert compare_similarity(
        curve, np.array([xs, ys]).T
    ), "The curve output of the Cubic Spline interpolator is incorrect."

    if t2 - t1 > t4 - t3:
        logging.warning(
            "The implemented Cubic Spline interpolator is %.2f times slower than Scipy's implementation. The efficiency needs further improvement."
            % ((t2 - t1) / (t4 + 1e-6 - t3) * 100)
        )


@pytest.mark.math
@pytest.mark.parametrize(
    "radius, start_point, start_heading, end_point, end_heading, step_size",
    [
        (7.5, np.array([10, 10]), 1, np.array([-20, -10]), 2, 0.01),
        (7.5, np.array([10, 10]), 1, np.array([-20, -10]), -1, 0.01),
        (7.5, np.array([10, 10]), -1, np.array([-20, -10]), 2, 0.01),
        (7.5, np.array([10, 10]), -1, np.array([-20, -10]), -1, 0.01),
        (7.5, np.array([10, 10]), 4, np.array([15, 5]), 2, 0.01),
        (7.5, np.array([10, 10]), 0.5, np.array([5, 5]), 2, 0.01),
        (-7.5, np.array([10, 10]), 4, np.array([15, 5]), 2, 0.01),
    ],
)
def test_dubins(radius, start_point, start_heading, end_point, end_heading, step_size):
    # t1 = time.time()
    try:
        dubins = Dubins(radius)
    except ValueError as err:
        if radius <= 0:
            assert (
                err.args[0] == "The minimum turning radius must be positive."
            ), "Test failed: error handling for invalid radius."
        else:
            raise err
        return

    path = dubins.get_curve(start_point, start_heading, end_point, end_heading, step_size)
    curve = path.curve

    curve_length = np.linalg.norm(curve[1:] - curve[:-1], axis=1).sum()
    assert abs(path.length - curve_length) / min(path.length, curve_length) < 0.01

    # t2 = time.time()
    # path = dubins.shortest_path(
    #     (start_point[0], start_point[1], start_heading),
    #     (end_point[0], end_point[1], end_heading),
    #     radius,
    # ).sample_many(step_size)
    # t3 = time.time()
    # curve = []
    # for point in path[0]:
    #     curve.append([point[0], point[1]])
    # curve = np.array(curve)

    # assert compare_similarity(
    #     my_curve, curve, 0.1
    # ), "The curve output of the Dubins interpolator is incorrect."

    # if t2 - t1 > t3 - t2:
    #     logging.warning(
    #         "The implemented Dubins interpolator is %.2f times slower than the Python library dubins. The efficiency needs further improvement."
    #         % ((t2 - t1) / (t3 - t2) * 100)
    #     )


@pytest.mark.math
@pytest.mark.parametrize(
    "radius, start_point, start_heading, end_point, end_heading, step_size",
    [
        (7.5, np.array([10, 10]), 1, np.array([-20, -10]), 2, 0.01),
        (7.5, np.array([10, 10]), 1, np.array([-20, -10]), -1, 0.01),
        (7.5, np.array([10, 10]), -1, np.array([-20, -10]), 2, 0.01),
        (7.5, np.array([10, 10]), -1, np.array([-20, -10]), -1, 0.01),
        (7.5, np.array([10, 10]), 4, np.array([15, 5]), 2, 0.01),
        (7.5, np.array([10, 10]), 0.5, np.array([5, 5]), 2, 0.01),
        (-7.5, np.array([10, 10]), 4, np.array([15, 5]), 2, 0.01),
    ],
)
def test_reeds_shepp(radius, start_point, start_heading, end_point, end_heading, step_size):
    try:
        rs = ReedsShepp(radius)
    except ValueError as err:
        if radius <= 0:
            assert (
                err.args[0] == "The minimum turning radius must be positive."
            ), "Test failed: error handling for invalid radius."
        else:
            raise err
        return

    path = rs.get_curve(start_point, start_heading, end_point, end_heading, step_size)
    curve = path.curve

    curve_length = np.linalg.norm(curve[1:] - curve[:-1], axis=1).sum()
    assert abs(path.length - curve_length) / min(path.length, curve_length) < 0.01


@pytest.mark.math
@pytest.mark.parametrize("length, n_interpolation", [(5, 500), (10, 1000), (105, 10000)])
def test_spiral(length, n_interpolation):
    spiral = Spiral()
    start_point = np.array([0, 0])
    heading = np.random.uniform(0, 2 * np.pi)
    start_curvature = 0.1
    gamma = 0.1
    curve = spiral.get_spiral(length, start_point, heading, start_curvature, gamma)

    curve_length = np.linalg.norm(curve[1:] - curve[:-1], axis=1).sum()
    assert abs(length - curve_length) / min(length, curve_length) < 0.01
    assert len(curve) == n_interpolation
