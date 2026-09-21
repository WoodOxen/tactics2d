# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the polyline geometry utilities."""

import numpy as np
import pytest
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
)

from tactics2d.geometry.polyline import (
    _intersection_points,
    concatenate,
    curvature_stats,
    cut,
    end_pose,
    find_intersection_point,
    has_self_intersection,
    heading_at_arc_length,
    offset,
    point_at_arc_length,
    project_arc_length,
    resample,
    signed_distance_to_polyline,
    smooth_joint,
)

STRAIGHT = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
ELBOW = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])


@pytest.mark.math
@pytest.mark.parametrize("bad", [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.0, 0.0]]])
def test_offset_rejects_malformed_polylines(bad):
    """Inputs that are not (N, 2) with N >= 2 are rejected."""
    with pytest.raises(ValueError):
        offset(np.asarray(bad), 1.0)


@pytest.mark.math
def test_arc_length_queries_clamp_and_follow_the_segment():
    """Projection and sampling report the arc-length coordinate, clamped to the extent."""
    assert project_arc_length(STRAIGHT, np.array([4.0, 3.0])) == pytest.approx(4.0)
    assert project_arc_length(STRAIGHT, np.array([-5.0, 0.0])) == pytest.approx(0.0)
    assert project_arc_length(ELBOW, np.array([12.0, 6.0])) == pytest.approx(16.0)

    np.testing.assert_allclose(point_at_arc_length(STRAIGHT, 5.0), [5.0, 0.0])
    np.testing.assert_allclose(point_at_arc_length(STRAIGHT, 99.0), [10.0, 0.0])
    assert heading_at_arc_length(ELBOW, 4.0) == pytest.approx(0.0)
    assert heading_at_arc_length(ELBOW, 15.0) == pytest.approx(np.pi / 2)

    with pytest.raises(ValueError, match="point must have shape"):
        project_arc_length(STRAIGHT, np.array([1.0, 2.0, 3.0]))


@pytest.mark.math
def test_end_pose_returns_the_last_point_and_heading():
    """The end pose is the final vertex plus its incoming tangent."""
    point, heading = end_pose(ELBOW)

    np.testing.assert_allclose(point, [10.0, 10.0])
    assert heading == pytest.approx(np.pi / 2)


@pytest.mark.math
def test_curvature_stats_report_zero_and_a_constant_radius_arc():
    """Straight runs carry no curvature; a quarter circle of radius 10 reports 1/10."""
    straight = curvature_stats(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]))
    assert straight["max_abs_curvature"] == 0.0

    angles = np.linspace(0.0, np.pi / 2, 20)
    arc = np.column_stack([10.0 * np.sin(angles), 10.0 * (1.0 - np.cos(angles))])
    stats = curvature_stats(arc)
    assert stats["max_abs_curvature"] == pytest.approx(1.0 / 10.0, rel=0.02)
    assert stats["max_abs_curvature_rate"] < 1e-2


@pytest.mark.math
def test_has_self_intersection_detects_a_bowtie():
    """Crossing segments report an intersection; a simple path does not."""
    bowtie = np.array([[0.0, 0.0], [2.0, 2.0], [2.0, 0.0], [0.0, 2.0]])

    assert has_self_intersection(bowtie) is True
    assert has_self_intersection(ELBOW) is False


@pytest.mark.math
def test_resample_spaces_the_requested_number_of_points():
    """Resampling returns exactly n points, evenly spaced by arc length."""
    points = resample(STRAIGHT, 5)

    assert points.shape == (5, 2)
    np.testing.assert_allclose(points[:, 0], np.linspace(0.0, 10.0, 5))
    # An empty polyline and n <= 1 keep their degenerate shape.
    assert resample(np.empty((0, 2)), 4).shape == (0, 2)
    assert resample(STRAIGHT, 1).shape == (1, 2)


@pytest.mark.math
def test_find_intersection_point_picks_by_rule():
    """Overlaps expose two candidate points, chosen by the pick rule; a miss is None."""
    long_line = LineString([(0.0, 0.0), (10.0, 0.0)])
    overlap = LineString([(2.0, 0.0), (6.0, 0.0)])

    assert find_intersection_point(long_line, overlap).x == pytest.approx(2.0)
    assert find_intersection_point(long_line, overlap, pick="last_on_line1").x == pytest.approx(6.0)
    # The same two points, measured along the other line.
    assert find_intersection_point(overlap, long_line, pick="first_on_line2").x == pytest.approx(
        2.0
    )
    assert find_intersection_point(overlap, long_line, pick="last_on_line2").x == pytest.approx(6.0)
    # Crossing lines share one point whatever the rule; parallel ones share none.
    vertical = LineString([(5.0, -5.0), (5.0, 5.0)])
    assert find_intersection_point(long_line, vertical, pick="leftmost").x == pytest.approx(5.0)
    assert find_intersection_point(long_line, LineString([(0.0, 1.0), (10.0, 1.0)])) is None


@pytest.mark.math
def test_intersection_points_cover_the_shapely_geometry_types():
    """Points, lines and collections yield their corners; empty and polygon inputs do not."""
    multipoint = _intersection_points(MultiPoint([(0.0, 0.0), (1.0, 1.0)]))
    collection = _intersection_points(
        GeometryCollection([Point(0.0, 0.0), MultiLineString([[(0.0, 0.0), (1.0, 0.0)]])])
    )

    def coords(geometries):
        return [tuple(map(float, geometry.coords[0])) for geometry in geometries]

    assert coords(multipoint) == [(0.0, 0.0), (1.0, 1.0)]
    # A collection contributes the points of every member it holds.
    assert coords(collection) == [(0.0, 0.0), (0.0, 0.0), (1.0, 0.0)]
    assert _intersection_points(LineString()) == []
    assert _intersection_points(Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])) == []


@pytest.mark.math
def test_cut_keeps_the_requested_side_of_the_cut_point():
    """Cutting returns one side of the projected point, which is inserted in the result."""
    line = LineString(ELBOW)

    np.testing.assert_allclose(
        cut(line, Point(10.0, 2.5), keep="before"), [[0.0, 0.0], [10.0, 0.0], [10.0, 2.5]]
    )
    np.testing.assert_allclose(
        cut(line, Point(10.0, 2.5), keep="after"), [[10.0, 2.5], [10.0, 10.0]]
    )

    with pytest.raises(ValueError, match="keep must be"):
        cut(line, Point(1.0, 0.0), keep="middle")


@pytest.mark.math
def test_smooth_joint_rounds_a_right_angle_and_skips_the_rest():
    """A perpendicular joint gets an arc; separated, smooth and degenerate ones get None."""
    incoming = np.array([[0.0, -4.0], [0.0, -2.0], [0.0, 0.0]])

    arc = smooth_joint(incoming, np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]]))
    assert arc is not None
    assert np.all(np.linalg.norm(arc - np.array([0.0, 0.0]), axis=1) >= 0.01)

    assert smooth_joint(incoming, np.array([[1.0, 0.0], [3.0, 0.0]])) is None
    assert smooth_joint(incoming, np.array([[0.0, 0.0], [0.0, 2.0]])) is None


@pytest.mark.math
def test_concatenate_joins_touching_polylines():
    """Shared endpoints are merged, gapped polylines are stacked, a kink gains an arc."""
    merged = concatenate([np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[1.0, 0.0], [3.0, 0.0]])])
    np.testing.assert_allclose(merged, [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])

    gapped = concatenate([np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[5.0, 0.0], [6.0, 0.0]])])
    assert gapped.shape == (4, 2)

    kinked = concatenate([np.array([[0.0, -4.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [4.0, 0.0]])])
    assert kinked.shape[0] > 3
    np.testing.assert_allclose(kinked[-1], [4.0, 0.0])
    assert concatenate([]) is None


@pytest.mark.math
def test_signed_distance_to_polyline_signs_by_side():
    """Distances are negative on the left of the direction and positive on the right."""
    distances = signed_distance_to_polyline(
        np.array([[5.0, 2.0], [5.0, -2.0], [5.0, 0.0]]), STRAIGHT
    )
    np.testing.assert_allclose(distances, [-2.0, 2.0, 0.0], atol=1e-12)

    # Inside a counter-clockwise ring is the left side, so distances are negative.
    ring = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    np.testing.assert_allclose(
        signed_distance_to_polyline(np.array([[0.5, 0.5]]), ring, cyclic=True), [-0.5]
    )

    with pytest.raises(ValueError, match="zero-length segment"):
        signed_distance_to_polyline(np.array([[1.0, 1.0]]), np.array([[0.0, 0.0], [0.0, 0.0]]))
