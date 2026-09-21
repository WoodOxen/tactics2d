# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the spatial geometry utilities."""

import numpy as np
import pytest

from tactics2d.geometry.spatial import (
    as_point,
    central_diff,
    central_logical_and,
    oriented_box_corners,
    rounded_box_distance,
)

PARKED = (0.0, 0.0, 0.0, 4.0, 2.0)


@pytest.mark.math
@pytest.mark.parametrize("bad", [[1.0], [1.0, 2.0, 3.0], 1.0])
def test_as_point_rejects_input_that_is_not_a_pair(bad):
    """Inputs that are not two-element vectors are rejected by name."""
    with pytest.raises(ValueError, match="origin must have shape"):
        as_point(bad, name="origin")


@pytest.mark.math
def test_central_diff_pads_the_ends_and_uses_neighbours():
    """The central difference keeps the input shape and pads both ends."""
    differences = central_diff(np.array([1.0, 2.0, 4.0, 7.0, 11.0]))

    assert differences.shape == (5,)
    assert np.isnan(differences[0]) and np.isnan(differences[-1])
    np.testing.assert_allclose(differences[1:-1], [1.5, 2.5, 3.5])

    np.testing.assert_allclose(
        central_diff(np.array([[0.0, 1.0, 3.0]]), pad_value=0.0), [[0.0, 1.5, 0.0]]
    )


@pytest.mark.math
def test_central_logical_and_requires_both_neighbours():
    """A flag survives only when the samples on either side of it are set."""
    combined = central_logical_and(np.array([True, True, False, True, True]))

    np.testing.assert_array_equal(combined, [False, False, True, False, False])
    assert combined.dtype == np.bool_


@pytest.mark.math
def test_oriented_box_corners_start_front_right_and_go_counter_clockwise():
    """Corners run counter-clockwise from the front-right one, one set per pose."""
    corners = oriented_box_corners(0.0, 0.0, 0.0, 4.0, 2.0)

    assert corners.shape == (4, 2)
    np.testing.assert_allclose(corners, [[2.0, -1.0], [2.0, 1.0], [-2.0, 1.0], [-2.0, -1.0]])

    signed_area = 0.5 * np.sum(
        corners[:, 0] * np.roll(corners[:, 1], -1) - np.roll(corners[:, 0], -1) * corners[:, 1]
    )
    assert signed_area > 0.0

    broadcast = oriented_box_corners(np.array([0.0, 10.0]), np.array([0.0, 0.0]), 0.0, 2.0, 1.0)
    assert broadcast.shape == (2, 4, 2)
    np.testing.assert_allclose(broadcast[1], broadcast[0] + [10.0, 0.0])


@pytest.mark.math
def test_rounded_box_distance_signs_by_separation_and_broadcasts_over_pairs():
    """The gap is the free space along the centre line, negative once the boxes overlap."""
    ahead = (10.0, 0.0, 0.0, 4.0, 2.0)
    intruding = (1.0, 0.0, 0.0, 4.0, 2.0)

    assert float(rounded_box_distance(PARKED, ahead)) == pytest.approx(6.0)
    # 3 m of overlap along x, 2 m along y: 2 m of travel separates them.
    assert float(rounded_box_distance(PARKED, intruding, rounding=0.0)) == pytest.approx(-2.0)

    distances = rounded_box_distance(np.array([PARKED, PARKED]), np.array([ahead, intruding]))
    assert distances.shape == (2,)
    np.testing.assert_allclose(distances, [6.0, -2.0])
