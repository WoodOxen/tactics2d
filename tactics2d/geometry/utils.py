# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""General geometry utility functions."""

import numpy as np


def normalize_angle(angle: float) -> float:
    """Normalize an angle to the range [-pi, pi]."""

    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def euclidean_distance(a, b) -> float:
    """Compute the Euclidean distance between two 2D points."""

    return float(np.linalg.norm(np.asarray(a, dtype=float)[:2] - np.asarray(b, dtype=float)[:2]))
