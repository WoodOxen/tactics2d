# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Width-profile helpers for map generators."""

from __future__ import annotations

import numpy as np


def hermite_cubic_width(s_local: np.ndarray, length: float, target_width: float) -> np.ndarray:
    """Return a zero-slope cubic width transition.

    The profile is a smoothstep Hermite cubic:

        w(t) = W * (3t^2 - 2t^3),  t = s / L

    It satisfies:

        w(0) = 0
        w(L) = W
        w'(0) = 0
        w'(L) = 0

    Args:
        s_local: Local arc-length coordinates.
        length: Transition length. Must be positive.
        target_width: Target width or width delta.

    Returns:
        Width values with the same shape as ``s_local``.
    """
    length = float(length)
    target_width = float(target_width)

    if length <= 0.0:
        raise ValueError("length must be positive.")

    s_local = np.asarray(s_local, dtype=float)
    t = np.clip(s_local / length, 0.0, 1.0)

    return target_width * (3.0 * t**2 - 2.0 * t**3)
