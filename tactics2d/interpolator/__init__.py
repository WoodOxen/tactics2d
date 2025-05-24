##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: __init__.py
# @Description: Initialize the interpolator module.
# @Author: Tactics2D Team
# @Version:0.1.9

from .b_spline import BSpline
from .bezier import Bezier

__all__ = ["Bezier", "BSpline"]
