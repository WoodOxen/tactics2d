# Copyright (C) 2022, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Setup implementation."""

import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

include_dirs = [
    "tactics2d/interpolator/cpp_interpolator/include",
    "tactics2d/geometry/cpp_geometry/include",
]

if os.name == "nt":  # Windows
    extra_compile_args = ["/O2", "/D_USE_MATH_DEFINES"]
else:  # Linux, MacOS
    extra_compile_args = ["-O3"]

ext_modules = [
    Pybind11Extension(
        "cpp_interpolator",
        [
            "tactics2d/interpolator/cpp_interpolator/src/b_spline.cpp",
            "tactics2d/interpolator/cpp_interpolator/src/bezier.cpp",
            "tactics2d/interpolator/cpp_interpolator/src/cubic_spline.cpp",
            "tactics2d/interpolator/cpp_interpolator/src/interpolator_bindings.cpp",
        ],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Pybind11Extension(
        "cpp_geometry",
        [
            "tactics2d/geometry/cpp_geometry/src/circle.cpp",
            "tactics2d/geometry/cpp_geometry/src/geometry_bindings.cpp",
        ],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
