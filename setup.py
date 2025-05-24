import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

include_dirs = [
    "tactics2d/math/cpp_function/include",
]

if os.name == "nt":  # Windows
    extra_compile_args = ["/O2", "/D_USE_MATH_DEFINES"]
else:  # Linux, MacOS
    extra_compile_args = ["-O3"]

ext_modules = [
    Pybind11Extension(
        "cpp_function",
        [
            "tactics2d/math/cpp_function/src/b_spline.cpp",
            "tactics2d/math/cpp_function/src/cubic_spline.cpp",
            "tactics2d/math/cpp_function/src/bezier.cpp",
            "tactics2d/math/cpp_function/src/circle.cpp",
            "tactics2d/math/cpp_function/src/pybind11_bindings.cpp",
        ],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
