// pybind11_bindings.cpp
#include <pybind11/pybind11.h>

#include "b_spline.h"

namespace py = pybind11;

PYBIND11_MODULE(cpp_interpolator, m) {
    py::class_<BSpline>(m, "BSpline").def_static("get_curve", &BSpline::get_curve);
}
