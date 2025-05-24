// pybind11_bindings.cpp
#include <pybind11/pybind11.h>

#include "b_spline.h"
#include "bezier.h"

namespace py = pybind11;

PYBIND11_MODULE(cpp_interpolator, m) {
    py::class_<Bezier>(m, "Bezier").def_static("get_curve", &Bezier::get_curve);

    py::class_<BSpline>(m, "BSpline").def_static("get_curve", &BSpline::get_curve);
}
