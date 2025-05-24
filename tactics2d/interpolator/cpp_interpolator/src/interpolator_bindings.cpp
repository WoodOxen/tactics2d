// pybind11_bindings.cpp
#include <pybind11/pybind11.h>

#include "b_spline.h"
#include "bezier.h"
#include "cubic_spline.h"

namespace py = pybind11;

PYBIND11_MODULE(cpp_interpolator, m) {
    py::class_<Bezier>(m, "Bezier").def("get_curve", &Bezier::get_curve);

    py::class_<BSpline>(m, "BSpline").def("get_curve", &BSpline::get_curve);

    py::class_<CubicSpline> CubicSpline_class(m, "CubicSpline");
    py::enum_<CubicSpline::BoundaryType>(CubicSpline_class, "BoundaryType")
        .value("Natural", CubicSpline::BoundaryType::Natural)
        .value("Clamped", CubicSpline::BoundaryType::Clamped)
        .value("NotAKnot", CubicSpline::BoundaryType::NotAKnot);
    CubicSpline_class.def(py::init<CubicSpline::BoundaryType>(),
                          py::arg("boundary_type") = CubicSpline::BoundaryType::NotAKnot);

    CubicSpline_class.def("get_curve", &CubicSpline::get_curve);
}
