#include <pybind11/pybind11.h>

#include "b_spline.hpp"
#include "bezier.hpp"
#include "cubic_spline.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cpp_interpolator, m) {
    py::class_<Bezier>(m, "Bezier").def_static("get_curve", &Bezier::get_curve);

    py::class_<BSpline>(m, "BSpline").def_static("get_curve", &BSpline::get_curve);

    py::class_<CubicSpline> CubicSpline_class(m, "CubicSpline");
    py::enum_<CubicSpline::BoundaryType>(CubicSpline_class, "BoundaryType")
        .value("Natural", CubicSpline::BoundaryType::Natural)
        .value("Clamped", CubicSpline::BoundaryType::Clamped)
        .value("NotAKnot", CubicSpline::BoundaryType::NotAKnot);

    CubicSpline_class.def_static("get_curve", &CubicSpline::get_curve);
}
