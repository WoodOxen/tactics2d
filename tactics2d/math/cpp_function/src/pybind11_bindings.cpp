// pybind11_bindings.cpp
#include "circle.h"
#include "bezier.h"
#include "b_spline.h"
#include "cubic_spline.h"

PYBIND11_MODULE(cpp_function, m) {
    py::class_<Circle> Circle_class(m, "Circle");

    Circle_class.def_static("get_circle_by_three_points", &Circle::get_circle_by_three_points);
    Circle_class.def_static("get_circle_by_tangent_vector", &Circle::get_circle_by_tangent_vector);
    Circle_class.def_static("get_circle", &Circle::get_circle);
    Circle_class.def_static("get_arc", &Circle::get_arc);


    py::enum_<Circle::ConstructBy>(Circle_class, "ConstructBy")
        .value("ThreePoints", Circle::ConstructBy::ThreePoints)
        .value("TangentVector", Circle::ConstructBy::TangentVector)
        .export_values();

    py::class_<Bezier> Bezier_class(m, "Bezier");

    Bezier_class.def_static("get_curve", &Bezier::get_curve);


    py::class_<BSpline> BSpline_class(m, "BSpline");

    BSpline_class.def_static("get_curve", &BSpline::get_curve);


    py::class_<CubicSpline>CubicSpline_class(m, "CubicSpline");
        py::enum_<CubicSpline::BoundaryType>(CubicSpline_class, "BoundaryType")
        .value("Natural", CubicSpline::BoundaryType::Natural)
        .value("Clamped", CubicSpline::BoundaryType::Clamped)
        .value("NotAKnot", CubicSpline::BoundaryType::NotAKnot);
    CubicSpline_class.def(py::init<CubicSpline::BoundaryType>(), py::arg("boundary_type") = CubicSpline::BoundaryType::NotAKnot);

    CubicSpline_class.def("get_curve", &CubicSpline::get_curve);



}
