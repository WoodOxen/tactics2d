// pybind11_bindings.cpp
#include <pybind11/pybind11.h>

#include "circle.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cpp_geometry, m) {
    py::class_<Circle>(m, "Circle")
        .def_static("get_circle_by_three_points", &Circle::get_circle_by_three_points)
        .def_static("get_circle_by_tangent_vector",
                    &Circle::get_circle_by_tangent_vector);
}
