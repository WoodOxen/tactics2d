// circle.h
#ifndef CIRCLE_H
#define CIRCLE_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <vector>
#include <array>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <algorithm>

namespace py = pybind11;


class Circle {    
    // This class implement some frequently operations on circle.
public:
    enum class ConstructBy {
        /*The method to derive a circle.

        Attributes:
            ThreePoints (int): Derive a circle by three points.
            TangentVector (int): Derive a circle by a tangent point, a heading and a radius.
        */

        ThreePoints = 1,
        TangentVector
    };

    static std::pair<std::array<double, 2>, double> get_circle_by_three_points(
        const std::array<double, 2>& pt1,
        const std::array<double, 2>& pt2,
        const std::array<double, 2>& pt3
    );

    static std::pair<std::array<double, 2>, double> get_circle_by_tangent_vector(
        const std::array<double, 2>& tangent_point,
        double heading,
        double radius,
        const std::string& side
    );

    static std::pair<std::array<double, 2>, double> get_circle(
        ConstructBy method, 
        py::args args
    );

    static std::vector<std::array<double, 2>> get_arc(
        const std::array<double, 2>& center_point,
        double radius,
        double delta_angle,
        double start_angle,
        bool clockwise,
        double step_size
    );

};

#endif // CIRCLE_H
