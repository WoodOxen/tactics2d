#ifndef CIRCLE_HPP
#define CIRCLE_HPP

#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace py = pybind11;

class Circle {
    // This class implement some frequently operations on circle.

   public:
    static std::pair<std::array<double, 2>, double> get_circle_by_three_points(
        const std::array<double, 2>& point1, const std::array<double, 2>& point2,
        const std::array<double, 2>& point3);

    static std::pair<std::array<double, 2>, double> get_circle_by_tangent_vector(
        const std::array<double, 2>& tangent_point, double tangent_heading, double radius,
        const std::string& side);
};

#endif  // CIRCLE_HPP
