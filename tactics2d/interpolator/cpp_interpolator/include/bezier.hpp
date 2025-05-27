#ifndef BEZIER_HPP
#define BEZIER_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

class Bezier {
   public:
    static std::vector<std::array<double, 2>> get_curve(
        const std::vector<std::array<double, 2>>& control_points, int n_interpolation);
};

#endif  // BEZIER_HPP
