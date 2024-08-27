// bezier.h
#ifndef BEZIER_H
#define BEZIER_H

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

class Bezier {
    //This class implement a Bezier curve interpolator.

    public:
        static std::vector<std::array<double, 2>> get_curve(
            const std::vector<std::array<double, 2>>& control_points,
            int n_interpolation
        );
};

#endif // BEZIER_H