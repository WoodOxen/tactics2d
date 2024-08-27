// cubic_spline.h
#ifndef CUBIC_SPLINE_H
#define CUBIC_SPLINE_H

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
#include <unordered_map>

namespace py = pybind11;



class CubicSpline {
    // This class implement a cubic spline interpolator.
public:
    enum class BoundaryType {
        Natural = 1,
        Clamped,
        NotAKnot
    };

    struct SplineParameters {
        std::vector<double> a;
        std::vector<double> b;
        std::vector<double> c;
        std::vector<double> d;
    };

    CubicSpline(BoundaryType boundary_type = BoundaryType::NotAKnot)
        : boundary_type_(boundary_type) {}

    SplineParameters get_parameters(const std::vector<std::array<double, 2>>& control_points, std::pair<double, double> xx = {0.0,0.0});
    std::vector<std::array<double, 2>> get_curve(const std::vector<std::array<double, 2>>& control_points, std::pair<double, double> xx = {0.0,0.0}, int n_interpolation = 100);

private:
    BoundaryType boundary_type_;

    std::vector<double> Gauss(std::vector<std::vector<double>> &A, std::vector<double> &b);

};
#endif // CUBIC_SPLINE_H