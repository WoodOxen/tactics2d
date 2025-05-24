#ifndef CUBIC_SPLINE_H
#define CUBIC_SPLINE_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace py = pybind11;

class CubicSpline {
   public:
    enum class BoundaryType { Natural = 1, Clamped, NotAKnot };

    struct SplineParameters {
        std::vector<double> a;
        std::vector<double> b;
        std::vector<double> c;
        std::vector<double> d;
    };

    CubicSpline(BoundaryType boundary_type = BoundaryType::NotAKnot)
        : boundary_type_(boundary_type) {}

    std::vector<std::array<double, 2>> get_curve(
        const std::vector<std::array<double, 2>>& control_points,
        std::pair<double, double> xx = {0.0, 0.0}, int n_interpolation = 100);

   private:
    BoundaryType boundary_type_;
    SplineParameters get_parameters(
        const std::vector<std::array<double, 2>>& control_points,
        std::pair<double, double> xx);
    std::vector<double> Gauss(std::vector<std::vector<double>>& A,
                              std::vector<double>& b);
    // std::vector<double> Thomas_algorithm(std::vector<std::vector<double>>& A,
    // std::vector<double>& b);
};

#endif  // CUBIC_SPLINE_H
