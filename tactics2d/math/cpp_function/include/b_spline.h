// b_spline.h
#ifndef BSPLINE_H
#define BSPLINE_H

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

class BSpline {
    //This class implements a B-spline curve interpolator.

public:
    static std::vector<std::array<double, 2>> get_curve(
        const std::vector<std::array<double, 2>>& control_points,
        std::vector<double>& knot_vectors,
        int degree,
        double n_interpolation
    );



private:
    static std::vector<double> compute_knots(int n, int degree);

    static double basis_function(int i, int degree, double u, const std::vector<double>& knot_vectors, std::vector<std::vector<double>>& N);

};

#endif // BSPLINE_H