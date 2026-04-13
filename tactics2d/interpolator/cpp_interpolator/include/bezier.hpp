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

/// Bézier curve interpolator using Bernstein polynomials.
/// Implements the de Casteljau algorithm for efficient computation of Bézier curves.
class Bezier {
   public:
    /// Compute Bézier curve points using Bernstein basis polynomials.
    ///
    /// @param control_points Control points (P₀, P₁, ..., Pₙ) as vector of (x,y) pairs.
    ///                       Must contain at least 2 points (order ≥ 1).
    /// @param n_interpolation Number of points to interpolate along the curve.
    ///                        Must be at least 2 for meaningful interpolation.
    /// @return Interpolated points along the Bézier curve, shape (n_interpolation, 2).
    /// @throws std::invalid_argument If control_points empty, has invalid shape,
    ///                               or n_interpolation < 2.
    static std::vector<std::array<double, 2>> get_curve(
        const std::vector<std::array<double, 2>>& control_points, int n_interpolation);
};

#endif  // BEZIER_HPP
