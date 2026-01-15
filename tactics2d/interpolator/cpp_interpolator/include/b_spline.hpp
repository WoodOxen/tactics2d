#ifndef BSPLINE_HPP
#define BSPLINE_HPP

#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <vector>

namespace py = pybind11;

/// B-spline curve interpolator using Cox-de Boor recursive algorithm.
/// Implements uniform B-splines with customizable degree and knot vectors.
class BSpline {
   public:
    /// Compute B-spline curve points using Cox-de Boor recurrence relation.
    ///
    /// @param control_points Control points (P₀, P₁, ..., Pₙ) as vector of (x,y) pairs.
    /// @param knot_vectors Knot vector (u₀, u₁, ..., uₘ) where m = n + degree + 1.
    ///                     Must be non-decreasing and of correct size.
    /// @param degree Degree of the B-spline curve (p in literature).
    ///               Must satisfy: 0 ≤ degree ≤ n, where n = control_points.size() - 1.
    /// @param n_interpolation Number of points to interpolate along the curve.
    ///                        Must be positive.
    /// @return Interpolated points along the B-spline curve, shape (n_interpolation, 2).
    /// @throws std::invalid_argument If parameters violate B-spline constraints
    ///                               or have invalid dimensions.
    static std::vector<std::array<double, 2>> get_curve(
        const std::vector<std::array<double, 2>>& control_points,
        std::vector<double>& knot_vectors, int degree, double n_interpolation);

   private:
    /// Compute uniform knot vector for given number of control points and degree.
    ///
    /// @param n Number of control points minus one (n in B-spline notation).
    /// @param degree Degree of the B-spline curve.
    /// @return Uniform knot vector of length n + degree + 2 with values in [0, 1].
    static std::vector<double> compute_knots(int n, int degree);

    /// Compute B-spline basis function N_{i,p}(u) using Cox-de Boor recursion.
    ///
    /// @param i Index of basis function (0 ≤ i ≤ n).
    /// @param degree Degree p of the basis function.
    /// @param u Parameter value in knot vector domain.
    /// @param knot_vectors Knot vector (u₀, u₁, ..., uₘ).
    /// @param N Temporary storage for basis function values at different degrees.
    ///          Should be pre-allocated as (degree+1) × (degree+1) matrix.
    /// @return Value of basis function N_{i,p}(u).
    static double basis_function(int i, int degree, double u,
                                 const std::vector<double>& knot_vectors,
                                 std::vector<std::vector<double>>& N);
};

#endif  // BSPLINE_HPP
