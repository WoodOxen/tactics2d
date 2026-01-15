#ifndef CUBIC_SPLINE_HPP
#define CUBIC_SPLINE_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace py = pybind11;

/// Cubic spline interpolator with three boundary condition types.
/// Uses Thomas algorithm for solving tridiagonal systems efficiently.
class CubicSpline {
   public:
    /// Boundary condition types for cubic spline interpolation.
    enum class BoundaryType {
        Natural = 1,  ///< Second derivative set to zero at endpoints.
        Clamped = 2,  ///< First derivative specified at endpoints.
        NotAKnot = 3  ///< Continuous third derivative at second and second-last knots.
    };

    /// Compute cubic spline interpolation points.
    ///
    /// @param control_points Control points (x₀,y₀), (x₁,y₁), ..., (xₙ,yₙ).
    ///                       x-values must be strictly increasing.
    /// @param first_derivatives First derivatives at endpoints (f'(x₀), f'(xₙ)).
    ///                          Only used for Clamped boundary condition.
    ///                          Default is (0.0, 0.0).
    /// @param n_interpolation Number of interpolation points between each pair of
    ///                        control points. Must be at least 2.
    /// @param boundary_type Type of boundary condition to apply.
    /// @return Interpolated points along cubic spline curve.
    /// @throws std::invalid_argument If control_points invalid or parameters out of
    /// range.
    static std::vector<std::array<double, 2>> get_curve(
        const std::vector<std::array<double, 2>>& control_points,
        std::pair<double, double> first_derivatives = {0.0, 0.0},
        int n_interpolation = 100, BoundaryType boundary_type = BoundaryType::NotAKnot);

   private:
    /// Coefficients of cubic polynomial segments: a + b(x-x_i) + c(x-x_i)² + d(x-x_i)³.
    struct SplineParameters {
        std::vector<double> a;  ///< Constant coefficients.
        std::vector<double> b;  ///< Linear coefficients.
        std::vector<double> c;  ///< Quadratic coefficients.
        std::vector<double> d;  ///< Cubic coefficients.
    };

    /// Compute spline parameters for given control points and boundary conditions.
    ///
    /// @param control_points Control points with strictly increasing x-values.
    /// @param first_derivatives First derivatives at endpoints for Clamped condition.
    /// @param boundary_type Type of boundary condition.
    /// @return Spline parameters for each segment.
    static SplineParameters get_parameters(
        const std::vector<std::array<double, 2>>& control_points,
        std::pair<double, double> first_derivatives, BoundaryType boundary_type);

    /// Solve linear system using Gaussian elimination (kept for backward compatibility).
    /// @deprecated Prefer ThomasSolve for tridiagonal systems.
    static std::vector<double> Gauss(std::vector<std::vector<double>>& A,
                                     std::vector<double>& b);

    /// Solve tridiagonal system using Thomas algorithm (TDMA).
    ///
    /// @param h Interval lengths between control points (h_i = x_{i+1} - x_i).
    /// @param B Right-hand side vector of linear system.
    /// @param boundary_type Boundary condition type.
    /// @param first_derivatives First derivatives at endpoints for Clamped condition.
    /// @return Solution vector m (second derivatives at knots).
    static std::vector<double> ThomasSolve(const std::vector<double>& h,
                                           const std::vector<double>& B,
                                           BoundaryType boundary_type,
                                           std::pair<double, double> first_derivatives);

    /// Solve banded system for Not-a-Knot boundary condition.
    ///
    /// @param h Interval lengths between control points.
    /// @param B Right-hand side vector.
    /// @return Solution vector m (second derivatives at knots).
    static std::vector<double> BandedSolve(const std::vector<double>& h,
                                           const std::vector<double>& B);
};

#endif  // CUBIC_SPLINE_HPP
