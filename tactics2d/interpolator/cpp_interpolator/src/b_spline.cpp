#include "b_spline.hpp"

/// Compute uniform knot vector for open uniform B-spline.
/// Knots are evenly spaced in [0, 1]: u_i = i / (n + degree) for i = 0..(n+degree).
///
/// @param n Number of control points minus one (n in B-spline notation).
/// @param degree Degree p of the B-spline curve.
/// @return Uniform knot vector of length n + degree + 1.
std::vector<double> BSpline::compute_knots(int n, int degree) {
    // Compute uniform knot vector of length (n + degree + 1)
    std::vector<double> knot_vectors(n + degree + 1);
    double denom = static_cast<double>(n + degree);
    for (size_t i = 0; i < knot_vectors.size(); ++i) {
        knot_vectors[i] = static_cast<double>(i) / denom;
    }
    return knot_vectors;
}

/// Compute B-spline basis function N_{i,p}(u) using Cox-de Boor recurrence.
/// Implements: N_{i,0}(u) = 1 if u_i ≤ u < u_{i+1}, else 0.
///             N_{i,p}(u) = (u - u_i)/(u_{i+p} - u_i) * N_{i,p-1}(u)
///                         + (u_{i+p+1} - u)/(u_{i+p+1} - u_{i+1}) * N_{i+1,p-1}(u)
///
/// @param i Basis function index (0 ≤ i ≤ n).
/// @param degree Degree p of basis function.
/// @param u Parameter value in knot vector domain.
/// @param knot_vectors Knot vector (u₀, u₁, ..., uₘ).
/// @param N Temporary storage for basis function values at different degrees.
///          Should be pre-allocated as (degree+1) × (degree+1) matrix.
/// @return Value of basis function N_{i,p}(u).
double BSpline::basis_function(int i, int degree, double u,
                               const std::vector<double>& knot_vectors,
                               std::vector<std::vector<double>>& N) {
    // Initialize zeroth-degree basis functions
    for (int j = 0; j <= degree; ++j) {
        int idx = i + j;
        if (idx < static_cast<int>(knot_vectors.size()) - 1) {
            N[j][0] = (knot_vectors[idx] <= u && u < knot_vectors[idx + 1]) ? 1.0 : 0.0;
        } else {
            N[j][0] = 0.0;
        }
    }

    // Recursively build higher-degree basis functions
    for (int p = 1; p <= degree; ++p) {
        for (int j = 0; j <= degree - p; ++j) {
            int left_idx = i + j;
            int right_idx = i + j + 1;

            double denom_left = knot_vectors[left_idx + p] - knot_vectors[left_idx];
            double denom_right = knot_vectors[right_idx + p] - knot_vectors[right_idx];

            double left_term =
                (denom_left != 0.0)
                    ? ((u - knot_vectors[left_idx]) / denom_left) * N[j][p - 1]
                    : 0.0;

            double right_term =
                (denom_right != 0.0)
                    ? ((knot_vectors[right_idx + p] - u) / denom_right) * N[j + 1][p - 1]
                    : 0.0;

            N[j][p] = left_term + right_term;
        }
    }

    return N[0][degree];
}

/// Compute B-spline curve points using Cox-de Boor algorithm.
/// Implements: C(u) = Σ_{i=0}^{n} N_{i,p}(u) * P_i for u in [u_p, u_n].
/// Uses uniform parameter sampling in the valid knot interval.
///
/// @param control_points Control points P₀, P₁, ..., Pₙ.
/// @param knot_vectors Knot vector (u₀, u₁, ..., uₘ) where m = n + degree + 1.
/// @param degree Degree p of B-spline curve.
/// @param n_interpolation Number of points to interpolate along curve.
/// @return Interpolated points along B-spline curve.
/// @throws std::invalid_argument If parameters violate B-spline constraints.
std::vector<std::array<double, 2>> BSpline::get_curve(
    const std::vector<std::array<double, 2>>& control_points,
    std::vector<double>& knot_vectors, int degree, double n_interpolation) {
    int n = static_cast<int>(control_points.size());

    // Input validation
    if (degree < 0) {
        throw std::invalid_argument("Degree must be non-negative.");
    }

    if (n < degree + 1) {
        throw std::invalid_argument(
            "Number of control points must be at least degree + 1.");
    }

    size_t expected_knots = n + degree + 1;
    if (knot_vectors.size() != expected_knots) {
        throw std::invalid_argument("Knot vector size must be n + degree + 1.");
    }

    // Check knot vector is non-decreasing
    for (size_t i = 1; i < knot_vectors.size(); ++i) {
        if (knot_vectors[i] < knot_vectors[i - 1]) {
            throw std::invalid_argument("Knot vector must be non-decreasing.");
        }
    }

    if (n_interpolation <= 0) {
        throw std::invalid_argument("Number of interpolation points must be positive.");
    }

    std::vector<std::array<double, 2>> curve_points(static_cast<size_t>(n_interpolation),
                                                    {0.0, 0.0});

    double u_start = knot_vectors[degree];
    double u_end = knot_vectors[n];  // open uniform B-spline: [u_p, u_n]
    double delta_u = (u_end - u_start) / n_interpolation;

    // Allocate once for reuse
    std::vector<std::vector<double>> N(degree + 1, std::vector<double>(degree + 1));

    for (int j = 0; j < static_cast<int>(n_interpolation); ++j) {
        double u = u_start + j * delta_u;
        std::array<double, 2> point = {0.0, 0.0};

        for (int i = 0; i < n; ++i) {
            double basis = basis_function(i, degree, u, knot_vectors, N);
            point[0] += basis * control_points[i][0];
            point[1] += basis * control_points[i][1];
        }

        curve_points[j] = point;
    }

    return curve_points;
}
