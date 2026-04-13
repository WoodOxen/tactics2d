#include "bezier.hpp"

/// Compute Bézier curve using Bernstein basis polynomials with precomputed binomial
/// coefficients. Implements: B(t) = Σ_{k=0}^{n} C(n,k) * (1-t)^{n-k} * t^k * P_k, where n
/// = N-1. Uses efficient recurrence relations to avoid redundant calculations.
///
/// @param control_points Control points P₀, P₁, ..., Pₙ (n = N-1).
/// @param n_interpolation Number of points to interpolate along curve.
/// @return Interpolated points along Bézier curve.
/// @throws std::invalid_argument If control_points.size() < 2 or n_interpolation < 2.
std::vector<std::array<double, 2>> Bezier::get_curve(
    const std::vector<std::array<double, 2>>& control_points, int n_interpolation) {
    int N = control_points.size();

    if (N < 2) {
        throw std::invalid_argument("Number of control points must be at least 2.");
    }

    if (n_interpolation < 2) {
        throw std::invalid_argument(
            "n_interpolation must be greater than or equal to 2.");
    }

    std::vector<std::array<double, 2>> curve_points(n_interpolation, {0.0, 0.0});

    // Precompute binomial coefficients C(N-1, k) for k = 0..N-1
    std::vector<double> binomial(N);
    binomial[0] = 1.0;
    for (int k = 1; k < N; ++k) {
        // Use recurrence: C(n, k) = C(n, k-1) * (n - k + 1) / k
        binomial[k] = binomial[k - 1] * (N - k) / k;
    }

    double delta_t = 1.0 / (n_interpolation - 1);
    double t = 0.0;

    // Precompute powers of t and (1 - t) to avoid recalculating them multiple times
    std::vector<double> powers_of_t(N), powers_of_one_minus_t(N);

    // Iterate over all interpolation points
    for (int i = 0; i < n_interpolation; ++i) {
        t = i * delta_t;
        double one_minus_t = 1.0 - t;

        // Compute powers of t and (1 - t) for each control point
        powers_of_t[0] = 1.0;            // t^0 = 1
        powers_of_one_minus_t[0] = 1.0;  // (1-t)^0 = 1
        for (int j = 1; j < N; ++j) {
            powers_of_t[j] = powers_of_t[j - 1] * t;
            powers_of_one_minus_t[j] = powers_of_one_minus_t[j - 1] * one_minus_t;
        }

        // Iterate over control points and accumulate results
        for (int k = 0; k < N; ++k) {
            int b = N - k - 1;  // exponent for (1-t)
            double temp = binomial[k] * powers_of_one_minus_t[b] * powers_of_t[k];

            // Accumulate the result for the x and y coordinates
            curve_points[i][0] += temp * control_points[k][0];
            curve_points[i][1] += temp * control_points[k][1];
        }
    }

    return curve_points;
}
