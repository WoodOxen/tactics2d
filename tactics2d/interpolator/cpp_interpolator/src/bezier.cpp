#include "bezier.h"

std::vector<std::array<double, 2>> Bezier::get_curve(
    const std::vector<std::array<double, 2>>& control_points, int n_interpolation) {
    int N = control_points.size();
    std::vector<std::array<double, 2>> curve_points(n_interpolation, {0.0, 0.0});

    if (n_interpolation < 2) {
        throw std::invalid_argument(
            "n_interpolation must be greater than or equal to 2.");
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
            int c = k;
            int b = N - c - 1;
            double binomial_coeff = 1.0;
            for (int j = 0; j < c; ++j) {
                binomial_coeff *= (N - 1 - j) / double(j + 1);
            }

            double temp = binomial_coeff * powers_of_one_minus_t[b] * powers_of_t[c];

            // Accumulate the result for the x and y coordinates
            curve_points[i][0] += temp * control_points[k][0];
            curve_points[i][1] += temp * control_points[k][1];
        }
    }

    return curve_points;
}
