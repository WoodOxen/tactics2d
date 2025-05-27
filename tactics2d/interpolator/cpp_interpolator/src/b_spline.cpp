#include "b_spline.hpp"

std::vector<double> BSpline::compute_knots(int n, int degree) {
    // Compute uniform knot vector of length (n + degree + 1)
    std::vector<double> knot_vectors(n + degree + 1);
    double denom = static_cast<double>(n + degree);
    for (size_t i = 0; i < knot_vectors.size(); ++i) {
        knot_vectors[i] = static_cast<double>(i) / denom;
    }
    return knot_vectors;
}

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

std::vector<std::array<double, 2>> BSpline::get_curve(
    const std::vector<std::array<double, 2>>& control_points,
    std::vector<double>& knot_vectors, int degree, double n_interpolation) {
    int n = static_cast<int>(control_points.size());
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
