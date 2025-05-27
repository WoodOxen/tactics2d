#include "cubic_spline.hpp"

#include <algorithm>  // For std::swap
#include <cmath>

CubicSpline::SplineParameters CubicSpline::get_parameters(
    const std::vector<std::array<double, 2>>& control_points,
    std::pair<double, double> xx, BoundaryType boundary_type) {
    // Get the parameters of the cubic functions

    size_t n = control_points.size() - 1;

    std::vector<double> x(n + 1), y(n + 1);
    for (size_t i = 0; i <= n; ++i) {
        x[i] = control_points[i][0];
        y[i] = control_points[i][1];
    }

    std::vector<double> h(n), b(n);
    for (size_t i = 0; i < n; ++i) {
        h[i] = x[i + 1] - x[i];
        b[i] = (y[i + 1] - y[i]) / h[i];
    }

    std::vector<std::vector<double>> A(n + 1, std::vector<double>(n + 1, 0.0));
    std::vector<double> B(n + 1, 0.0);

    for (size_t i = 1; i < n; ++i) {
        A[i][i - 1] = h[i - 1];
        A[i][i] = 2 * (h[i - 1] + h[i]);
        A[i][i + 1] = h[i];
        B[i] = 6 * (b[i] - b[i - 1]);
    }

    if (boundary_type == BoundaryType::Natural) {
        A[0][0] = 1;
        A[n][n] = 1;
    } else if (boundary_type == BoundaryType::Clamped) {
        A[0][0] = 2 * h[0];
        A[0][1] = h[0];
        A[n][n] = 2 * h[n - 1];
        A[n][n - 1] = h[n - 1];
        B[0] = 6 * (b[0] - xx.first);
        B[n] = 6 * (xx.second - b[n - 1]);
    } else if (boundary_type == BoundaryType::NotAKnot) {
        A[0][0] = -h[1];
        A[0][1] = h[0] + h[1];
        A[0][2] = -h[0];
        A[n][n] = -h[n - 2];
        A[n][n - 1] = h[n - 2] + h[n - 1];
        A[n][n - 2] = -h[n - 1];
    }

    std::vector<double> m = Gauss(A, B);

    std::vector<double> a(n), c(n), d(n);
    for (size_t i = 0; i < n; ++i) {
        a[i] = y[i];
        c[i] = m[i] / 2.0;
        d[i] = (m[i + 1] - m[i]) / (6.0 * h[i]);
        b[i] = b[i] - h[i] * (2 * m[i] + m[i + 1]) / 6.0;
    }

    return {a, b, c, d};
}

std::vector<std::array<double, 2>> CubicSpline::get_curve(
    const std::vector<std::array<double, 2>>& control_points,
    std::pair<double, double> xx, int n_interpolation, BoundaryType boundary_type) {
    // Get the interpolation points of a cubic spline curve

    auto [a, b, c, d] = get_parameters(control_points, xx, boundary_type);
    size_t n = control_points.size() - 1;

    std::vector<std::array<double, 2>> curve_points;
    curve_points.reserve(n * n_interpolation + 1);  // Reserve space for expected size

    for (size_t i = 0; i < n; ++i) {
        double step = (control_points[i + 1][0] - control_points[i][0]) /
                      double(n_interpolation - 1);
        double xi = control_points[i][0];

        for (int j = 0; j < n_interpolation; ++j) {
            double xi_diff = xi - control_points[i][0];
            double yi = a[i] + b[i] * xi_diff + c[i] * std::pow(xi_diff, 2) +
                        d[i] * std::pow(xi_diff, 3);
            curve_points.push_back({xi, yi});
            xi += step;
        }
    }

    curve_points.push_back(control_points[n]);  // Add last point
    return curve_points;
}

std::vector<double> CubicSpline::Gauss(std::vector<std::vector<double>>& A,
                                       std::vector<double>& b) {
    int n = A.size();
    for (int i = 0; i < n; i++) {
        int maxRow = i;
        for (int j = i + 1; j < n; j++) {
            if (std::abs(A[j][i]) > std::abs(A[maxRow][i])) {
                maxRow = j;
            }
        }
        std::swap(A[i], A[maxRow]);
        std::swap(b[i], b[maxRow]);

        for (int j = i + 1; j < n; j++) {
            double ratio = A[j][i] / A[i][i];
            for (int k = i; k < n; k++) {
                A[j][k] -= ratio * A[i][k];
            }
            b[j] -= ratio * b[i];
        }
    }

    std::vector<double> x(n);
    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= A[i][j] * x[j];
        }
        x[i] /= A[i][i];
    }

    return x;
}
