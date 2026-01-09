#include "cubic_spline.hpp"

#include <algorithm>  // For std::swap
#include <cmath>

CubicSpline::SplineParameters CubicSpline::get_parameters(
    const std::vector<std::array<double, 2>>& control_points,
    std::pair<double, double> xx, BoundaryType boundary_type) {
    // Get the parameters of the cubic functions using Thomas algorithm

    size_t n = control_points.size() - 1;

    std::vector<double> x(n + 1), y(n + 1);
    for (size_t i = 0; i <= n; ++i) {
        x[i] = control_points[i][0];
        y[i] = control_points[i][1];
    }

    std::vector<double> h(n), slope(n);
    for (size_t i = 0; i < n; ++i) {
        h[i] = x[i + 1] - x[i];
        slope[i] = (y[i + 1] - y[i]) / h[i];
    }

    // Build right-hand side vector B
    std::vector<double> B(n + 1, 0.0);
    for (size_t i = 1; i < n; ++i) {
        B[i] = 6.0 * (slope[i] - slope[i - 1]);
    }

    // Set boundary conditions in B vector
    if (boundary_type == BoundaryType::Clamped) {
        B[0] = 6.0 * (slope[0] - xx.first);
        B[n] = 6.0 * (xx.second - slope[n - 1]);
    }
    // For Natural and NotAKnot, B[0] and B[n] remain 0

    // Solve for m using Thomas algorithm
    std::vector<double> m = ThomasSolve(h, B, boundary_type, xx);

    // Compute spline parameters a, b, c, d
    std::vector<double> a(n), b(n), c(n), d(n);
    for (size_t i = 0; i < n; ++i) {
        a[i] = y[i];
        c[i] = m[i] / 2.0;
        d[i] = (m[i + 1] - m[i]) / (6.0 * h[i]);
        b[i] = slope[i] - h[i] * (2.0 * m[i] + m[i + 1]) / 6.0;
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
            double xi_diff2 = xi_diff * xi_diff;
            double yi =
                a[i] + b[i] * xi_diff + c[i] * xi_diff2 + d[i] * xi_diff2 * xi_diff;
            curve_points.push_back({xi, yi});
            xi += step;
        }
    }

    curve_points.push_back(control_points[n]);  // Add last point
    return curve_points;
}

std::vector<double> CubicSpline::Gauss(std::vector<std::vector<double>>& A,
                                       std::vector<double>& b) {
    // This is kept for backward compatibility but should not be used
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

std::vector<double> CubicSpline::ThomasSolve(const std::vector<double>& h,
                                             const std::vector<double>& B,
                                             BoundaryType boundary_type,
                                             std::pair<double, double> xx) {
    // Solve the tridiagonal system for cubic spline parameters using Thomas algorithm
    // The system is: A * m = B, where A is (n+1) x (n+1)
    int n = h.size();  // h has size n, but matrix has size n+1
    int N = n + 1;     // N = n+1

    // For Not-a-knot boundary, use Gaussian elimination (TODO: optimize)
    if (boundary_type == BoundaryType::NotAKnot) {
        std::vector<std::vector<double>> A(N, std::vector<double>(N, 0.0));
        std::vector<double> B_vec = B;

        // Set up the matrix as in original get_parameters
        for (int i = 1; i < n; ++i) {
            A[i][i - 1] = h[i - 1];
            A[i][i] = 2.0 * (h[i - 1] + h[i]);
            A[i][i + 1] = h[i];
        }

        // Not-a-knot boundary conditions
        A[0][0] = -h[1];
        A[0][1] = h[0] + h[1];
        A[0][2] = -h[0];
        A[n][n] = -h[n - 2];
        A[n][n - 1] = h[n - 2] + h[n - 1];
        A[n][n - 2] = -h[n - 1];

        return BandedSolve(h, B_vec);
    }

    // For Natural and Clamped boundaries, use Thomas algorithm
    std::vector<double> m(N, 0.0);

    // Vectors for tridiagonal matrix: lower diag a, main diag b, upper diag c
    std::vector<double> a(N, 0.0), b(N, 0.0), c(N, 0.0);
    std::vector<double> d = B;  // Right-hand side

    // Set up the tridiagonal matrix based on boundary conditions
    if (boundary_type == BoundaryType::Natural) {
        // Natural boundary conditions
        b[0] = 1.0;
        c[0] = 0.0;
        // d[0] already 0 from B

        for (int i = 1; i < n; ++i) {
            a[i] = h[i - 1];
            b[i] = 2.0 * (h[i - 1] + h[i]);
            c[i] = h[i];
            // d[i] already set from B
        }

        a[n] = 0.0;
        b[n] = 1.0;
        c[n] = 0.0;
        // d[n] already 0 from B

    } else if (boundary_type == BoundaryType::Clamped) {
        // Clamped boundary conditions
        b[0] = 2.0 * h[0];
        c[0] = h[0];
        // d[0] already contains 6*(slope[0] - xx.first) from B

        for (int i = 1; i < n; ++i) {
            a[i] = h[i - 1];
            b[i] = 2.0 * (h[i - 1] + h[i]);
            c[i] = h[i];
            // d[i] already set from B
        }

        a[n] = h[n - 1];
        b[n] = 2.0 * h[n - 1];
        c[n] = 0.0;
        // d[n] already contains 6*(xx.second - slope[n-1]) from B
    }

    // Thomas algorithm for tridiagonal system
    // Forward elimination
    std::vector<double> c_prime(N, 0.0);
    std::vector<double> d_prime(N, 0.0);

    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    for (int i = 1; i < N; ++i) {
        double denom = b[i] - a[i] * c_prime[i - 1];
        c_prime[i] = c[i] / denom;
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom;
    }

    // Back substitution
    m[N - 1] = d_prime[N - 1];
    for (int i = N - 2; i >= 0; --i) {
        m[i] = d_prime[i] - c_prime[i] * m[i + 1];
    }

    return m;
}

std::vector<double> CubicSpline::BandedSolve(const std::vector<double>& h,
                                             const std::vector<double>& B) {
    // Solve the banded system for Not-a-knot boundary conditions
    // Matrix is (n+1) x (n+1) with bandwidth 2 (pentadiagonal)
    int n = h.size();  // h has size n
    int N = n + 1;     // matrix size N = n+1

    // Use full matrix for simplicity and correctness
    // This is still more efficient than the general Gauss function
    // because we only process the band region
    std::vector<std::vector<double>> A(N, std::vector<double>(N, 0.0));
    std::vector<double> rhs = B;

    // Build the matrix as in get_parameters
    for (int i = 1; i < n; ++i) {
        A[i][i - 1] = h[i - 1];
        A[i][i] = 2.0 * (h[i - 1] + h[i]);
        A[i][i + 1] = h[i];
    }

    // Not-a-knot boundary conditions
    A[0][0] = -h[1];
    A[0][1] = h[0] + h[1];
    A[0][2] = -h[0];
    A[n][n] = -h[n - 2];
    A[n][n - 1] = h[n - 2] + h[n - 1];
    A[n][n - 2] = -h[n - 1];

    // Perform banded Gaussian elimination (bandwidth = 2)
    // We only process elements within the band
    for (int i = 0; i < N; ++i) {
        // Find pivot within the band (only check up to 2 rows below)
        int pivotRow = i;
        double maxVal = std::abs(A[i][i]);

        for (int j = i + 1; j < std::min(N, i + 3); ++j) {
            if (std::abs(A[j][i]) > maxVal) {
                maxVal = std::abs(A[j][i]);
                pivotRow = j;
            }
        }

        // Swap rows if needed
        if (pivotRow != i) {
            std::swap(A[i], A[pivotRow]);
            std::swap(rhs[i], rhs[pivotRow]);
        }

        // Check for singular matrix
        if (std::abs(A[i][i]) < 1e-12) {
            // Singular or near-singular matrix
            // Return zeros or use a fallback
            return std::vector<double>(N, 0.0);
        }

        // Normalize pivot row
        double pivot = A[i][i];
        for (int j = i; j < std::min(N, i + 3); ++j) {  // Only up to 2 columns right
            A[i][j] /= pivot;
        }
        rhs[i] /= pivot;

        // Eliminate below (only up to 2 rows below)
        for (int j = i + 1; j < std::min(N, i + 3); ++j) {
            double factor = A[j][i];
            if (std::abs(factor) > 1e-12) {
                // Subtract factor * row i from row j
                // Only process columns within the band
                for (int k = i; k < std::min(N, i + 3); ++k) {
                    A[j][k] -= factor * A[i][k];
                }
                rhs[j] -= factor * rhs[i];
                A[j][i] = 0.0;  // Already eliminated, set to zero explicitly
            }
        }
    }

    // Back substitution
    std::vector<double> x(N, 0.0);
    for (int i = N - 1; i >= 0; --i) {
        x[i] = rhs[i];
        // Subtract contributions from columns to the right
        // Only up to 2 columns to the right (band structure)
        for (int j = i + 1; j < std::min(N, i + 3); ++j) {
            x[i] -= A[i][j] * x[j];
        }
    }

    return x;
}
