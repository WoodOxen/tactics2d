#include "bezier.h"



std::vector<std::array<double, 2>> Bezier::get_curve(
    const std::vector<std::array<double, 2>>& control_points,
    int n_interpolation
) {
    /*
    Args:
        control_points (std::vector<std::array<double, 2>>): The control points of the curve. The shape is (order + 1, 2).
        n_interpolation (int): The number of interpolations.

    Returns:
        curve_points (std::vector<std::array<double, 2>>): The interpolated points of the curve. The shape is (n_interpolation, 2).
    */

    int N = control_points.size();
    std::vector<std::vector<double>> ta(N, std::vector<double>(N, 0.0));
    ta.reserve(N); // Reserve memory in advance
    // Initialize the left and right sides of Pascal's triangle to 1
    for (int i = 0; i < N; ++i) {
        ta[i][0] = 1;
        ta[i][i] = 1;
    }

     // Calculate Pascal's triangle
    for (int row = 2; row < N; ++row) {
        for (int col = 1; col < row; ++col) {
            ta[row][col] = ta[row - 1][col - 1] + ta[row - 1][col];
        }
    }

    std::vector<std::array<double, 2>> curve_points(n_interpolation, {0.0, 0.0});
    curve_points.reserve(n_interpolation); // Reserve memory in advance
    double delta_t = 1.0 / (n_interpolation - 1);
    double t = -delta_t;
    double one_minus_t = 1 - t;
    for (int i = 0; i < n_interpolation; ++i) {
        t += delta_t;
        one_minus_t -= delta_t;
        for (int k = 0; k < N; ++k) {
            int c = k;  
            int b = N - c - 1;  
            double a = ta[N - 1][k]; 

            double one_minus_t_pow_b = std::pow(one_minus_t, b);
            double t_pow_c = std::pow(t, c);
            double temp = a * one_minus_t_pow_b * t_pow_c;
            // Calculate the x and y coordinates of the point
            curve_points[i][0] += temp * control_points[k][0];
            curve_points[i][1] += temp * control_points[k][1];
        }
    }

    return curve_points;
}
