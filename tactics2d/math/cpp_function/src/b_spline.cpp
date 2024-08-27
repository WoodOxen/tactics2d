#include "b_spline.h"


std::vector<double> BSpline::compute_knots(int n, int degree) {
    /* Compute the uniform knots for a B-spline curve.

    Args:
        n (int): The number of control points minus one. 
        degree (int): The degree of the B-spline.

    Returns:
        knot_vectors (std::vector<double>): A vector containing the knots of the B-spline curve. The size of the vector is n + p + 1.
    */

    std::vector<double> knot_vectors(n + degree + 1, 0.0);
    for (int i = 0; i <= n + degree + 1; ++i) {
        knot_vectors[i] = static_cast<double>(i) / (n + degree + 1);
    }
    return knot_vectors;
}

double BSpline::basis_function(int i, int degree, double u, const std::vector<double>& knot_vectors, std::vector<std::vector<double>>& N) {

    /* Calculate the B-spline basis function value at a given parameter u.
    
    Args:
        i (int): The index of the control point. It is used to determine the segment of the B-spline curve.
        degree (int): The degree of the B-spline curve, which determines the number of control points influencing a given point on the curve.
        u (double): The parameter at which the B-spline basis function is to be evaluated.
        knot_vectors (const std::vector<double>&): The knot vector of the B-spline curve.
        N (std::vector<std::vector<double>>&): A reference to a matrix that will store the computed basis function values for each control point and each order up to the specified degree.
    
    Returns:
        N (std::vector<std::vector<double>>&): A reference to a matrix that will store the computed basis function values for each control point and each order up to the specified degree.
    */

    // Initialize the basis functions of order 0, i.e., degree = 1
    for (int j = 0; j <= degree; ++j) {
        if (i == 0 || j == degree) { // Basis function of order 0 for j has not been calculated
            N[j][0] = (knot_vectors[i + j] <= u && u < knot_vectors[i + j + 1]) ? 1.0 : 0.0;
        }else{ // // Basis function of order 0 for j has been calculated
                N[j][0] = N[j + 1][0]; // Shift
            
        }
    }

    // Iteratively calculate higher order basis functions
    for (int p = 1; p <= degree; ++p) {
        for (int j = 0; j <= degree - p; ++j) {
            if(i == 0 || j == degree - p){
                double denominator_left = (knot_vectors[i + j + p] - knot_vectors[i + j]);
                double denominator_right = (knot_vectors[i + j + p + 1] - knot_vectors[i + j + 1]);

                double left_term = (denominator_left != 0) ? ((u - knot_vectors[i + j]) / denominator_left) * N[j][p - 1] : 0.0;
                double right_term = (denominator_right != 0) ? ((knot_vectors[i + j + p + 1] - u) / denominator_right) * N[j + 1][p - 1] : 0.0;

                N[j][p] = left_term + right_term;
            }else{
                
                N[j][p] = N[j + 1][p]; // Shift

        }
    }
    }

    return N[0][degree];
}


std::vector<std::array<double, 2>> BSpline::get_curve(
    const std::vector<std::array<double, 2>>& control_points,
    std::vector<double>& knot_vectors,
    int degree,
    double n_interpolation
) {
    /*
    Get the interpolation points of a b-spline curve.

    Args:
        control_points (std::vector<std::array<double, 2>>): The control points of the curve. Usually denoted as $P_0, P_1, \dots, P_n$ in literature. The shape is $(n + 1, 2)$.
        knot_vectors (std::vector<double>): The knots of the curve. Usually denoted as $u_0, u_1, \dots, u_t$ in literature. The shape is $(t + 1, )$.
        degree (int): The degree of the curve.
        n_interpolation (int): The number of interpolation points.

    Returns:
        curve_points (np.ndarray): The interpolation points of the curve. The shape is (n_interpolation, 2).
    */

    int n = control_points.size();
    std::vector<std::array<double, 2>> curve_points(n_interpolation, {0.0, 0.0});
    double ustart = knot_vectors[degree];
    double uend = knot_vectors[n ];
    double delta_u = (uend - ustart) / (n_interpolation );
    double u = ustart;
    std::vector<std::vector<double>> N(degree + 1, std::vector<double>(degree + 1, 0.0));

    for (int j = 0;j < n_interpolation; j++) {
        for (int i = 0; i < n; i++) {
            double basis = basis_function(i, degree + 0, u, knot_vectors,N);
            curve_points[j][0] += basis * control_points[i][0];
            curve_points[j][1] += basis * control_points[i][1];
        }
        u += delta_u;
    }

    return curve_points;
}
