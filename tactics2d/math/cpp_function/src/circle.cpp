// circle.cpp
#include "circle.h"


std::pair<std::array<double, 2>, double> Circle::get_circle_by_three_points(
    const std::array<double, 2>& pt1,
    const std::array<double, 2>& pt2,
    const std::array<double, 2>& pt3
) {
    /*This function gets a circle by three points.

    Args:
        pt1 (std::array<double, 2>): The first point. The shape is (2,).
        pt2 (std::array<double, 2>): The second point. The shape is (2,).
        pt3 (std::array<double, 2>): The third point. The shape is (2,).

    Returns:
        center (std::array<double, 2>): The center of the circle. The shape is (2,).
        radius (double): The radius of the circle.
   */

    double a = pt1[0] - pt2[0];
	double b = pt1[1] - pt2[1];
	double c = pt1[0] - pt3[0];
	double d = pt1[1] - pt3[1];
	double e = ((pt1[0] * pt1[0] - pt2[0] * pt2[0])-(pt2[1] * pt2[1] - pt1[1] * pt1[1])) / 2;
	double f = ((pt1[0] * pt1[0] - pt3[0] * pt3[0])-(pt3[1] * pt3[1] - pt1[1] * pt1[1])) / 2;
	

    double denom = a * d - b * c;
    if (std::abs(denom) < 1e-10) {
        throw std::runtime_error("Points are collinear");
    }


    double x = (e * d - b * f) / denom;
    double y = (a * f - e * c) / denom;

    std::array<double, 2> center = {x, y};

    double temp1 = pt1[0] - center[0];
    double temp2 = pt1[1] - center[1];
    double radius = std::sqrt(temp1 * temp1 + temp2 * temp2);

    return {center, radius};
}

std::pair<std::array<double, 2>, double> Circle::get_circle_by_tangent_vector(
    const std::array<double, 2>& tangent_point,
    double heading,
    double radius,
    const std::string& side
) {
    /*This function gets a circle by a tangent point, a heading and a radius.

    Args:
        tangent_point (std::array<double, 2>): The tangent point on the circle. The shape is (2,).
        heading (double): The heading of the tangent point. The unit is radian.
        radius (double): The radius of the circle.
        side (int): The location of circle center relative to the tangent point. "L" represents left. "R" represents right.

    Returns:
        center (std::array<double, 2>): The center of the circle. The shape is (2,).
        radius (double): The radius of the circle.
    */

    std::array<double, 2> vec;
    if (side == "R") {
        vec = {std::cos(heading - M_PI / 2) * radius, std::sin(heading - M_PI / 2) * radius};
    } else if (side == "L") {
        vec = {std::cos(heading + M_PI / 2) * radius, std::sin(heading + M_PI / 2) * radius};
    } else {
        throw std::invalid_argument("Invalid side. Use 'L' or 'R'.");
    }

    return { {tangent_point[0] + vec[0], tangent_point[1] + vec[1]}, radius };
}

std::pair<std::array<double, 2>, double> Circle::get_circle(ConstructBy method, py::args args) {
    /*This function gets a circle by different given conditions.

    Args:
        method (ConstructBy): The method to derive a circle. The available choices are Circle.ConstructBy.ThreePoints) and Circle.ConstructBy.TangentVector).
        *args (py::args): The arguments of the method.

    Returns:
        center (std::array<double, 2>): The center of the circle. The shape is (2,).
        radius (double: The radius of the circle.
    */

    if (method == ConstructBy::ThreePoints) {
        return get_circle_by_three_points(
            args[0].cast<std::array<double, 2>>(),
            args[1].cast<std::array<double, 2>>(),
            args[2].cast<std::array<double, 2>>()
        );
    } else if (method == ConstructBy::TangentVector) {
        if (args.size() != 4) {
            throw std::invalid_argument("TangentVector method requires 4 arguments");
        }
        return get_circle_by_tangent_vector(
            args[0].cast<std::array<double, 2>>(),
            args[1].cast<double>(),
            args[2].cast<double>(),
            args[3].cast<std::string>()
        );
    } else {
        throw std::invalid_argument("Invalid method");
    }    
}


std::vector<std::array<double, 2>> Circle::get_arc(
    const std::array<double, 2>& center_point,
    double radius,
    double delta_angle,
    double start_angle,
    bool clockwise,
    double step_size
) {
    /*This function gets the points on an arc curve line.

    Args:
        center_point (std::array<double, 2>): The center of the arc. The shape is (2,).
        radius (double): The radius of the arc.
        delta_angle (double): The angle of the arc. This values is expected to be positive. The unit is radian.
        start_angle (double): The start angle of the arc. The unit is radian.
        clockwise (bool): The direction of the arc. True represents clockwise. False represents counterclockwise.
        step_size (double): The step size of the arc. The unit is meter.

    Returns:
        arc_points(std::array<double, 2>): The points on the arc. The shape is (int(radius * delta / step_size), 2).
    */

    double angle_step = step_size / radius;
    int num_steps = static_cast<int>(std::ceil(delta_angle / angle_step));

    std::vector<std::array<double, 2>> arc_points_vec(num_steps);

    if (clockwise) {
        angle_step = -angle_step;
    } else {
        angle_step = angle_step;
    }
    double angle = start_angle - angle_step;
    for (int i = 0; i < num_steps; ++i) {
        
        angle += angle_step;

        arc_points_vec[i][0] = center_point[0] + radius * std::cos(angle);
        arc_points_vec[i][1] = center_point[1] + radius * std::sin(angle);
    }

    return arc_points_vec;
}

