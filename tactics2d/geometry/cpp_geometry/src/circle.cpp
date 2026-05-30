#include "circle.hpp"

std::pair<std::array<double, 2>, double> Circle::get_circle_by_three_points(
    const std::array<double, 2>& point1, const std::array<double, 2>& point2,
    const std::array<double, 2>& point3) {
    // Using the perpendicular bisector method to find the circle from 3 points

    double a = point1[0] - point2[0];
    double b = point1[1] - point2[1];
    double c = point1[0] - point3[0];
    double d = point1[1] - point3[1];

    double e = (point1[0] * point1[0] - point2[0] * point2[0] + point1[1] * point1[1] -
                point2[1] * point2[1]) /
               2.0;
    double f = (point1[0] * point1[0] - point3[0] * point3[0] + point1[1] * point1[1] -
                point3[1] * point3[1]) /
               2.0;

    double denom = a * d - b * c;
    if (std::abs(denom) < 1e-10) {
        throw std::runtime_error("Cannot define a unique circle: points are collinear");
    }

    double cx = (e * d - b * f) / denom;
    double cy = (a * f - e * c) / denom;
    std::array<double, 2> center = {cx, cy};

    double dx = point1[0] - cx;
    double dy = point1[1] - cy;
    double radius = std::sqrt(dx * dx + dy * dy);

    return {center, radius};
}

std::pair<std::array<double, 2>, double> Circle::get_circle_by_tangent_vector(
    const std::array<double, 2>& tangent_point, double tangent_heading, double radius,
    const std::string& side) {
    // This function gets a circle by a tangent point, a tangent_heading and a radius.

    std::array<double, 2> vector;
    if (side == "R") {
        vector = {std::cos(tangent_heading - M_PI / 2) * radius,
                  std::sin(tangent_heading - M_PI / 2) * radius};
    } else if (side == "L") {
        vector = {std::cos(tangent_heading + M_PI / 2) * radius,
                  std::sin(tangent_heading + M_PI / 2) * radius};
    } else {
        throw std::invalid_argument("Invalid side. Use 'L' or 'R'.");
    }

    return {{tangent_point[0] + vector[0], tangent_point[1] + vector[1]}, radius};
}
