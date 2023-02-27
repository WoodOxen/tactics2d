from math import pi, cos, sin

import numpy as np
from numpy.random import randn, random
from shapely.geometry import LinearRing, Point
from shapely.affinity import affine_transform


MIN_DIST_TO_OBST = 0.1
MIN_PARA_PARK_LOT_LEN = LENGTH * 1.25
MIN_BAY_PARK_LOT_WIDTH = WIDTH + 1.2
# the distance that the obstacles out of driving area is from dest
BAY_PARK_WALL_DIST = 7.0
PARA_PARK_WALL_DIST = 4.5
origin = (0.0, 0.0)
bay_half_len = 18.0


class Position:
    "A simpler representation of State which only consider (x, y, heading)."

    def __init__(self, raw_state: list):
        self.loc: Point = Point(raw_state[:2])  # (x,y)
        self.heading: float = raw_state[2]

    def create_box(self, VehicleBox: LinearRing) -> LinearRing:
        cos_theta = np.cos(self.heading)
        sin_theta = np.sin(self.heading)
        mat = [cos_theta, -sin_theta, sin_theta, cos_theta, self.loc.x, self.loc.y]
        return affine_transform(VehicleBox, mat)

    def get_pos(
        self,
    ):
        return (self.loc.x, self.loc.y, self.heading)


def _get_box_coords(pos: list, vehicle_box: LinearRing):
    vehicle_box = Position(pos).create_box(vehicle_box)
    return list(vehicle_box.coords)[:-1]


def _random_gaussian_num(mean, std, clip_low, clip_high):
    rand_num = randn() * std + mean
    return np.clip(rand_num, clip_low, clip_high)


def _random_uniform_num(clip_low, clip_high):
    rand_num = random() * (clip_high - clip_low) + clip_low
    return rand_num


def _get_rand_pos(origin_x, origin_y, angle_min, angle_max, radius_min, radius_max):
    angle_mean = (angle_max + angle_min) / 2
    angle_std = (angle_max - angle_min) / 4
    angle_rand = _random_gaussian_num(angle_mean, angle_std, angle_min, angle_max)
    radius_rand = _random_gaussian_num(
        (radius_min + radius_max) / 2,
        (radius_max - radius_min) / 4,
        radius_min,
        radius_max,
    )
    return (
        origin_x + cos(angle_rand) * radius_rand,
        origin_y + sin(angle_rand) * radius_rand,
    )


def _gene_park_back_obst():
    """generate the obstacle on back of destination"""
    obstacle_back = LinearRing(
        (
            (origin[0] + bay_half_len, origin[1]),
            (origin[0] + bay_half_len, origin[1] - 1),
            (origin[0] - bay_half_len, origin[1] - 1),
            (origin[0] - bay_half_len, origin[1]),
        )
    )
    return obstacle_back


def _gene_bay_park_dest(vehicle_box):
    "generate the destination for bay parking cases"
    dest_yaw = _random_gaussian_num(pi / 2, pi / 36, pi * 5 / 12, pi * 7 / 12)
    rb, _, _, lb = _get_box_coords([origin[0], origin[1], dest_yaw], vehicle_box)
    min_dest_y = -min(rb[1], lb[1]) + MIN_DIST_TO_OBST
    dest_x = origin[0]
    dest_y = _random_gaussian_num(min_dest_y + 0.4, 0.2, min_dest_y, min_dest_y + 0.8)
    return [dest_x, dest_y, dest_yaw]


def _gene_bay_left_obst(dest_pos, vehicle_box):
    """generate the obstacle on left of destination"""
    _, _, car_lf, car_lb = _get_box_coords(dest_pos, vehicle_box)
    if random() < 0.5:  # generate simple obstacle
        max_dist_to_obst = 1.0
        min_dist_to_obst = 0.4 + MIN_DIST_TO_OBST
        left_obst_rf = _get_rand_pos(
            *car_lf, pi * 11 / 12, pi * 13 / 12, min_dist_to_obst, max_dist_to_obst
        )
        left_obst_rb = _get_rand_pos(
            *car_lb, pi * 11 / 12, pi * 13 / 12, min_dist_to_obst, max_dist_to_obst
        )
        obstacle_left = LinearRing(
            (
                left_obst_rf,
                left_obst_rb,
                (origin[0] - bay_half_len, origin[1]),
                (origin[0] - bay_half_len, left_obst_rf[1]),
            )
        )
    else:  # generate another vehicle as obstacle on left
        max_dist_to_obst = 1.0
        min_dist_to_obst = 0.4 + MIN_DIST_TO_OBST
        left_car_x = origin[0] - (
            WIDTH + _random_uniform_num(min_dist_to_obst, max_dist_to_obst)
        )
        left_car_yaw = _random_gaussian_num(pi / 2, pi / 36, pi * 5 / 12, pi * 7 / 12)
        rb, _, _, lb = _get_box_coords(
            [left_car_x, origin[1], left_car_yaw], vehicle_box
        )
        min_left_car_y = -min(rb[1], lb[1]) + MIN_DIST_TO_OBST
        left_car_y = _random_gaussian_num(
            min_left_car_y + 0.4, 0.2, min_left_car_y, min_left_car_y + 0.8
        )
        obstacle_left = Position([left_car_x, left_car_y, left_car_yaw]).create_box(
            vehicle_box
        )
    return obstacle_left


def _gene_bay_right_obst(dest_pos, vehicle_box, obstacle_left):
    """generate the obstacle on right of destination"""
    dest_box = Position(dest_pos).create_box(vehicle_box)
    car_rb, car_rf, _, _ = _get_box_coords(dest_pos, vehicle_box)
    generate_success = True
    dist_dest_to_left_obst = dest_box.distance(obstacle_left)
    min_dist_to_obst = (
        max(MIN_BAY_PARK_LOT_WIDTH - WIDTH - dist_dest_to_left_obst, 0)
        + MIN_DIST_TO_OBST
    )
    max_dist_to_obst = 1.0
    if random() < 0.5:  # generate simple obstacle
        right_obst_lf = _get_rand_pos(
            *car_rf, -pi / 12, pi / 12, min_dist_to_obst, max_dist_to_obst
        )
        right_obst_lb = _get_rand_pos(
            *car_rb, -pi / 12, pi / 12, min_dist_to_obst, max_dist_to_obst
        )
        obstacle_right = LinearRing(
            (
                (origin[0] + bay_half_len, right_obst_lf[1]),
                (origin[0] + bay_half_len, origin[1]),
                right_obst_lb,
                right_obst_lf,
            )
        )
    else:  # generate another vehicle as obstacle on right
        right_car_x = origin[0] + (
            WIDTH + _random_uniform_num(min_dist_to_obst, max_dist_to_obst)
        )
        right_car_yaw = _random_gaussian_num(pi / 2, pi / 36, pi * 5 / 12, pi * 7 / 12)
        rb, _, _, lb = _get_box_coords(
            [right_car_x, origin[1], right_car_yaw], vehicle_box
        )
        min_right_car_y = -min(rb[1], lb[1]) + MIN_DIST_TO_OBST
        right_car_y = _random_gaussian_num(
            min_right_car_y + 0.4, 0.2, min_right_car_y, min_right_car_y + 0.8
        )
        obstacle_right = Position([right_car_x, right_car_y, right_car_yaw]).create_box(
            vehicle_box
        )
    dist_dest_to_right_obst = dest_box.distance(obstacle_right)
    if (
        dist_dest_to_right_obst + dist_dest_to_left_obst
        < MIN_BAY_PARK_LOT_WIDTH - WIDTH
        or dist_dest_to_left_obst < MIN_DIST_TO_OBST
        or dist_dest_to_right_obst < MIN_DIST_TO_OBST
    ):
        generate_success = False
    return obstacle_right, generate_success


def gene_bay_park(vehicle_box: LinearRing):
    """
    Generate the parameters that a bay parking case need.
    Returns
    ----------
        `start` (list): [x, y, yaw]
        `dest` (list): [x, y, yaw]
        `obstacles` (list): [ obstacle (`LinearRing`) , ...]
    """
    generate_success = True
    # generate obstacle on back
    obstacle_back = _gene_park_back_obst()

    # generate dest
    dest_pos = _gene_bay_park_dest(vehicle_box)
    dest_box = Position(dest_pos).create_box(vehicle_box)

    # generate obstacle on left
    # the obstacle can be another vehicle or just a simple obstacle
    obstacle_left = _gene_bay_left_obst(dest_pos, vehicle_box)
    # generate obstacle on right
    obstacle_right, generate_success = _gene_bay_right_obst(
        dest_pos, vehicle_box, obstacle_left
    )
    # check collision
    obstacles = [obstacle_back, obstacle_left, obstacle_right]
    for obst in obstacles:
        if obst.intersects(dest_box):
            generate_success = False

    # generate obstacles out of start range
    max_obstacle_y = (
        max([np.max(np.array(obs.coords)[:, 1]) for obs in obstacles])
        + MIN_DIST_TO_OBST
    )
    other_obstcales = []
    if random() < 0.2:  # in this case only a wall will be generate
        other_obstcales = [
            LinearRing(
                (
                    (
                        origin[0] - bay_half_len,
                        BAY_PARK_WALL_DIST + max_obstacle_y + MIN_DIST_TO_OBST,
                    ),
                    (
                        origin[0] + bay_half_len,
                        BAY_PARK_WALL_DIST + max_obstacle_y + MIN_DIST_TO_OBST,
                    ),
                    (
                        origin[0] + bay_half_len,
                        BAY_PARK_WALL_DIST + max_obstacle_y + MIN_DIST_TO_OBST + 0.1,
                    ),
                    (
                        origin[0] - bay_half_len,
                        BAY_PARK_WALL_DIST + max_obstacle_y + MIN_DIST_TO_OBST + 0.1,
                    ),
                )
            )
        ]
    else:
        other_obstacle_range = LinearRing(
            (
                (origin[0] - bay_half_len, BAY_PARK_WALL_DIST + max_obstacle_y),
                (origin[0] + bay_half_len, BAY_PARK_WALL_DIST + max_obstacle_y),
                (origin[0] + bay_half_len, BAY_PARK_WALL_DIST + max_obstacle_y + 8),
                (origin[0] - bay_half_len, BAY_PARK_WALL_DIST + max_obstacle_y + 8),
            )
        )
        valid_obst_x_range = (
            origin[0] - bay_half_len + 2,
            origin[0] + bay_half_len - 2,
        )
        valid_obst_y_range = (
            BAY_PARK_WALL_DIST + max_obstacle_y + 2,
            BAY_PARK_WALL_DIST + max_obstacle_y + 6,
        )
        for _ in range(3):
            obs_x = _random_uniform_num(*valid_obst_x_range)
            obs_y = _random_uniform_num(*valid_obst_y_range)
            obs_yaw = random() * pi * 2
            obs_coords = np.array(_get_box_coords([obs_x, obs_y, obs_yaw], vehicle_box))
            obs = LinearRing(obs_coords + 0.5 * random(obs_coords.shape))
            if obs.intersects(other_obstacle_range):
                continue
            other_obstcales.append(obs)

    # merge two kind of obstacles
    obstacles.extend(other_obstcales)

    # generate start position
    start_box_valid = False
    valid_start_x_range = (origin[0] - bay_half_len / 2, origin[0] + bay_half_len / 2)
    valid_start_y_range = (max_obstacle_y + 1, BAY_PARK_WALL_DIST + max_obstacle_y - 1)
    while not start_box_valid:
        start_box_valid = True
        start_x = _random_uniform_num(*valid_start_x_range)
        start_y = _random_uniform_num(*valid_start_y_range)
        start_yaw = _random_gaussian_num(0, pi / 6, -pi / 2, pi / 2)
        start_yaw = start_yaw + pi if random() < 0.5 else start_yaw
        start_box = Position([start_x, start_y, start_yaw]).create_box(vehicle_box)
        # check collision
        for obst in obstacles:
            if obst.intersects(start_box):
                start_box_valid = False
        # check overlap with dest box
        if dest_box.intersects(start_box):
            start_box_valid = False

    # randomly drop the obstacles
    for obs in obstacles:
        if random() < 0.1:
            obstacles.remove(obs)
    # return if generate successfully
    if generate_success:
        return [start_x, start_y, start_yaw], dest_pos, obstacles
    else:
        return gene_bay_park(vehicle_box)


def _gene_parallel_dest(vehicle_box):
    "generate the destination for parallel parking cases"
    dest_yaw = _random_gaussian_num(0, pi / 36, -pi / 12, pi / 12)
    rb, rf, _, _ = _get_box_coords([origin[0], origin[1], dest_yaw], vehicle_box)
    min_dest_y = -min(rb[1], rf[1]) + MIN_DIST_TO_OBST
    dest_x = origin[0]
    dest_y = _random_gaussian_num(min_dest_y + 0.4, 0.2, min_dest_y, min_dest_y + 0.8)
    return [dest_x, dest_y, dest_yaw]


def _gene_parallel_left_obst(dest_pos, vehicle_box):
    """generate the obstacle on left of destination"""
    car_rb, _, _, car_lb = _get_box_coords(dest_pos, vehicle_box)
    if random() < 0.5:  # generate simple obstacle
        max_dist_to_obst = 1.0
        min_dist_to_obst = 0.4 + MIN_DIST_TO_OBST
        left_obst_rf = _get_rand_pos(
            *car_lb, pi * 11 / 12, pi * 13 / 12, min_dist_to_obst, max_dist_to_obst
        )
        left_obst_rb = _get_rand_pos(
            *car_rb, pi * 11 / 12, pi * 13 / 12, min_dist_to_obst, max_dist_to_obst
        )
        obstacle_left = LinearRing(
            (
                left_obst_rf,
                left_obst_rb,
                (origin[0] - bay_half_len, origin[1]),
                (origin[0] - bay_half_len, left_obst_rf[1]),
            )
        )
    else:  # generate another vehicle as obstacle on left
        max_dist_to_obst = 1.0
        min_dist_to_obst = 0.4 + MIN_DIST_TO_OBST
        left_car_x = origin[0] - (
            LENGTH + _random_uniform_num(min_dist_to_obst, max_dist_to_obst)
        )
        left_car_yaw = _random_gaussian_num(0, pi / 36, -pi / 12, pi / 12)
        rb, rf, _, _ = _get_box_coords(
            [left_car_x, origin[1], left_car_yaw], vehicle_box
        )
        min_left_car_y = -min(rb[1], rf[1]) + MIN_DIST_TO_OBST
        left_car_y = _random_gaussian_num(
            min_left_car_y + 0.4, 0.2, min_left_car_y, min_left_car_y + 0.8
        )
        obstacle_left = Position([left_car_x, left_car_y, left_car_yaw]).create_box(
            vehicle_box
        )
    return obstacle_left


def _gene_parallel_right_obst(dest_pos, vehicle_box, obstacle_left):
    """generate the obstacle on right of destination"""
    dest_box = Position(dest_pos).create_box(vehicle_box)
    _, car_rf, car_lf, _ = _get_box_coords(dest_pos, vehicle_box)
    dist_dest_to_left_obst = dest_box.distance(obstacle_left)
    min_dist_to_obst = (
        max(MIN_PARA_PARK_LOT_LEN - LENGTH - dist_dest_to_left_obst, 0)
        + MIN_DIST_TO_OBST
    )
    max_dist_to_obst = 1.0
    if random() < 0.5:  # generate simple obstacle
        right_obst_lf = _get_rand_pos(
            *car_lf, -pi / 12, pi / 12, min_dist_to_obst, max_dist_to_obst
        )
        right_obst_lb = _get_rand_pos(
            *car_rf, -pi / 12, pi / 12, min_dist_to_obst, max_dist_to_obst
        )
        obstacle_right = LinearRing(
            (
                (origin[0] + bay_half_len, right_obst_lf[1]),
                (origin[0] + bay_half_len, origin[1]),
                right_obst_lb,
                right_obst_lf,
            )
        )
    else:  # generate another vehicle as obstacle on right
        right_car_x = origin[0] + (
            LENGTH + _random_uniform_num(min_dist_to_obst, max_dist_to_obst)
        )
        right_car_yaw = _random_gaussian_num(0, pi / 36, -pi / 12, pi / 12)
        rb, rf, _, _ = _get_box_coords(
            [right_car_x, origin[1], right_car_yaw], vehicle_box
        )
        min_right_car_y = -min(rb[1], rf[1]) + MIN_DIST_TO_OBST
        right_car_y = _random_gaussian_num(
            min_right_car_y + 0.4, 0.2, min_right_car_y, min_right_car_y + 0.8
        )
        obstacle_right = Position([right_car_x, right_car_y, right_car_yaw]).create_box(
            vehicle_box
        )
    dist_dest_to_right_obst = dest_box.distance(obstacle_right)
    if (
        dist_dest_to_right_obst + dist_dest_to_left_obst < LENGTH / 4
        or dist_dest_to_left_obst < MIN_DIST_TO_OBST
        or dist_dest_to_right_obst < MIN_DIST_TO_OBST
    ):
        generate_success = False
    return obstacle_right, generate_success


def gene_parallel_park(vehicle_box: LinearRing):
    """
    Generate the parameters that a parallel parking case need.

    Returns
    ----------
        `start` (list): [x, y, yaw]
        `dest` (list): [x, y, yaw]
        `obstacles` (list): [ obstacle (`LinearRing`) , ...]
    """
    generate_success = True
    # generate obstacle on back
    obstacle_back = _gene_park_back_obst()
    # generate dest
    dest_pos = _gene_parallel_dest(vehicle_box)
    dest_x, dest_y, dest_yaw = dest_pos
    dest_box = Position(dest_pos).create_box(vehicle_box)
    # generate obstacle on left
    # the obstacle can be another vehicle or just a simple obstacle
    obstacle_left = _gene_parallel_left_obst(dest_pos, vehicle_box)
    # generate obstacle on right
    obstacle_right, generate_success = _gene_parallel_right_obst(
        dest_pos, vehicle_box, obstacle_left
    )
    # check collision
    obstacles = [obstacle_back, obstacle_left, obstacle_right]
    for obst in obstacles:
        if obst.intersects(dest_box):
            generate_success = False

    # generate obstacles out of start range
    max_obstacle_y = (
        max([np.max(np.array(obs.coords)[:, 1]) for obs in obstacles])
        + MIN_DIST_TO_OBST
    )
    other_obstcales = []
    if random() < 0.2:  # in this case only a wall will be generate
        other_obstcales = [
            LinearRing(
                (
                    (
                        origin[0] - bay_half_len,
                        PARA_PARK_WALL_DIST + max_obstacle_y + MIN_DIST_TO_OBST,
                    ),
                    (
                        origin[0] + bay_half_len,
                        PARA_PARK_WALL_DIST + max_obstacle_y + MIN_DIST_TO_OBST,
                    ),
                    (
                        origin[0] + bay_half_len,
                        PARA_PARK_WALL_DIST + max_obstacle_y + MIN_DIST_TO_OBST + 0.1,
                    ),
                    (
                        origin[0] - bay_half_len,
                        PARA_PARK_WALL_DIST + max_obstacle_y + MIN_DIST_TO_OBST + 0.1,
                    ),
                )
            )
        ]
    else:
        other_obstacle_range = LinearRing(
            (
                (origin[0] - bay_half_len, PARA_PARK_WALL_DIST + max_obstacle_y),
                (origin[0] + bay_half_len, PARA_PARK_WALL_DIST + max_obstacle_y),
                (origin[0] + bay_half_len, PARA_PARK_WALL_DIST + max_obstacle_y + 8),
                (origin[0] - bay_half_len, PARA_PARK_WALL_DIST + max_obstacle_y + 8),
            )
        )
        valid_obst_x_range = (
            origin[0] - bay_half_len + 2,
            origin[0] + bay_half_len - 2,
        )
        valid_obst_y_range = (
            PARA_PARK_WALL_DIST + max_obstacle_y + 2,
            PARA_PARK_WALL_DIST + max_obstacle_y + 6,
        )
        for _ in range(3):
            obs_x = _random_uniform_num(*valid_obst_x_range)
            obs_y = _random_uniform_num(*valid_obst_y_range)
            obs_yaw = random() * pi * 2
            obs_coords = np.array(_get_box_coords([obs_x, obs_y, obs_yaw], vehicle_box))
            obs = LinearRing(obs_coords + 0.5 * random(obs_coords.shape))
            if obs.intersects(other_obstacle_range):
                continue
            other_obstcales.append(obs)

    # merge two kind of obstacles
    obstacles.extend(other_obstcales)

    # generate start position
    start_box_valid = False
    valid_start_x_range = (origin[0] - bay_half_len / 2, origin[0] + bay_half_len / 2)
    valid_start_y_range = (max_obstacle_y + 1, PARA_PARK_WALL_DIST + max_obstacle_y - 1)
    while not start_box_valid:
        start_box_valid = True
        start_x = _random_uniform_num(*valid_start_x_range)
        start_y = _random_uniform_num(*valid_start_y_range)
        start_yaw = _random_gaussian_num(0, pi / 6, -pi / 2, pi / 2)
        start_yaw = start_yaw + pi if random() < 0.5 else start_yaw
        start_box = Position([start_x, start_y, start_yaw]).create_box(vehicle_box)
        # check collision
        for obst in obstacles:
            if obst.intersects(start_box):
                start_box_valid = False
        # check overlap with dest box
        if dest_box.intersects(start_box):
            start_box_valid = False

    # flip the dest box so that the orientation of start matches the dest
    if cos(start_yaw) < 0:
        dest_box_center = np.mean(np.array(dest_box.coords[:-1]), axis=0)
        dest_x = 2 * dest_box_center[0] - dest_x
        dest_y = 2 * dest_box_center[1] - dest_y
        dest_yaw += pi

    # randomly drop the obstacles
    for obs in obstacles:
        if random() < 0.1:
            obstacles.remove(obs)
    # return if success
    if generate_success:
        return [start_x, start_y, start_yaw], [dest_x, dest_y, dest_yaw], obstacles
    else:
        return gene_parallel_park(vehicle_box)
