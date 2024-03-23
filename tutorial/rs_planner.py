import sys

sys.path.append(".")
sys.path.append("..")

import numpy as np
from heapdict import heapdict
from shapely.geometry import LinearRing, LineString, Point

from tactics2d.math.interpolate import ReedsShepp


class RsPlanner:
    def __init__(self, vehicle, lidar_num=120, lidar_range=20.0) -> None:
        self.radius = vehicle.wheel_base / np.tan(0.75)
        self.VehicleBox = vehicle.geometry
        self.lidar_num = lidar_num
        self.vehicle_base = self.init_vehicle_base()
        self.center_shift = 0.5 * vehicle.length - vehicle.rear_overhang
        self.start_pos = [0, 0, 0]
        self.move_vehicle_center()
        self.distance_tolerance = 0.05
        self.lidar_range = lidar_range
        self.threshold_distance = lidar_range - 2.0

    def init_vehicle_base(self):
        self.lidar_lines = []
        lidar_num = self.lidar_num
        lidar_range = 100.0
        for a in range(lidar_num):
            self.lidar_lines.append(
                LineString(
                    (
                        (0, 0),
                        (
                            np.cos(a * np.pi / lidar_num * 2) * lidar_range,
                            np.sin(a * np.pi / lidar_num * 2) * lidar_range,
                        ),
                    )
                )
            )
        lidar_base = []
        ORIGIN = Point((0, 0))
        for l in self.lidar_lines:
            distance = l.intersection(self.VehicleBox).distance(ORIGIN)
            lidar_base.append(distance)
        return np.array(lidar_base)

    def move_vehicle_center(
        self,
    ):
        vehicle_coords = np.array(self.VehicleBox.coords[:4])
        vehicle_coords[:, 0] = vehicle_coords[:, 0] + self.center_shift
        self.VehicleBox = LinearRing(vehicle_coords)
        self.start_pos[0] = self.start_pos[0] - self.center_shift

    def get_rs_path(self, info):
        startX, startY, startYaw = self.start_pos
        dest_coords = np.mean(np.array(list(info["target_area"].geometry.exterior.coords)), axis=0)
        dest_heading = info["target_heading"]
        if self.center_shift != 0:
            dest_coords[0] -= self.center_shift * np.cos(dest_heading)
            dest_coords[1] -= self.center_shift * np.sin(dest_heading)
        ego_pos = (info["state"].x, info["state"].y, info["state"].heading)
        dest_pos = (dest_coords[0], dest_coords[1], dest_heading)
        self.dest_pos = dest_pos
        rel_distance = np.sqrt((dest_pos[0] - ego_pos[0]) ** 2 + (dest_pos[1] - ego_pos[1]) ** 2)
        if rel_distance > self.threshold_distance:
            return None

        rel_angle = np.arctan2(dest_pos[1] - ego_pos[1], dest_pos[0] - ego_pos[0]) - ego_pos[2]
        rel_dest_heading = dest_pos[2] - ego_pos[2]
        goalX, goalY, goalYaw = (
            rel_distance * np.cos(rel_angle),
            rel_distance * np.sin(rel_angle),
            rel_dest_heading,
        )

        rs_planner1 = ReedsShepp(self.radius)
        #  Find all possible reeds-shepp paths between current and goal node
        reedsSheppPaths1 = rs_planner1.get_all_path(
            np.array([startX, startY]), startYaw, np.array([goalX, goalY]), goalYaw
        )

        # Check if reedsSheppPaths is empty
        if not reedsSheppPaths1:
            return None

        # Find path with lowest cost considering non-holonomic constraints
        costQueue1 = heapdict()  # TODO: deprecate the use of heapdict
        for path in reedsSheppPaths1:
            if path is None:
                continue
            costQueue1[path] = path.length

        # Find first path in priority queue that is collision free
        min_path_len = -1
        obstacles_params = self.construct_obstacles(info)
        while len(costQueue1) != 0:
            path = costQueue1.popitem()[0]
            if min_path_len < 0:
                min_path_len = path.length
            if path.length > 2 * min_path_len:
                break

            path.get_curve_line(np.array([startX, startY]), startYaw, self.radius, 0.1)
            traj1 = [
                [path.curve[k][0], path.curve[k][1], path.yaw[k]] for k in range(len(path.yaw))
            ]
            traj_valid1 = self.is_traj_valid(traj1, obstacles_params)
            if traj_valid1:
                return path

        return None

    def construct_obstacles(self, info):
        lidar_obs = info["lidar"]
        lidar_obs = np.clip(lidar_obs, 0.0, self.lidar_range)
        assert len(lidar_obs) == self.lidar_num
        lidar_obs = np.maximum(self.vehicle_base, lidar_obs - self.distance_tolerance)
        angle_vec = np.arange(self.lidar_num) * np.pi / self.lidar_num * 2
        obstacle_range_x1 = np.cos(angle_vec) * lidar_obs  # (N,)
        obstacle_range_y1 = np.sin(angle_vec) * lidar_obs
        obstacle_range_coords = np.concatenate(
            (np.expand_dims(obstacle_range_x1, 1), np.expand_dims(obstacle_range_y1, 1)), axis=1
        )  # (N, 2)
        shifted_obstacle_coords = obstacle_range_coords.copy()
        shifted_obstacle_coords[:-1] = obstacle_range_coords[1:]
        shifted_obstacle_coords[-1] = obstacle_range_coords[0]
        obstacle_range_x2 = shifted_obstacle_coords[:, 0].reshape(1, -1)
        obstacle_range_y2 = shifted_obstacle_coords[:, 1].reshape(1, -1)
        obstacle_range_x1 = obstacle_range_x1.reshape(1, -1)
        obstacle_range_y1 = obstacle_range_y1.reshape(1, -1)

        # remove the edges intersects with target area
        collide_map = self.is_traj_valid(
            [self.dest_pos],
            [obstacle_range_x1, obstacle_range_x2, obstacle_range_y1, obstacle_range_y2],
            True,
        )  # (4,E)
        collide_edge = np.sum(collide_map, axis=0).reshape(1, -1)  # (1,E)
        valid_edge_idx = collide_edge == 0
        obstacle_range_x1 = obstacle_range_x1[valid_edge_idx].reshape(1, -1)
        obstacle_range_x2 = obstacle_range_x2[valid_edge_idx].reshape(1, -1)
        obstacle_range_y1 = obstacle_range_y1[valid_edge_idx].reshape(1, -1)
        obstacle_range_y2 = obstacle_range_y2[valid_edge_idx].reshape(1, -1)

        return [obstacle_range_x1, obstacle_range_x2, obstacle_range_y1, obstacle_range_y2]

    def is_traj_valid(self, traj, obstacles_params: list, return_collide_map=False):
        VehicleBox = self.VehicleBox
        car_coords1 = np.array(VehicleBox.coords)[:4]  # (4,2)
        car_coords2 = np.array(VehicleBox.coords)[1:]  # (4,2)
        car_coords_x1 = car_coords1[:, 0].reshape(1, -1)
        car_coords_y1 = car_coords1[:, 1].reshape(1, -1)  # (1,4)
        car_coords_x2 = car_coords2[:, 0].reshape(1, -1)
        car_coords_y2 = car_coords2[:, 1].reshape(1, -1)  # (1,4)
        vxs = np.array([t[0] for t in traj])
        vys = np.array([t[1] for t in traj])
        vthetas = np.array([t[2] for t in traj])
        cos_theta = np.cos(vthetas).reshape(-1, 1)  # (T,1)
        sin_theta = np.sin(vthetas).reshape(-1, 1)
        vehicle_coords_x1 = (
            cos_theta * car_coords_x1 - sin_theta * car_coords_y1 + vxs.reshape(-1, 1)
        )  # (T,4)
        vehicle_coords_y1 = (
            sin_theta * car_coords_x1 + cos_theta * car_coords_y1 + vys.reshape(-1, 1)
        )
        vehicle_coords_x2 = (
            cos_theta * car_coords_x2 - sin_theta * car_coords_y2 + vxs.reshape(-1, 1)
        )  # (T,4)
        vehicle_coords_y2 = (
            sin_theta * car_coords_x2 + cos_theta * car_coords_y2 + vys.reshape(-1, 1)
        )
        vx1s = vehicle_coords_x1.reshape(-1, 1)
        vx2s = vehicle_coords_x2.reshape(-1, 1)
        vy1s = vehicle_coords_y1.reshape(-1, 1)
        vy2s = vehicle_coords_y2.reshape(-1, 1)
        # Line 1: the edges of vehicle box, ax + by + c = 0
        a = (vy2s - vy1s).reshape(-1, 1)  # (4*t,1)
        b = (vx1s - vx2s).reshape(-1, 1)
        c = (vy1s * vx2s - vx1s * vy2s).reshape(-1, 1)
        # print('prepare vehicle', time.time()-t1)

        x1s, x2s, y1s, y2s = obstacles_params
        # Line 2: the edges of obstacles, dx + ey + f = 0
        d = (y2s - y1s).reshape(1, -1)  # (1,E)
        e = (x1s - x2s).reshape(1, -1)
        f = (y1s * x2s - x1s * y2s).reshape(1, -1)

        # calculate the intersections
        det = a * e - b * d  # (4*t, E)
        parallel_line_pos = det == 0  # (4*t, E)
        det[parallel_line_pos] = 1  # temporarily set "1" to avoid "divided by zero"
        raw_x = (b * f - c * e) / det  # (4*t, E)
        raw_y = (c * d - a * f) / det

        collide_map_x = np.ones_like(raw_x, dtype=np.uint8)
        collide_map_y = np.ones_like(raw_x, dtype=np.uint8)
        # the false positive intersections on line L2(not on edge L2)
        tolerance_precision = 1e-4
        collide_map_x[raw_x > np.maximum(x1s, x2s) + tolerance_precision] = 0
        collide_map_x[raw_x < np.minimum(x1s, x2s) - tolerance_precision] = 0
        collide_map_y[raw_y > np.maximum(y1s, y2s) + tolerance_precision] = 0
        collide_map_y[raw_y < np.minimum(y1s, y2s) - tolerance_precision] = 0
        # the false positive intersections on line L1(not on edge L1)
        collide_map_x[raw_x > np.maximum(vx1s, vx2s) + tolerance_precision] = 0
        collide_map_x[raw_x < np.minimum(vx1s, vx2s) - tolerance_precision] = 0
        collide_map_y[raw_y > np.maximum(vy1s, vy2s) + tolerance_precision] = 0
        collide_map_y[raw_y < np.minimum(vy1s, vy2s) - tolerance_precision] = 0

        collide_map = collide_map_x * collide_map_y
        collide_map[parallel_line_pos] = 0
        if return_collide_map:
            return collide_map
        collide = np.sum(collide_map) > 0

        if collide:
            return False
        return True
