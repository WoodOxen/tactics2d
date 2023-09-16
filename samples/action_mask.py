import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage.filters import minimum_filter1d
from shapely.geometry import LineString, Point

from tactics2d.trajectory.element.state import State
from samples.parking_config import *


class ActionMask:
    def __init__(self, VehicleBox=VehicleBox, n_iter=10) -> None:
        self.vehicle_box_base = VehicleBox
        self.n_iter = n_iter
        self.action_space = discrete_actions
        self.step_time = 0.5
        self.vehicle_boxes = self.init_vehicle_box()
        self.vehicle_lidar_base = 0
        self.lidar_num = lidar_num
        self.lidar_range = lidar_range
        self.distance_tolerance = 0.1
        self.vehicle_base = self.init_vehicle_base()

    def init_vehicle_box(
        self,
    ):
        VehicleBox = self.vehicle_box_base
        car_coords = np.array(VehicleBox.coords)[:4]  # (4,2)
        car_coords_x = car_coords[:, 0].reshape(-1)
        car_coords_y = car_coords[:, 1].reshape(-1)  # (4)
        vehicle_boxes = []
        for action in self.action_space:
            state = State(0, 0, 0, 0, 0, 0, 0)
            for _ in range(self.n_iter):
                state, _ = physic_model.step(state, action, self.step_time / self.n_iter)
                x, y, heading = state.x, state.y, state.heading
                car_x_ = car_coords_x * np.cos(heading) - car_coords_y * np.sin(heading) + x  # (4)
                car_y_ = car_coords_x * np.sin(heading) + car_coords_y * np.cos(heading) + y
                vehicle_coords = np.concatenate(
                    (np.expand_dims(car_x_, axis=-1), np.expand_dims(car_y_, axis=-1)), axis=-1
                )  # (4, 2)
                vehicle_boxes.append(vehicle_coords)
        vehicle_boxes = (
            np.array(vehicle_boxes)
            .reshape(len(self.action_space), self.n_iter, 4, 2)
            .transpose(1, 0, 2, 3)
        )  # (10, 42, 4, 2)
        return vehicle_boxes

    def init_vehicle_base(
        self,
    ):
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
            distance = l.intersection(VehicleBox).distance(ORIGIN)
            lidar_base.append(distance)
        return np.array(lidar_base)

    def get_steps(self, raw_lidar_obs: np.ndarray):
        lidar_obs = np.maximum(self.vehicle_base, raw_lidar_obs - self.distance_tolerance)

        car_coords = np.array(self.vehicle_boxes).reshape(-1, 4, 2)  # (10, 42, 4, 2) -> (420, 4, 2)
        car_edge_x1 = car_coords[:, :, 0].reshape(-1, 4, 1)
        car_edge_y1 = car_coords[:, :, 1].reshape(-1, 4, 1)
        shifted_car_coords = car_coords.copy()
        shifted_car_coords[:, :-1] = car_coords[:, 1:]
        shifted_car_coords[:, -1] = car_coords[:, 0]
        car_edge_x2 = shifted_car_coords[:, :, 0].reshape(-1, 4, 1)
        car_edge_y2 = shifted_car_coords[:, :, 1].reshape(-1, 4, 1)  # (420, 4, 1)
        # Line 1: the edges of vehicle box, ax + by + c = 0
        a = car_edge_y2 - car_edge_y1  # (420, 4, 1)
        b = car_edge_x1 - car_edge_x2
        c = car_edge_y1 * car_edge_x2 - car_edge_x1 * car_edge_y2

        # lidar_obs = np.clip(lidar_obs, 0, self.lidar_range) + self.vehicle_lidar_base
        lidar_num = self.lidar_num
        angle_vec = np.arange(lidar_num) * np.pi / lidar_num * 2
        obstacle_range_x1 = np.cos(angle_vec) * lidar_obs  # (N,)
        obstacle_range_y1 = np.sin(angle_vec) * lidar_obs
        obstacle_range_coords = np.concatenate(
            (np.expand_dims(obstacle_range_x1, 1), np.expand_dims(obstacle_range_y1, 1)), axis=1
        )  # (N, 2)
        shifted_obstacle_coords = obstacle_range_coords.copy()
        shifted_obstacle_coords[:-1] = obstacle_range_coords[1:]
        shifted_obstacle_coords[-1] = obstacle_range_coords[0]
        obstacle_range_x2 = shifted_obstacle_coords[:, 0].reshape(1, 1, -1)
        obstacle_range_y2 = shifted_obstacle_coords[:, 1].reshape(1, 1, -1)
        obstacle_range_x1 = obstacle_range_x1.reshape(1, 1, -1)
        obstacle_range_y1 = obstacle_range_y1.reshape(1, 1, -1)
        # Line 2: the edges of obstacles, dx + ey + f = 0
        d = obstacle_range_y2 - obstacle_range_y1  # (1, 1, N)
        e = obstacle_range_x1 - obstacle_range_x2
        f = obstacle_range_y1 * obstacle_range_x2 - obstacle_range_x1 * obstacle_range_y2

        # calculate the intersections
        det = a * e - b * d  # (420, 4, N)
        parallel_line_pos = det == 0  # (420, 4, N)
        det[parallel_line_pos] = 1  # temporarily set "1" to avoid "divided by zero"
        raw_x = (b * f - c * e) / det  # (420, 4, N)
        raw_y = (c * d - a * f) / det

        collide_map_x = np.ones_like(raw_x, dtype=np.uint8)
        collide_map_y = np.ones_like(raw_x, dtype=np.uint8)
        # the false positive intersections on line L2(not on edge L2)
        tolerance_precision = 1e-4
        collide_map_x[
            raw_x > np.maximum(obstacle_range_x1, obstacle_range_x2) + tolerance_precision
        ] = 0
        collide_map_x[
            raw_x < np.minimum(obstacle_range_x1, obstacle_range_x2) - tolerance_precision
        ] = 0
        collide_map_y[
            raw_y > np.maximum(obstacle_range_y1, obstacle_range_y2) + tolerance_precision
        ] = 0
        collide_map_y[
            raw_y < np.minimum(obstacle_range_y1, obstacle_range_y2) - tolerance_precision
        ] = 0
        # the false positive intersections on line L1(not on edge L1)
        collide_map_x[raw_x > np.maximum(car_edge_x1, car_edge_x2) + tolerance_precision] = 0
        collide_map_x[raw_x < np.minimum(car_edge_x1, car_edge_x2) - tolerance_precision] = 0
        collide_map_y[raw_y > np.maximum(car_edge_y1, car_edge_y2) + tolerance_precision] = 0
        collide_map_y[raw_y < np.minimum(car_edge_y1, car_edge_y2) - tolerance_precision] = 0

        collide_map = collide_map_x * collide_map_y  # (420, 4, N)
        collide_map[parallel_line_pos] = 0
        collides = np.sum(collide_map, axis=(1, 2)).reshape(
            self.n_iter, len(self.action_space)
        )  # (420,) -> (10, 42)
        collides[collides != 0] = 1
        collide_free_binary = np.sum(collides, axis=0) == 0  # (42)
        step_len = np.argmax(collides, axis=0)
        step_len[collide_free_binary.astype(bool)] = self.n_iter

        action_mask = self.post_process(step_len)
        if np.sum(action_mask) == 0:
            return np.clip(action_mask, 0.01, 1)
        return action_mask

    def post_process(self, step_len: np.ndarray):
        kernel = 5
        forward_step_len = step_len[: len(step_len) // 2]
        backward_step_len = step_len[len(step_len) // 2 :]
        forward_step_len[0] -= 1
        forward_step_len[-1] -= 1
        backward_step_len[0] -= 1
        backward_step_len[-1] -= 1
        forward_step_len_ = minimum_filter1d(forward_step_len, kernel)
        backward_step_len_ = minimum_filter1d(backward_step_len, kernel)
        return np.clip(np.concatenate((forward_step_len_, backward_step_len_)), 0, 10) / 10

    def choose_action(self, action_mean, action_std, action_mask):
        if isinstance(action_mean, torch.Tensor):
            action_mean = action_mean.cpu().numpy()
            action_std = action_std.cpu().numpy()
        if isinstance(action_mask, torch.Tensor):
            action_mask = action_mask.cpu().numpy()
        if len(action_mean.shape) == 2:
            action_mean = action_mean.squeeze(0)
            action_std = action_std.squeeze(0)
        if len(action_mask.shape) == 2:
            action_mask = action_mask.squeeze(0)

        def calculate_probability(mean, std, values):
            z_scores = (values - mean) / std
            log_probabilities = -0.5 * z_scores**2 - np.log((np.sqrt(2 * np.pi) * std))
            return np.sum(np.clip(log_probabilities, -10, 10), axis=1)

        possible_actions = np.array(self.action_space)
        # deal the scaling
        action_mean[1] = 1 if action_mean[1] > 0 else -1
        scale_steer = VALID_STEER[1]
        scale_speed = 1
        possible_actions = possible_actions / np.array([scale_steer, scale_speed])
        prob = calculate_probability(action_mean, action_std, possible_actions)
        exp_prob = np.exp(prob) * action_mask
        prob_softmax = exp_prob / np.sum(exp_prob)
        actions = np.arange(len(possible_actions))
        action_chosen = np.random.choice(actions, p=prob_softmax)

        return possible_actions[action_chosen]
