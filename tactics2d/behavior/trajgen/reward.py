##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: reward.py
# @Description: The reward functions of TrajGen. Original code from https://github.com/gaoyinfeng/TrajGen/blob/main/python/interaction_gym/reward.py.
# @Author: Tactics2D Team
# @Version: 0.1.9


import numpy as np


def calculate_trajectory_pos_reward(observation_dict, ego_id):
    ego_trajectory_distance = observation_dict["trajectory_distance"][ego_id][0]

    trajectory_pos_reward = 1 - 0.2 * ego_trajectory_distance

    return trajectory_pos_reward


def calculate_speed_reward(observation_dict, ego_id, control_steering=True):
    # ego_current_target_speed = observation_dict['target_speed'][ego_id][0]
    ego_current_target_speed = 10

    if control_steering:
        ego_speed = observation_dict["lane_observation"][ego_id][2]  # along route direction
    else:
        ego_speed = observation_dict["current_speed"][ego_id][0]

    l_3 = 0.5
    if ego_current_target_speed != 0:
        if ego_speed <= ego_current_target_speed:
            speed_reward = ego_speed / ego_current_target_speed
        else:
            speed_reward = 1 - ((ego_speed - ego_current_target_speed) / ego_current_target_speed)
    else:
        speed_reward = -ego_speed

    speed_reward *= l_3

    return speed_reward


def calculate_lane_keeping_reward(observation_dict, ego_id):
    ego_x_in_point_axis = observation_dict["lane_observation"][ego_id][0]

    ego_speed_along_lane = observation_dict["lane_observation"][ego_id][2]
    current_heading_error = observation_dict["lane_observation"][ego_id][3]
    future_heading_errors = observation_dict["lane_observation"][ego_id][4:]

    l_1 = 0.75
    l_2 = 0.75
    lk_reward_current = ego_speed_along_lane * (
        np.cos(current_heading_error) - l_1 * (np.sin(abs(current_heading_error)))
    ) - l_2 * (abs(ego_x_in_point_axis))
    # lk_reward_future = np.sum(np.cos(future_heading_errors))*0.25 - l_1*np.sum(np.sin(np.abs(future_heading_errors)))*0.25

    lk_reward = lk_reward_current
    return lk_reward


def calculate_steer_reward(previous_steer, current_steer):

    l_4 = 1
    # steer_reward = -l_4*abs(current_steer - previous_steer)
    steer_reward = -l_4 * abs(current_steer)

    return steer_reward
