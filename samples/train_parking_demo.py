import sys

sys.path.append(".")
sys.path.append("..")

import os
import time
from typing import Union

import gymnasium as gym
from gymnasium import Wrapper
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from shapely.geometry import LineString
from shapely.affinity import affine_transform

from tactics2d.envs import ParkingEnv
from tactics2d.scenario.traffic_event import TrafficEvent
from tactics2d.math.interpolate import ReedsShepp
from samples.demo_ppo import DemoPPO
from samples.action_mask import ActionMask
from samples.rs_planner import RsPlanner

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MAX_SPEED = 2.0
LIDAR_RANGE = 20.0
LIDAR_LINE = 120


class PathPlanner:
    def __init__(self, radius, threshold_distance):
        self.interpolator = ReedsShepp(radius)
        self.threshold_distance = threshold_distance

    def validate_path(self, path, obstacles):
        collide = False
        center_line = LineString(path.curve)
        bounding_lines = []

        for line in bounding_lines:
            if not line.intersection(obstacles).is_empty:
                collide = True
                break

        return collide

    def get_path(self, start_point, start_heading, end_point, end_heading):
        relative_distance = np.linalg.norm(end_point - start_point)
        if relative_distance < self.threshold_distance:
            return None

        candidate_paths = self.interpolator.get_all_path(
            start_point, start_heading, end_point, end_heading
        )
        if len(candidate_paths) == 0:
            return None

        candidate_paths.sort(key=lambda x: x.length)
        min_length = candidate_paths[0].length
        for path in candidate_paths:
            if path.length > min_length * 2:
                break

            path.get_curve_line(start_point, start_heading, self.radius, 0.1)
            if self.validate_path(path):
                return path

        return None


def _get_box(state, vehicle_box):
    x = state.x
    y = state.y
    heading = state.heading
    transform_matrix = [np.cos(heading), -np.sin(heading), np.sin(heading), np.cos(heading), x, y]
    return affine_transform(vehicle_box, transform_matrix)


class ParkingWrapper(Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_mask = ActionMask(self.env.scenario_manager.agent)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(120 + 42 + 6, 1),
            dtype=np.float32,
        )

    def _preprocess_action(self, action):
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, -1, 1)
        action_space = self.env.action_space
        action = (
            action * (action_space.high - action_space.low) / 2
            + (action_space.high + action_space.low) / 2
        )

        # force the agent to avoid collision by split the action to smaller time scale
        agent_state = self.env.scenario_manager.agent.get_state()
        n_iter = 10
        for i in range(n_iter):
            collide = False
            agent_state, _ = self.env.scenario_manager.agent.physics_model.step(
                agent_state, action, 0.5 / n_iter
            )
            agent_pose = _get_box(agent_state, self.env.scenario_manager.agent.bbox)
            for _, area in self.env.scenario_manager.map_.areas.items():
                if area.type_ == "obstacle" and agent_pose.intersects(area.geometry):
                    collide = True
            if collide:
                break
        if not collide:
            i += 1
        action[1] = action[1] * i / 10
        return action

    def _preprocess_observation(self, info):
        lidar_info = np.clip(info["lidar"], 0, self.env.scenario_manager.lidar_range)
        lidar_feature = lidar_info
        action_mask_feature = self.action_mask.get_steps(lidar_info)
        other_feature = np.array(
            [
                info["diff_position"],
                np.cos(info["diff_angle"]),
                np.sin(info["diff_angle"]),
                np.cos(info["diff_heading"]),
                np.sin(info["diff_heading"]),
                info["state"].speed,
            ]
        )

        observation = np.concatenate([lidar_feature, action_mask_feature, other_feature])

        return observation, action_mask_feature

    def reset(self, seed: int = None, options: dict = None):
        _, info = self.env.reset(seed, options)
        custom_observation, action_mask_feature = self._preprocess_observation(info)
        info["action_mask"] = action_mask_feature

        return custom_observation, info

    def step(self, action):
        action = self._preprocess_action(action)
        _, reward, terminated, truncated, info = self.env.step(action)
        custom_observation, action_mask_feature = self._preprocess_observation(info)
        info["action_mask"] = action_mask_feature

        return custom_observation, reward, terminated, truncated, info


def execute_rs_path(rs_path, agent: DemoPPO, env, obs, step_ratio=MAX_SPEED / 2):
    action_type = {"L": 1, "S": 0, "R": -1}
    action_list = []
    radius = env.scenario_manager.agent.wheel_base / np.tan(
        env.scenario_manager.agent.steer_range[1]
    )

    for i in range(len(rs_path.actions)):
        steer = action_type[rs_path.actions[i]]
        step_len = rs_path.signs[i] * rs_path.segments[i] / step_ratio * radius
        action_list.append([steer, step_len])

    # divide the action
    filtered_actions = []
    for action in action_list:
        action[0] *= 1
        if abs(action[1]) < 1 and abs(action[1]) > 1e-3:
            filtered_actions.append(action)
        elif action[1] > 1:
            while action[1] > 1:
                filtered_actions.append([action[0], 1])
                action[1] -= 1
            if abs(action[1]) > 1e-3:
                filtered_actions.append(action)
        elif action[1] < -1:
            while action[1] < -1:
                filtered_actions.append([action[0], -1])
                action[1] += 1
            if abs(action[1]) > 1e-3:
                filtered_actions.append(action)

    # step actions
    total_reward = 0
    for action in filtered_actions:
        log_prob, value = agent.get_log_prob(obs, action)
        next_obs, reward, terminate, truncated, info = env.step(action)
        observations = ([next_obs], [reward], [terminate], [truncated], [info])
        env.render()
        done = terminate or truncated
        total_reward += reward
        agent.push((observations, [obs], [action], [log_prob], [value]))
        obs = next_obs
        agent.train()
        if done:
            if info["status"] == TrafficEvent.COLLISION_STATIC:
                info["status"] = TrafficEvent.COLLISION_VEHICLE
            break

    return total_reward, done, reward, info


def test_parking_env(save_path):
    render_mode = ["rgb_array", "human"][0]
    env = ParkingEnv(render_mode=render_mode, render_fps=60, max_step=200)
    env = ParkingWrapper(env)
    env = gym.wrappers.NormalizeObservation(env)
    env.reset()
    agent = DemoPPO()
    rs_planner = RsPlanner(env.scenario_manager.agent)
    # agent.load("./PPO_parking_demo.pt", params_only=True)
    writer = SummaryWriter(save_path)

    reward_list = []
    succ_record = []
    status_info = []
    print("start train!")
    for i in range(100000):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step_num = 0
        while not done:
            step_num += 1
            rs_path = rs_planner.get_rs_path(info)
            if rs_path is not None:
                total_reward, done, reward, info = execute_rs_path(
                    rs_path,
                    agent,
                    env,
                    obs,
                )
                if not done:
                    info["status"] = TrafficEvent.FAILED
                    done = True
            else:
                action, log_prob, value = agent.choose_action(obs, info)  # time consume: 3ms
                next_obs, reward, terminate, truncated, info = env.step(action)
                observations = ([next_obs], [reward], [terminate], [truncated], [info])
                env.render()
                done = terminate or truncated
                total_reward += reward
                agent.push((observations, [obs], [action], [log_prob], [value]))
                obs = next_obs
                agent.train()

            if done:
                status_info.append(info["status"])
                if info["status"] == TrafficEvent.COMPLETED:
                    succ_record.append(1)
                else:
                    succ_record.append(0)

        writer.add_scalar("total_reward", total_reward, i)
        writer.add_scalar("success_rate", np.mean(succ_record[-100:]), i)
        writer.add_scalar(
            "log_std1", agent.actor_net.log_std.detach().cpu().numpy().reshape(-1)[0], i
        )
        writer.add_scalar(
            "log_std2", agent.actor_net.log_std.detach().cpu().numpy().reshape(-1)[1], i
        )
        writer.add_scalar("step_num", step_num, i)
        reward_list.append(total_reward)

        if i % 10 == 0 and i > 0:
            print("success rate:", np.sum(succ_record), "/", len(succ_record))
            print(agent.actor_net.log_std.detach().cpu().numpy().reshape(-1))
            print("episode:%s  average reward:%s" % (i, np.mean(reward_list[-50:])))
            print("time_cost ,rs_dist_reward ,dist_reward ,angle_reward ,box_union_reward")
            for j in range(10):
                print(reward_list[-(10 - j)], status_info[-(10 - j)])
            print("")


if __name__ == "__main__":
    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    save_path = "./log/parking_ppo_demo/%s/" % timestamp
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    test_parking_env(save_path)
