#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File: trainer_parkinglot.py
# @Description: This script gives an example on how to train a PPO model in tactic2d's parking lot traffic scenario.
# @Time: 2023/11/10
# @Author: Mingyang Jiang, Yueyuan Li

import sys

sys.path.append(".")
sys.path.append("./rllib")
sys.path.append("..")

import os
import time
import argparse
from collections import deque

import gymnasium as gym
from gymnasium import Wrapper
import numpy as np
import torch
from torch.distributions import Normal, Categorical
from torch.utils.tensorboard import SummaryWriter
from shapely.geometry import LineString
from shapely.affinity import affine_transform

from rllib.algorithms.ppo import *
from tactics2d.envs import ParkingEnv
from tactics2d.traffic.violation_detection import TrafficEvent
from tactics2d.math.interpolate import ReedsShepp
from samples.action_mask import ActionMask
from samples.rs_planner import RsPlanner

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

LIDAR_RANGE = 20.0
LIDAR_LINE = 120

OBS_SHAPE = {"state": (120 + 42 + 6,)}


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
            low=-np.inf, high=np.inf, shape=(120 + 42 + 6, 1), dtype=np.float32
        )
        self.scaled_actions = np.concatenate(
            [np.linspace([1, 2], [-1, 2], 21), np.linspace([1, -2], [-1, -2], 21)]
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


def execute_path(path, agent, env, obs, speed_step=1):
    action_type = {"L": 1, "S": 0, "R": -1}
    radius = env.scenario_manager.agent.wheel_base / np.tan(
        env.scenario_manager.agent.steer_range[1]
    )

    actions = []
    for i in range(len(path.actions)):
        steer = action_type[path.actions[i]]
        speed = path.signs[i] * path.segments[i] / speed_step * radius
        if 1e-3 < abs(speed) <= 1:
            actions.append([steer, speed])
        elif speed > speed_step:
            while speed > speed_step:
                actions.append([steer, speed_step])
                speed -= speed_step
            if abs(speed) > 1e-3:
                actions.append([steer, speed])
        elif speed < -speed_step:
            while speed < -speed_step:
                actions.append([steer, -speed_step])
                speed += speed_step
            if abs(speed) > 1e-3:
                actions.append([steer, speed])

    # step actions
    total_reward = 0
    for action in actions:
        log_prob, value = agent.evaluate_action(obs, action)
        next_obs, reward, terminate, truncated, info = env.step(action)
        env.render()
        done = terminate or truncated
        total_reward += reward
        observations = [[next_obs], [reward], [terminate], [truncated], [info]]
        agent.push([observations, [obs], [action], [log_prob], [value]])
        obs = next_obs
        agent.train()
        if done:
            break

    return total_reward, done, info


class ParkingActor(PPOActor):
    def get_dist(self, state):
        policy_dist = self.forward(state)
        mean = torch.clamp(policy_dist, -1, 1)
        std = self.log_std.expand_as(mean).exp()
        dist = Normal(mean, std)

        return dist

    def action(self, state, scaled_actions, action_mask):
        dist = self.get_dist(state)
        exp_probs = dist.log_prob(scaled_actions).exp()
        exp_prob_masked = exp_probs[:, 0] * exp_probs[:, 1] * action_mask
        prob_softmax = exp_prob_masked / torch.sum(exp_prob_masked)
        action_id = Categorical(prob_softmax).sample()
        action = scaled_actions[action_id]
        action = torch.clamp(action, -1, 1)
        log_prob = dist.log_prob(action)

        action = action.detach().cpu().numpy()
        log_prob = log_prob.detach().cpu().numpy()

        return action, log_prob


class ParkingAgent(PPO):
    def get_action(self, states, scaled_actions, action_mask):
        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(states).to(self.device)

        if not isinstance(scaled_actions, torch.Tensor):
            scaled_actions = torch.FloatTensor(scaled_actions).to(self.device)

        if not isinstance(action_mask, torch.Tensor):
            action_mask = torch.FloatTensor(action_mask).to(self.device)

        action, log_prob = self.actor_net.action(states, scaled_actions, action_mask)
        value = self.critic_net(states)
        value = value.detach().cpu().numpy().flatten()

        return action, log_prob, value

    def evaluate_action(self, states, actions: np.ndarray):
        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(states).to(self.device)

        if not isinstance(actions, torch.Tensor):
            actions = torch.FloatTensor(actions).to(self.device)

        log_prob, _ = self.actor_net.evaluate(states, actions)
        value = self.critic_net(states)

        log_prob = log_prob.detach().cpu().numpy().flatten()
        value = value.detach().cpu().numpy().flatten()

        return log_prob, value


def trainer(args):
    env = ParkingEnv(
        render_mode=args.render_mode, render_fps=args.render_fps, max_step=args.max_step
    )
    env = ParkingWrapper(env)
    env = gym.wrappers.NormalizeObservation(env)
    scaled_actions = torch.FloatTensor(env.scaled_actions).to(device)

    agent_config = PPOConfig(
        {
            "debug": True,
            "state_space": env.observation_space,
            "action_space": env.action_space,
            "gamma": 0.95,
            "lr": 2e-6,
            "actor_net": ParkingActor,
            "actor_kwargs": {
                "state_dim": env.observation_space.shape[0],
                "action_dim": env.action_space.shape[0],
                "hidden_size": 256,
                "continuous": True,
            },
            "critic_kwargs": {"state_dim": env.observation_space.shape[0], "hidden_size": 256},
            "horizon": 8192,
            "batch_size": 32,
            "adam_epsilon": 1e-8,
        }
    )
    agent = ParkingAgent(agent_config, device)

    path_planner = RsPlanner(env.scenario_manager.agent)

    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    log_path = os.path.join(args.log_path, timestamp)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)

    reward_list = deque(maxlen=100)
    success_list = deque(maxlen=100)
    loss_list = deque(maxlen=100)
    status_info = deque(maxlen=100)

    step_num = int(1e6)
    step_cnt = 0
    episode_cnt = 0

    print("start train!")

    while step_cnt < step_num:
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            step_cnt += 1
            rs_path = path_planner.get_rs_path(info)
            # If the RS planner can find a path, the agent will execute the path. Otherwise, the agent will use the PPO policy to choose an action.
            if rs_path is not None:
                total_reward, done, info = execute_path(rs_path, agent, env, state)
                if not done:
                    info["status"] = TrafficEvent.FAILED
                    done = True
            else:
                action, log_prob, value = agent.get_action(
                    state, scaled_actions, info["action_mask"]
                )
                next_state, reward, terminate, truncated, info = env.step(action)
                env.render()
                done = terminate or truncated
                total_reward += reward
                observations = [[next_state], [reward], [terminate], [truncated], [info]]
                agent.push([observations, [state], [action], [log_prob], [value]])
                state = next_state
                loss = agent.train()
                if not loss is None:
                    loss_list.append(loss)

        status_info.append(info["status"])
        success_list.append(int(info["status"] == TrafficEvent.COMPLETED))
        reward_list.append(total_reward)

        episode_cnt += 1

        if episode_cnt % 10 == 0:
            print(
                "episode: %d, total step: %d, average reward: %s, success rate: %s"
                % (episode_cnt, step_cnt, np.mean(reward_list), np.mean(success_list))
            )
            print("last 10 episode:")
            for i in range(10):
                print(reward_list[-(10 - i)], status_info[-(10 - i)])
            print("")

            writer.add_scalar("average_reward", np.mean(reward_list), step_cnt)
            writer.add_scalar("average_loss", np.mean(loss_list), step_cnt)
            writer.add_scalar("success_rate", np.mean(success_list), step_cnt)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--render-mode", type=str, choices=["rgb_array", "human"], default="rgb_array"
    )
    arg_parser.add_argument("--render-fps", type=int, default=60)
    arg_parser.add_argument("--max-step", type=int, default=200)
    arg_parser.add_argument("--log-path", type=str, default="./logs")
    args = arg_parser.parse_args()
    trainer(args)
