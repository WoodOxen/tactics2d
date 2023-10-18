import sys

sys.path.append(".")
sys.path.append("..")

import os
import time
from typing import Union

from gymnasium import Env, Wrapper
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from shapely.geometry import LineString
from shapely.affinity import affine_transform


from tactics2d.envs import ParkingEnv
from tactics2d.scenario.traffic_event import TrafficEvent
from tactics2d.math.interpolate import ReedsShepp
from samples.demo_ppo import DemoPPO
from samples.action_mask import ActionMask, VehicleBox, physic_model, WHEEL_BASE
from samples.rs_planner import RsPlanner
from samples.parking_config import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

action_mask = ActionMask()
dist_rear_hang = physic_model.dist_rear_hang


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


def _get_box(x, y, heading, vehicle_box=VehicleBox):
    transform_matrix = [
        np.cos(heading),
        -np.sin(heading),
        np.sin(heading),
        np.cos(heading),
        x,
        y,
    ]
    return affine_transform(vehicle_box, transform_matrix)


class ParkingWrapper(Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)

    def _preprocess_action(self, action):
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, -1, 1)
        action_space = self.env.action_space
        action = (
            action * (action_space.high - action_space.low) / 2
            + (action_space.high + action_space.low) / 2
        )

        agent_state = self.env.scenario_manager.agent.get_state()
        n_iter = 10
        for i in range(n_iter):
            collide = False
            agent_state, _ = physic_model.step(agent_state, action, 0.5 / n_iter)
            agent_pos = agent_state.x, agent_state.y, agent_state.heading
            agent_pose = _get_box(*agent_pos)
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
        lidar_info = np.clip(info["lidar"], 0, 10)
        observation = {
            "lidar": lidar_info,
            "other": np.array(
                [
                    info["diff_position"],
                    np.cos(info["diff_angle"]),
                    np.sin(info["diff_angle"]),
                    np.cos(info["diff_heading"]),
                    np.sin(info["diff_heading"]),
                    info["state"].speed,
                ]
            ),
            "action_mask": action_mask.get_steps(lidar_info),
        }

        return observation

    def reset(self, seed: int = None, options: dict = None):
        _, info = self.env.reset(seed, options)
        custom_observation = self._preprocess_observation(info)

        return custom_observation, info

    def step(self, action):
        action = self._preprocess_action(action)
        _, reward, terminated, truncated, info = self.env.step(action)
        custom_observation = self._preprocess_observation(info)

        return custom_observation, reward, terminated, truncated, info


def execute_rs_path(rs_path, agent: DemoPPO, env, obs, step_ratio=max_speed / 2):
    action_type = {"L": 1, "S": 0, "R": -1}
    action_list = []
    radius = WHEEL_BASE / np.tan(VALID_STEER[1])

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
        log_prob = agent.get_log_prob(obs, action)
        next_obs, reward, terminate, truncated, info = env.step(action)
        env.render()
        done = terminate or truncated
        total_reward += reward
        agent.push_memory((obs, action, reward, done, log_prob, next_obs))
        obs = next_obs
        if len(agent.memory) % agent.batch_size == 0:
            _ = agent.update()
        if done:
            if info["status"] == TrafficEvent.COLLISION_STATIC:
                info["status"] = TrafficEvent.COLLISION_VEHICLE
            break

    return total_reward, done, reward, info


def test_parking_env(save_path):
    render_mode = ["rgb_array", "human"][0]
    env = ParkingEnv(render_mode=render_mode, render_fps=60, max_step=200)
    env = ParkingWrapper(env)
    env.reset(42)
    agent = DemoPPO()
    rs_planner = RsPlanner(
        VehicleBox,
        radius=WHEEL_BASE / np.tan(VALID_STEER[1]),
        lidar_num=lidar_num,
        dist_rear_hang=dist_rear_hang,
        lidar_range=lidar_range,
    )
    agent.load("./PPO_parking_demo.pt", params_only=True)
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
                    info["status"] = "RS FAIL"
                    done = True
            else:
                action, log_prob = agent.choose_action(obs)  # time consume: 3ms
                next_obs, reward, terminate, truncated, info = env.step(action)
                env.render()
                done = terminate or truncated
                total_reward += reward
                agent.push_memory((obs, action, reward, done, log_prob, next_obs))
                obs = next_obs
                if len(agent.memory) % agent.batch_size == 0:
                    actor_loss, critic_loss = agent.update()
                    writer.add_scalar("actor_loss", actor_loss, i)
                    writer.add_scalar("critic_loss", critic_loss, i)

            if done:
                status_info.append(info["status"])
                if info["status"] == TrafficEvent.COMPLETED:
                    succ_record.append(1)
                else:
                    succ_record.append(0)

        writer.add_scalar("total_reward", total_reward, i)
        writer.add_scalar("success_rate", np.mean(succ_record[-100:]), i)
        writer.add_scalar("log_std1", agent.log_std.detach().cpu().numpy().reshape(-1)[0], i)
        writer.add_scalar("log_std2", agent.log_std.detach().cpu().numpy().reshape(-1)[1], i)
        writer.add_scalar("step_num", step_num, i)
        reward_list.append(total_reward)

        if i % 10 == 0 and i > 0:
            print("success rate:", np.sum(succ_record), "/", len(succ_record))
            print(agent.log_std.detach().cpu().numpy().reshape(-1))
            print("episode:%s  average reward:%s" % (i, np.mean(reward_list[-50:])))
            print(np.mean(agent.actor_loss_list[-100:]), np.mean(agent.critic_loss_list[-100:]))
            print("time_cost ,rs_dist_reward ,dist_reward ,angle_reward ,box_union_reward")
            for j in range(10):
                print(reward_list[-(10 - j)], status_info[-(10 - j)])
            print("")

        if (i + 1) % 5000 == 0:
            agent.save("%s/PPO_%s.pt" % (save_path, i), params_only=True)


if __name__ == "__main__":
    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    save_path = "./log/parking_ppo_demo/%s/" % timestamp
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    test_parking_env(save_path)
