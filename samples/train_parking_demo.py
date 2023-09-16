import sys

sys.path.append(".")
sys.path.append("..")

import os
import random
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from shapely.affinity import affine_transform

from tactics2d.envs import ParkingEnv
from tactics2d.scenario import TrafficEvent
from samples.demo_ppo import DemoPPO
from samples.action_mask import ActionMask, VehicleBox, physic_model, WHEEL_BASE
from samples.rs_planner import RsPlanner
from samples.parking_config import *

action_mask = ActionMask()
dist_rear_hang = physic_model.dist_rear_hang


class RewardShaping:
    def __init__(self) -> None:
        self.prev_max_iou = 0
        self.reward_scale = 0.1
        self.curr_iou = 0

    def reward_shaping(self, info):
        rewards = info["rewards"]
        self.curr_iou += rewards["iou_reward"]
        r_iou = max(0, self.curr_iou - self.prev_max_iou)
        if rewards["collision_penalty"] < -1e-4:
            reward = -50
        elif rewards["time_exceeded_penalty"] < -1e-4:
            reward = -1
        elif rewards["outside_map_penalty"] < -1e-4:
            reward = -50
        elif rewards["complete_reward"] > 1e-4:
            reward = 50
        else:
            reward = rewards["time_penalty"] + r_iou * 10  # + rewards['distance_reward']*10
        reward *= self.reward_scale
        self.prev_max_iou = max(self.prev_max_iou, self.curr_iou)
        return reward

    def reset(
        self,
    ):
        self.prev_max_iou = 0
        self.curr_iou = 0


reward_shaping = RewardShaping()


def save_step(env: ParkingEnv, action):
    action = resize_action(action, env.action_space)

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

    agent_state = env.scenario_manager.agent.get_state()
    env_map = env.scenario_manager.map_
    n_iter = 10
    for i in range(n_iter):
        collide = False
        agent_state, _ = physic_model.step(agent_state, action, 0.5 / n_iter)  # TODO step time
        agent_pos = agent_state.x, agent_state.y, agent_state.heading
        agent_pose = _get_box(*agent_pos)
        for _, area in env_map.areas.items():
            if area.type_ == "obstacle" and agent_pose.intersects(area.geometry):
                collide = True
        if collide:
            break
    if not collide:
        i += 1
    action[1] = action[1] * i / 10
    returns = env.step(action)
    return returns


def execute_rs_path(rs_path, agent: DemoPPO, env, obs, step_ratio=max_speed / 2):
    action_type = {"L": 1, "S": 0, "R": -1}
    action_list = []
    for i in range(len(rs_path.ctypes)):
        steer = action_type[rs_path.ctypes[i]]
        step_len = rs_path.lengths[i] / step_ratio
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
        next_obs, reward, terminate, truncated, info = save_step(env, action)
        reward = reward_shaping.reward_shaping(info)
        env.render()
        done = terminate or truncated
        total_reward += reward
        next_obs = preprocess_obs(info)
        agent.push_memory((obs, action, reward, done, log_prob, next_obs))
        obs = next_obs
        if len(agent.memory) % agent.batch_size == 0:
            _ = agent.update()
        if done:
            if info["status"] == TrafficEvent.COLLISION_STATIC:
                info["status"] = TrafficEvent.COLLISION_VEHICLE
            break

    return total_reward, done, info


def preprocess_obs(info):
    lidar_obs = info["lidar"]
    lidar_obs = np.clip(lidar_obs, 0.0, lidar_range)

    def _move_center(x, y, heading):
        x -= dist_rear_hang * np.cos(heading)
        y -= dist_rear_hang * np.sin(heading)
        return x, y, heading

    # process target pose
    dest_coords = np.mean(np.array(info["target_area"]), axis=0)
    dest_heading = info["target_heading"]
    dest_pos = _move_center(dest_coords[0], dest_coords[1], dest_heading)
    ego_pos = _move_center(info["position_x"], info["position_y"], info["heading"])
    rel_distance = np.sqrt((dest_pos[0] - ego_pos[0]) ** 2 + (dest_pos[1] - ego_pos[1]) ** 2)
    rel_angle = np.arctan2(dest_pos[1] - ego_pos[1], dest_pos[0] - ego_pos[0]) - ego_pos[2]
    rel_dest_heading = dest_pos[2] - ego_pos[2]
    tgt_repr = (
        rel_distance,
        np.cos(rel_angle),
        np.sin(rel_angle),
        np.cos(rel_dest_heading),
        np.cos(rel_dest_heading),
    )
    speed = info["speed"]
    other_info_repr = np.array(tgt_repr + (speed,))
    action_mask_info = action_mask.get_steps(lidar_obs)
    obs = {"lidar": lidar_obs, "other": other_info_repr, "action_mask": action_mask_info}
    return obs


def resize_action(
    action: np.ndarray,
    action_space,
    raw_action_range=(-1, 1),
    explore: bool = True,
    epsilon: float = 0.0,
):
    action = np.array(action, dtype=np.float32)
    action = np.clip(action, *raw_action_range)
    action = (
        action * (action_space.high - action_space.low) / 2
        + (action_space.high + action_space.low) / 2
    )
    if explore and np.random.random() < epsilon:
        action = action_space.sample()
    return action


def test_parking_env(save_path):
    render_mode = ["rgb_array", "human"][0]
    env = ParkingEnv(render_mode=render_mode, render_fps=60, max_step=200)
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
    reward_info_list = []
    succ_record = []
    status_info = []
    print("start train!")
    for i in range(100000):
        # print(i)
        reward_shaping.reset()
        _, info = env.reset()
        obs = preprocess_obs(info)
        done = False
        total_reward = 0
        step_num = 0
        reward_info = []
        while not done:
            step_num += 1
            rs_path = rs_planner.get_rs_path(info)
            if rs_path is not None:
                total_reward, done, info = execute_rs_path(
                    rs_path,
                    agent,
                    env,
                    obs,
                )
                reward_info.append(list(info["rewards"].values()))
                if not done:
                    info["status"] = "RS FAIL"
                    done = True
            else:
                action, log_prob = agent.choose_action(obs)  # time consume: 3ms
                _, _, terminate, truncated, info = save_step(env, action)
                env.render()
                reward = reward_shaping.reward_shaping(info)
                done = terminate or truncated
                next_obs = preprocess_obs(info)
                reward_info.append(list(info["rewards"].values()))
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
        reward_info = np.sum(np.array(reward_info), axis=0)
        reward_info = np.round(reward_info, 2)
        reward_info_list.append(list(reward_info))

        if i % 10 == 0 and i > 0:
            print("success rate:", np.sum(succ_record), "/", len(succ_record))
            print(agent.log_std.detach().cpu().numpy().reshape(-1))
            print("episode:%s  average reward:%s" % (i, np.mean(reward_list[-50:])))
            print(np.mean(agent.actor_loss_list[-100:]), np.mean(agent.critic_loss_list[-100:]))
            print("time_cost ,rs_dist_reward ,dist_reward ,angle_reward ,box_union_reward")
            for j in range(10):
                print(reward_list[-(10 - j)], reward_info_list[-(10 - j)], status_info[-(10 - j)])
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
