import sys

sys.path.append(".")
sys.path.append("..")

import os
import random
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tactics2d.envs import ParkingEnv
from tactics2d.scenario import TrafficEvent
from samples.demo_ppo import DemoPPO

def preprocess_obs(info):
    # process lidar
    # print(info['velocity'], info['speed'])
    lidar_obs = info['lidar']
    lidar_obs = np.clip(lidar_obs, 0.0, 10.0)/10.0
    # process target pose
    half_wheel_base = 1.319
    dest_coords = np.mean(np.array(info['target_area']), axis=0)
    dest_heading = info['target_heading']
    # dest_pos = (dest_coords[0]-half_wheel_base*np.cos(dest_heading), \
    #             dest_coords[1]-half_wheel_base*np.sin(dest_heading), dest_heading)
    dest_pos = (dest_coords[0], dest_coords[1], dest_heading)
    ego_pos = (info['position_x'], info['position_y'], info['heading'])
    rel_distance = np.sqrt((dest_pos[0]-ego_pos[0])**2 + (dest_pos[1]-ego_pos[1])**2)
    rel_angle = np.arctan2(dest_pos[1]-ego_pos[1], dest_pos[0]-ego_pos[0]) - ego_pos[2]
    rel_dest_heading = dest_pos[2] - ego_pos[2]
    tgt_repr = (rel_distance, np.cos(rel_angle), np.sin(rel_angle),\
        np.cos(rel_dest_heading), np.cos(rel_dest_heading),)
    speed = info['speed']
    other_info_repr = np.array(tgt_repr + (speed,))
    obs = {'lidar':lidar_obs, 'other':other_info_repr}
    return obs

def resize_action(action:np.ndarray, action_space, raw_action_range=(-1,1), explore:bool=True, epsilon:float=0.0):
    action = np.clip(action, *raw_action_range)
    action = action * (action_space.high - action_space.low) / 2 + (action_space.high + action_space.low) / 2
    if explore and np.random.random() < epsilon:
        action = action_space.sample()
    return action

def reward_shaping(info):
    rewards = info['rewards']
    reward = (rewards['time_exceeded_penalty']*10 + rewards['collision_penalty']*50 +\
              rewards['outside_map_penalty']*50 + rewards['complete_reward']*50 +\
                rewards['time_penalty'] + rewards['iou_reward']*50 + rewards['distance_reward']*10)*0.1
    return reward

def test_parking_env(save_path):
    render_mode = ["rgb_array", "human"][0]
    env = ParkingEnv(render_mode=render_mode, render_fps=60, max_step=200)
    env.reset(42)
    agent = DemoPPO()
    # agent.load('./PPO_10000.pt',params_only=True)
    writer = SummaryWriter(save_path)

    reward_list = []
    reward_info_list = []
    case_id_list = []
    succ_record = []
    print('start train!')
    for i in range(100000):
        # print(i)
        _, info = env.reset()
        obs = preprocess_obs(info)
        done = False
        total_reward = 0
        step_num = 0
        reward_info = []
        while not done:
            step_num += 1
            # action, log_prob = agent.get_action(obs) # time consume: 3ms
            action, log_prob = agent.choose_action(obs) # time consume: 3ms
            # action = env.action_space.sample()
            # action = np.array([1.0, 1], dtype=np.float32)
            # t = time.time()
            # print(action)
            action = resize_action(action, env.action_space)
            # t = time.time()
            _, _, terminate, truncated, info = env.step(action)
            env.render()
            reward = reward_shaping(info)
            done = terminate or truncated
            # if done:
            #     print("#"*10)
            #     print("DONE!!!!")
            #     print(info['status'])
            #     time.sleep(2)
            # print(time.time()-t)
            next_obs = preprocess_obs(info)
            # print(time.time()-t)
            reward_info.append(list(info['rewards'].values()))
            total_reward += reward
            agent.push_memory((obs, action, reward, done, log_prob, next_obs))
            obs = next_obs
            if len(agent.memory) % agent.batch_size == 0:
                # print('update agent!')
                actor_loss, critic_loss = agent.update()
                writer.add_scalar("actor_loss", actor_loss, i)
                writer.add_scalar("critic_loss", critic_loss, i)
            
            # if info['path_to_dest'] is not None:
            #     succ_record.append(1)
            #     rs_reward = execute_rs_path(info['path_to_dest'], agent, env, obs, writer)
            #     total_reward += rs_reward
            #     break
            if done:
                if info['status']==TrafficEvent.COMPLETED:
                    succ_record.append(1)
                else:
                    succ_record.append(0)

            
        writer.add_scalar("total_reward", total_reward, i)
        writer.add_scalar("success_rate", np.mean(succ_record[-100:]), i)
        writer.add_scalar("log_std1", agent.log_std.detach().cpu().numpy().reshape(-1)[0], i)
        writer.add_scalar("log_std2", agent.log_std.detach().cpu().numpy().reshape(-1)[1], i)
        # for type_id in scene_chooser.scene_types:
        #     writer.add_scalar("success_rate_%s"%scene_chooser.scene_types[type_id],
        #         np.mean(scene_chooser.success_record[type_id][-100:]), i)
        writer.add_scalar("step_num", step_num, i)
        reward_list.append(total_reward)
        reward_info = np.sum(np.array(reward_info), axis=0)
        reward_info = np.round(reward_info,2)
        reward_info_list.append(list(reward_info))

        if i%10==0 and i>0:
            print('success rate:',np.sum(succ_record),'/',len(succ_record))
            # print(reward_list[-10:])
            print(agent.log_std.detach().cpu().numpy().reshape(-1))
            # print(agent.state_mean, agent.state_std, agent.n_state)
            print("episode:%s  average reward:%s"%(i,np.mean(reward_list[-50:])))
            print(np.mean(agent.actor_loss_list[-100:]),np.mean(agent.critic_loss_list[-100:]))
            print('time_cost ,rs_dist_reward ,dist_reward ,angle_reward ,box_union_reward')
            for j in range(10):
                print(reward_list[-(10-j)],reward_info_list[-(10-j)])
            print("")

        if i%10000==0:
            agent.save("%s/PPO_%s.pt" % (save_path, i),params_only=True)

    



if __name__ == "__main__":
    # test_manual_control()relative_path = '.'#os.path.dirname(os.getcwd())
    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    save_path = './log/parking_ppo_demo/%s/' % timestamp
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    test_parking_env(save_path)
