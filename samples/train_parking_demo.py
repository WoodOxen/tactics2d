import sys

sys.path.append(".")
sys.path.append("..")

import os
import random
import time

import numpy as np

from tactics2d.envs import ParkingEnv
from tactics2d.scenario import TrafficEvent
# from samples.demo_ppo import DemoPPO

def preprocess_obs(info):
    # process lidar
    lidar_obs = info['lidar']
    lidar_obs = np.clip(lidar_obs, 0.0, 10.0)/10.0
    # process target pose
    half_wheel_base = 1.319
    dest_coords = np.mean(np.array(info['target_area']), axis=0)
    dest_heading = info['target_heading']
    dest_pos = (dest_coords[0]-half_wheel_base*np.cos(dest_heading), \
                dest_coords[1]-half_wheel_base*np.sin(dest_heading), dest_heading)
    ego_pos = (info['position_x'], info['position_y'], info['heading'])
    rel_distance = np.sqrt((dest_pos[0]-ego_pos[0])**2 + (dest_pos[1]-ego_pos[1])**2)
    rel_angle = np.arctan2(dest_pos[1]-ego_pos[1], dest_pos[0]-ego_pos[0]) - ego_pos[2]
    rel_dest_heading = dest_pos[2] - ego_pos[2]
    tgt_repr = (rel_distance, np.cos(rel_angle), np.sin(rel_angle),\
        np.cos(rel_dest_heading), np.cos(rel_dest_heading),)
    speed = np.sqrt(info['velocity'][0]**2+info['velocity'][1]**2)
    other_info_repr = np.array(tgt_repr + (speed,))
    obs = {'lidar':lidar_obs, 'other':other_info_repr}
    print(obs)
    return obs

def test_parking_env():
    render_mode = "human"
    env = ParkingEnv(render_mode=render_mode, render_fps=60, max_step=2000)
    env.reset(42)
    agent = None#DemoPPO()

    reward_list = []
    reward_info_list = []
    case_id_list = []
    succ_record = []
    for i in range(100000):
        _, info = env.reset()
        obs = preprocess_obs(info)
        done = False
        total_reward = 0
        step_num = 0
        reward_info = []
        while not done:
            step_num += 1
            # action, log_prob = agent.get_action(obs) # time consume: 3ms
            action = env.action_space.sample()
            # t = time.time()
            _, reward, done, _, info = env.step(action)
            next_obs = preprocess_obs(info)
            # print(time.time()-t)
            reward_info.append(list(info['rewards'].values()))
            total_reward += reward
            # agent.push_memory((obs, action, reward, done, log_prob, next_obs))
            obs = next_obs
            # if len(agent.memory) % agent.batch_size == 0:
            #     actor_loss, critic_loss = agent.update()
                # writer.add_scalar("actor_loss", actor_loss, i)
                # writer.add_scalar("critic_loss", critic_loss, i)
            
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

            
        # writer.add_scalar("total_reward", total_reward, i)
        # for type_id in scene_chooser.scene_types:
        #     writer.add_scalar("success_rate_%s"%scene_chooser.scene_types[type_id],
        #         np.mean(scene_chooser.success_record[type_id][-100:]), i)
        # writer.add_scalar("step_num", step_num, i)
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
                print(case_id_list[-(10-j)],reward_list[-(10-j)],reward_info_list[-(10-j)])
            print("")

    



if __name__ == "__main__":
    # test_manual_control()
    test_parking_env()
