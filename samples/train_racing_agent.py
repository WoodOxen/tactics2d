import sys

sys.path.append(".")
sys.path.append("..")

from collections import deque

from gymnasium import spaces
import torch

from rllib.algorithms.sac import SAC
from tactics2d.envs import RacingEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    env = RacingEnv(render_mode="human", render_fps=60)
    agent_configs = {
        "state_space": spaces.Box(0, 255, (200 * 200 * 256, 1)),
        "action_space": env.action_space,
    }
    agent = SAC(agent_configs)

    accumulated_rewards = deque([], maxlen=10)
    step_cnt = 0
    episode_cnt = 0
    print_interval = 10

    while step_cnt < env.max_step:
        accumulated_reward = 0
        observation = env.reset()
        done = False

        while not done:
            env.render()
            state = torch.flatten(torch.FloatTensor(observation).to(device))
            action = agent.get_action(state)
            observation, reward, _, done, info = env.step(action)
            done_mask = 0.0 if done else 1.0
            agent.buffer.push((state, action, reward, done_mask))

            accumulated_reward += reward
            step_cnt += 1

        accumulated_rewards.append(accumulated_reward)
        episode_cnt += 1
        if episode_cnt % print_interval == 0:
            print(
                "Episode: %d, score: %f, buffer capacity: %d"
                % (episode_cnt, accumulated_reward, len(agent.buffer))
            )
