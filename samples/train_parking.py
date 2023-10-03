import sys

sys.path.append(".")
sys.path.append("./rllib")
sys.path.append("..")

import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from rllib.algorithms import PPOConfig, PPO
from rllib.buffer import RandomReplayMemory
from tactics2d.envs import ParkingEnv
from tactics2d.scenario import TrafficEvent


render_modes = ["rgb_array", "human"]
random_seed = 42


class CustomPPO(PPO):
    def __init__(self, configs, device):
        super().__init__(configs, device)


def train(args):
    ppo_config = PPOConfig(
        {
            "horizon": 8192,
            "batch_size": 32,
            "gamma": 0.95,
            "lr": 2e-6,
            "entropy_coef": 0.01,
            "self.adam_epsilon": 1e-8,
        }
    )

    env = ParkingEnv(
        render_mode=render_modes[args.render_mode],
        render_fps=args.render_fps,
        max_step=args.max_step,
    )
    env.reset(random_seed)

    agent = CustomPPO()
    planner = ReedsSheppPlanner()
    writer = SummaryWriter()

    for step in range(args.num_iter):
        _, info = env.reset()
        done = False
        total_reward = 0
        step


def demo(args):
    env = ParkingEnv(
        render_mode=render_modes[args.render_mode],
        render_fps=args.render_fps,
        max_step=args.max_step,
    )
    env.reset(random_seed)

    agent = CustomPPO()
    agent.load("./models/parking_agent.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-mode", type=int, default=0)
    parser.add_argument("--render-fps", type=int, default=60)
    parser.add_argument("--max-step", type=int, default=200)
    parser.add_argument("--num-iter", type=int, default=int(1e5))
