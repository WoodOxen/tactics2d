import sys

sys.path.append(".")
sys.path.append("..")

import logging
import os
import random
import time

import pytest

logging.basicConfig(level=logging.INFO)

import numpy as np

# from tactics2d.envs import RacingEnv, ParkingEnv
from tactics2d.envs import ParkingEnv

# @pytest.mark.env
# @pytest.mark.skipif("DISPLAY" not in os.environ, reason="requires display server")
# def test_racing_env():
#     random.seed = 42
#     np.random.seed = 42
#     render_mode = "rgb_array"
#     env = RacingEnv(render_mode=render_mode, render_fps=60, max_step=2000)
#     env.reset(42)

#     n_step = 600
#     t1 = time.time()
#     for _ in range(n_step):
#         _, _, _, _, _ = env.step(env.action_space.sample())
#         if render_mode == "human":
#             env.render()
#     t2 = time.time()
#     logging.info(f"Simulation took {t2 - t1:.2f} seconds.")
#     logging.info(f"The average fps is {n_step / (t2 - t1): .2f} Hz.")


@pytest.mark.env
def test_parking_env():
    render_mode = "human" if "DISPLAY" in os.environ else "rgb_array"
    env = ParkingEnv(render_mode=render_mode, render_fps=60, max_step=2000)
    env.reset(42)

    n_iter = 600
    t1 = time.time()
    for _ in range(n_iter):
        _ = env.step(action=env.action_space.sample())
        if render_mode == "human":
            env.render()
    t2 = time.time()
    logging.info(f"Simulation took {t2 - t1:.2f} seconds.")
    logging.info(f"The average fps is {n_iter / (t2 - t1): .2f} Hz.")


@pytest.mark.env
@pytest.mark.skip(reason="Terminal only")
def test_manual_control(env):
    pass


if __name__ == "__main__":
    render_mode = "human"
    env = ParkingEnv(render_mode=render_mode, render_fps=60, max_step=2000, continuous=True)
    env.reset(42)
    n_iter = 100
    t1 = time.time()
    for _ in range(n_iter):
        _ = env.step(env.action_space.sample())
        # if render_mode == "human":
        env.render()
    t2 = time.time()
    print(f"Simulation took {t2 - t1:.2f} seconds.")
    print(f"The average fps is {n_iter / (t2 - t1): .2f} Hz.")
