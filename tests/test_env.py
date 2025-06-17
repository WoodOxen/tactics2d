##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: test_env.py
# @Description:
# @Author: Yueyuan Li
# @Version: 1.0.0


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

from tactics2d.envs import ParkingEnv, RacingEnv


@pytest.mark.env
@pytest.mark.skip(reason="TODO")
def test_racing_env():
    render_mode = "human" if "DISPLAY" in os.environ else "rgb_array"
    env = RacingEnv(render_mode=render_mode, render_fps=60, max_step=2000)
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
@pytest.mark.skip(reason="TODO")
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
