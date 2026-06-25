# Copyright (C) 2023, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for env."""


import sys

sys.path.append(".")
sys.path.append("..")

import logging
import os
import time

import pytest

logging.basicConfig(level=logging.INFO)

_HAS_DISPLAY = bool(os.getenv("DISPLAY"))


from tactics2d.envs import ParkingEnv, RacingEnv


def _simulate(env_type, render_mode, n_iter=600):
    env = env_type(render_mode=render_mode, render_fps=60, max_step=2000)
    env.reset(42)

    t1 = time.time()
    for i in range(n_iter):
        _ = env.step(action=env.action_space.sample())
        if render_mode == "human":
            env.render()
    t2 = time.time()
    env.close()

    name = env_type.__name__
    logging.info(f"{name}[{render_mode}] simulation took {t2 - t1:.2f} seconds.")
    logging.info(f"{name}[{render_mode}] average fps is {n_iter / (t2 - t1): .2f} Hz.")


@pytest.mark.env
@pytest.mark.slow
@pytest.mark.parametrize(
    "render_mode",
    [
        "rgb_array",
        pytest.param(
            "human",
            marks=pytest.mark.skipif(
                not _HAS_DISPLAY, reason="DISPLAY not available, cannot open window."
            ),
        ),
    ],
)
def test_racing_env(render_mode):
    _simulate(RacingEnv, render_mode)


@pytest.mark.env
@pytest.mark.slow
@pytest.mark.parametrize(
    "render_mode",
    [
        "rgb_array",
        pytest.param(
            "human",
            marks=pytest.mark.skipif(
                not _HAS_DISPLAY, reason="DISPLAY not available, cannot open window."
            ),
        ),
    ],
)
def test_parking_env(render_mode):
    _simulate(ParkingEnv, render_mode)


@pytest.mark.env
@pytest.mark.skip(reason="Terminal only")
def test_manual_control():
    pass
