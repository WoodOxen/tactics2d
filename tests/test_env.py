import sys

sys.path.append(".")
sys.path.append("..")

import time

import pytest
import logging

logging.basicConfig(level=logging.INFO)

from tactics2d.envs import RacingEnv, ParkingEnv


# @pytest.mark.skip(reason="not implemented")
# def test_parking_env():
#     return

@pytest.mark.env
# @pytest.mark.skip(reason="not implemented")
def test_racing_env():
    env = RacingEnv(render_mode="rgb_array", render_fps=60, max_step=2000)
    env.reset()

    n_step = 500
    t1 = time.time()
    for _ in range(n_step):
        observation, reward, terminated, truncated, info = env.step((0, 1))
        # env.render()
    t2 = time.time()
    logging.info(f"Simulation took {t2 - t1:.2f} seconds.")
    logging.info(f"The average fps is {n_step / (t2 - t1): .2f} Hz.")


if __name__ == "__main__":
    pass
