###! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: AU7043_project_2_2025.py
# @Description:
# @Author: Tactics2D Team
# @Version:

import sys

sys.path.append(".")

import logging

logging.basicConfig(level=logging.INFO)

import numpy as np
from lane_changing import LaneChangingEnv


class MyLaneChangingModel:
    def __init__(self):
        pass

    def step(self):  # customize the input for your car following model
        steering = 0
        accel = 0.2
        # steering = np.random.uniform(low=-0.5, high=0.5)
        # accel = np.random.uniform(low=-4, high=2)
        return steering, accel


def main(level="easy"):
    if level == "easy":
        max_step = 500
    elif level == "medium":
        max_step = 400
    elif level == "hard":
        max_step = 300

    env = LaneChangingEnv(render_mode="human", max_step=max_step)
    observation, infos = env.reset()

    # The infors include the traffic status, the ego vehicle status, the other vehicles status, and the centerlines
    logging.info(f"Infos: {infos.keys()}")

    lane_changing_model = MyLaneChangingModel()

    for step in range(max_step + 10):
        env.render()

        action = lane_changing_model.step()
        observation, infos = env.step(action)

        logging.debug(infos["status"].name)

        if infos["status"].name not in ["NORMAL", "COMPLETED"]:
            raise RuntimeError(
                f"Simulation failed with status: {infos['status'].name} at step {step}."
            )
        elif infos["status"].name == "COMPLETED":
            logging.info(f"Simulation completed successfully at step {step}.")
            break


if __name__ == "__main__":
    np.random.seed(0)  # define the random seed to reproduce the scenario
    main()
