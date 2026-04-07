##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: AU7043_project_1_2025.py
# @Description:
# @Author: Tactics2D Team
# @Version:

import sys

sys.path.append(".")

import logging

logging.basicConfig(level=logging.INFO)

from car_following import CarFollowingEnv


class MyCarFollowingModel:
    def __init__(self):
        # you can define the default parameters here
        pass

    def step(self):  # customize the input for your car following model
        steering = 0
        accel = 0.1
        # steering = np.random.uniform(low=-0.5, high=0.5)
        # accel = np.random.uniform(low=-4, high=2)
        return steering, accel


def main():
    env = CarFollowingEnv(dataset="ngsim", render_mode="human")
    observation, _ = env.reset()

    car_following_model = MyCarFollowingModel()

    for step in range(200):
        env.render()
        ego_state = env.get_ego_state()
        target_vehicle_state = env.get_target_state()

        action = car_following_model.step()
        observation = env.step(action)


if __name__ == "__main__":
    main()
