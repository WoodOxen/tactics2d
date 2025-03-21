##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: AU7043_project_1_2025.py
# @Description:
# @Author: Tactics2D Team
# @Version:

import tactics2d

from .car_following import CarFollowingEnv

NGSIM_PATH = {
    "folder": "./data/NGSIM",
    "I-80-Emeryville-CA": {
        "trajectories": ["trajectories-0500-0515.csv", "trajectories-0515-0530.csv"]
    },
    "Lankershim-Boulevard-LosAngeles-CA": {"trajectories": ["trajectories.csv"]},
    "US-101-LosAngeles-CA": {
        "trajectories": [
            "trajectories-0750am-0805am.csv",
            "trajectories-0805am-0820am.csv",
            "trajectories-0820am-0835am.csv",
        ]
    },
}


class MyCarFollowingModel:
    def __init__(self):
        super().__init__()

    def step(self):
        return


def main():
    env = CarFollowingEnv(dataset="ngsim", render_mode="human")


if __name__ == "__main__":
    main()
