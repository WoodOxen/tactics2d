##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: car_following.py
# @Description:
# @Author: Tactics2D Team
# @Version:


import logging

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.error import InvalidAction
from shapely.geometry import Polygon

from tactics2d.participant.element import Vehicle
from tactics2d.physics import SingleTrackKinematics
from tactics2d.sensor import RenderManager
from tactics2d.traffic import ScenarioManager, ScenarioStatus, TrafficStatus

MAX_STEER = 0.5
MAX_ACCEL = 2.0
MIN_ACCEL = -4.0


class CarFollowingEnv(gym.Env):
    """This class provides a simplified environment for the car-following task. It randomly selects a vehicle trajectory from either the NGSIM or HighD dataset, ensuring the chosen vehicle has remained in the same lane throughout its recorded history. All other vehicles in the scenario are then removed, leaving only the selected trajectory. The goal of the ego vehicle is to follow this log-replay vehicle as accurately as possible.

    The utility of this environment is mainly for SJTU's course AU7043. The community can refer to this environment as an example about how to call the dataset parser and map parsers to customize your own training environment.

    ## Observation
    `CarFollowingEnv` provides a top-down semantic segmentation image of agent and its surrounding. The observation is represented as an np.ndarray.

    ## Action

    `CarFollowingEnv` accepts either a continuous or a discrete action command for the agent vehicle.

    - The continuous action is a 2D numpy array with the shape of (2,) representing the steering angle and the acceleration.
    - The discrete action is an integer range in [0, 142] that points to a 2D numpy array with the shape of (2,) representing the steering angle and the acceleration. The steering ranges in [-0.5, 0.5, 0.1]. The acceleration ranges in [-2.0, 4.0, 0.5].

    The first element of the action tuple is the steering angle, which should be in the range of [-0.5, 0.5]. Its unit is radian. The second element of the action tuple is the acceleration, which should be in the range of [-2.0, 4.0]. Its unit is m/s$^2$.
    """

    _metadata = {"render_modes": ["human", "rgb_array"]}
    _max_fps = 200
    _max_steer = MAX_STEER
    _max_accel = MAX_ACCEL
    _min_accel = MIN_ACCEL

    def __init__(
        self,
        dataset: str = "ngsim",
        render_mode: str = "human",
        render_fps: int = 60,
        max_step: int = int(1e3),
        continuous: bool = True,
    ):
        super().__init__()

        if dataset.lower() not in ["ngsim", "highd"]:
            raise NotImplementedError(f"Parser for dataset {dataset} is not supported.")
        self.dataset = dataset.lower()

        if render_mode not in self._metadata["render_modes"]:
            raise NotImplementedError(f"Render mode {render_mode} is not supported.")
        self.render_mode = render_mode

        self.render_fps = np.clip(render_fps, 1, self._max_fps)
        if self.render_fps != render_fps:
            logging.warning(f"Render FPS {render_fps} is not supported. Set to {self.render_fps}.")

        self.max_step = max_step
        self.continuous = continuous

        self.observation_space = spaces.Box(0, 255, shape=(200, 200, 3), dtype=np.uint8)

        if self.continuous:
            self.action_space = spaces.Box(
                np.array([-self._max_steer, self._min_accel]),
                np.array([self._max_steer, self._max_accel]),
                dtype=np.float32,
            )
        else:
            x = np.linspace(-self._max_steer, self._max_steer, 11)
            y = np.linspace(self._min_accel, self._max_accel, 13)
            xx, yy = np.meshgrid(x, y)
            self._discrete_action = np.vstack([xx.ravel(), yy.ravel()]).T
            self.action_space = spaces.Discrete(len(self._discrete_action))

    def render(self):
        if self.render_mode == "human":
            self.scenario_manager.render()

    class _CarFollowingScenarioManager(ScenarioManager):
        _max_steer = MAX_STEER
        _max_accel = MAX_ACCEL
        _window_size = (500, 500)
        _state_size = (200, 200)

        def __init__(
            self,
            dataset: str,
            max_step: int = None,
            step_size: int = None,
            render_fps: int = 60,
            off_screen: bool = False,
        ):
            super().__init__(max_step, step_size, render_fps, off_screen)

            self.agent = Vehicle(id_=0)
            self.agent.load_from_template("medium_car")
            self.agent.physics_model = SingleTrackKinematics(
                lf=self.agent.length / 2 - self.agent.front_overhang,
                lr=self.agent.length / 2 - self.agent.rear_overhang,
                steer_range=(-self._max_steer, self._max_steer),
                speed_range=(-0.5, 0.5),
                accel_range=(-self._max_accel, self._max_accel),
                interval=self.step_size,
            )

            if dataset == "ngsim":
                self.heading_vehicle, self.map_ = self.load_from_ngsim()
            elif dataset == "highd":
                self.heading_vehicle, self.map_ = self.load_from_highd()

            self.render_manager = RenderManager(
                fps=self.render_fps, windows_size=self._window_size, off_screen=self.off_screen
            )

        def update(self, steering: float, accel: float):
            self.cnt_step += 1

        def render(self):
            self.render_manager.render()

        def reset(self):
            self.cnt_step = 0
