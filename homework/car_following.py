##! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: car_following.py
# @Description:
# @Author: Tactics2D Team
# @Version:

import logging
import os

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.error import InvalidAction
from pyproj import Proj

from tactics2d.dataset_parser import NGSIMParser
from tactics2d.map.element import Map
from tactics2d.map.parser import GISParser
from tactics2d.participant.element import Vehicle
from tactics2d.participant.trajectory import State
from tactics2d.physics import SingleTrackKinematics
from tactics2d.sensor import RenderManager, TopDownCamera
from tactics2d.traffic import ScenarioManager, ScenarioStatus, TrafficStatus
from tactics2d.utils.common import get_absolute_path

MAX_STEER = 0.5
MAX_ACCEL = 2.0
MIN_ACCEL = -4.0

NGSIM_TREE = {
    "./data/NGSIM/I-80-Emeryville-CA": {
        "gis-files": ["emeryville.shp"],
        "epsg": 2229,
        "bounds": [-118.359, -118.365, 34.139, 34.135],  # west, east, north, south
        "trajectories": ["trajectories-0500-0515.csv", "trajectories-0515-0530.csv"],
    },
    "./data/NGSIM/US-101-LosAngeles-CA": {
        "gis-files": ["US-101.shp"],
        "epsg": 2227,
        "bounds": [-122.2987, -122.2962, 37.8466, 37.8385],
        "trajectories": [
            "trajectories-0750am-0805am.csv",
            "trajectories-0805am-0820am.csv",
            "trajectories-0820am-0835am.csv",
        ],
    },
    "./data/NGSIM/Lankershim-Boulevard-LosAngeles-CA": {
        "gis-files": ["LA-UniversalCity.shp"],
        "epsg": 2229,
        "bounds": [-118.363, -118.360, 34.143, 34.137],
        "trajectories": ["trajectories.csv"],
    },
}


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
        render_fps: int = 10,
        max_step: int = int(1e3),
        continuous: bool = True,
    ):
        """Initialize the racing environment.

        Args:
            dataset (str, optional): The dataset providing the target vehicle for following. The available choices are ["ngsim", "highd"]. Defaults to "ngsim".
            render_mode (str, optional): The mode of the rendering. It can be "human" or "rgb_array". Defaults to "human".
            render_fps (int, optional): The frame rate of the rendering. Defaults to 10 Hz.
            max_step (int, optional): The maximum time step of the scenario. Defaults to 1000.
            continuous (bool, optional): Whether to use continuous action space. Defaults to True.

        Raises:
            NotImplementedError: If the render mode is not supported.
        """
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

        self.scenario_manager = self._CarFollowingScenarioManager(
            self.dataset,
            self.max_step,
            100,
            self.render_fps,
            off_screen=self.render_mode != "human",
        )

    def get_ego_state(self) -> State:
        """This function returns the current state of the ego vehicle."""
        return self.scenario_manager.agent.trajectory.last_state

    def get_target_state(self) -> State:
        frame = self.scenario_manager.agent.trajectory.last_state.frame
        return self.scenario_manager.participants[1].trajectory.get_state(frame)

    def step(self, action):
        """This function takes a step in the environment.

        Args:
            action (Union[tuple, int]): The action command for the agent vehicle. If the action space is continuous, the input should be a tuple, whose first element controls the steering value and the second controls the acceleration. If the action space is discrete, the input should be an index that points to a pre-defined control command.

        Raises:
            InvalidAction: If the action is not in the action space."

        Returns:
            observation (np.array): The BEV image observation of the environment.
        """
        if not self.action_space.contains(action):
            raise InvalidAction(f"Action {action} is not in the action space.")
        action = action if self.continuous else self._discrete_action[action]

        steering, accel = action
        observation = self.scenario_manager.update(steering, accel)

        return

    def render(self):
        if self.render_mode == "human":
            self.scenario_manager.render()

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed, options=options)
        self.scenario_manager.reset()
        observations = self.scenario_manager.get_observation()
        observation = observations[0]

        infos = {}

        return observation, infos

    class _CarFollowingScenarioManager(ScenarioManager):
        _max_steer = MAX_STEER
        _max_accel = MAX_ACCEL
        _min_accel = MIN_ACCEL
        _window_size = (1000, 1000)
        _state_size = (1000, 1000)

        def __init__(
            self,
            dataset: str,
            max_step: int = None,
            step_size: int = None,
            render_fps: int = None,
            off_screen: bool = False,
        ):
            super().__init__(max_step, step_size, render_fps, off_screen)

            self.dataset = dataset

            self.agent = Vehicle(id_=0)
            self.agent.load_from_template("medium_car")
            self.agent.physics_model = SingleTrackKinematics(
                lf=self.agent.length / 2 - self.agent.front_overhang,
                lr=self.agent.length / 2 - self.agent.rear_overhang,
                steer_range=(-self._max_steer, self._max_steer),
                speed_range=self.agent.speed_range,
                accel_range=(self._min_accel, self._max_accel),
                interval=self.step_size,
            )

            # Only two participants:
            # - 0: ego vehicle
            # - 1: target vehicle
            self.participants = {self.agent.id_: self.agent}

            self.map_ = Map(name="car-following")

            if self.dataset == "ngsim":
                self.target_vehicle_index = pd.read_csv(
                    get_absolute_path("./data/NGSIM/target-vehicle-index.csv")
                )
                self.trajectory_parser = NGSIMParser()
                self.map_parser = GISParser()
            elif self.dataset == "highd":
                pass

            self.render_manager = RenderManager(
                fps=self.render_fps, windows_size=self._window_size, off_screen=self.off_screen
            )

        def _load_from_ngsim(self):
            target_vehicle_index = self.target_vehicle_index.sample(1).to_dict(orient="records")[0]

            logging.info(
                "Loading vehicle %d from %s"
                % (target_vehicle_index["Vehicle_ID"], target_vehicle_index["File"])
            )
            logging.info(
                "Using state of vehicle %d at frame %d to initialize the agent."
                % (target_vehicle_index["Following"], target_vehicle_index["Start_Following"])
            )

            # Parse the map
            map_folder = target_vehicle_index["Folder"]

            self.map_ = self.map_parser.parse(
                [
                    get_absolute_path(map_folder) + "/gis-files/" + gis_file
                    for gis_file in NGSIM_TREE[map_folder]["gis-files"]
                ],
            )

            # Parse the trajectory file
            # TODO: Bottleneck in loading csv file

            participants, actual_time_stamp = self.trajectory_parser.parse_trajectory(
                target_vehicle_index["File"],
                get_absolute_path(target_vehicle_index["Folder"]),
                stamp_range=(target_vehicle_index["Start_Following"], np.inf),
                ids=[target_vehicle_index["Vehicle_ID"], target_vehicle_index["Following"]],
            )

            self.participants[1] = participants[target_vehicle_index["Vehicle_ID"]]
            self.participants[1].color = "light-purple"

            # estimate the heading of the target vehicle, not accurate, only for visualizing
            first_frame = self.participants[1].trajectory.first_frame
            last_frame = self.participants[1].trajectory.last_frame
            overall_heading = np.arctan2(
                self.participants[1].trajectory.get_state(first_frame + 100 * self.step_size).y
                - self.participants[1].trajectory.get_state(first_frame).y,
                self.participants[1].trajectory.get_state(first_frame + 100 * self.step_size).x
                - self.participants[1].trajectory.get_state(first_frame).x,
            )

            self.compensation_step = first_frame

            for frame in np.arange(first_frame, last_frame, self.step_size):
                current_state = self.participants[1].trajectory.get_state(frame)
                next_state = self.participants[1].trajectory.get_state(frame + self.step_size)
                estimated_heading = np.arctan2(
                    next_state.y - current_state.y, next_state.x - current_state.x
                )

                delta = np.abs(overall_heading - estimated_heading)
                if delta > np.pi:
                    delta = 2 * np.pi - delta

                if delta > np.pi / 2:
                    estimated_heading = -estimated_heading

                self.participants[1].trajectory.get_state(frame).set_heading(estimated_heading)

            self.participants[1].trajectory.get_state(last_frame).set_heading(
                self.participants[1].trajectory.get_state(last_frame - self.step_size).heading
            )

            ego_state = participants[target_vehicle_index["Following"]].trajectory.get_state(
                first_frame
            )

            ego_state.set_heading(self.participants[1].trajectory.get_state(first_frame).heading)

            self.agent.reset(ego_state)

            logging.info(
                "The ego vehicle starts at ({:.3f}, {:.3f}), heading to {:.3f} rad.".format(
                    ego_state.x, ego_state.y, estimated_heading
                )
            )

        def update(self, steering: float, accel: float):
            self.cnt_step += self.step_size
            current_state = self.agent.current_state
            next_state, _, _ = self.agent.physics_model.step(current_state, accel, steering)
            self.agent.add_state(next_state)

            self.render_manager.update(
                self.participants, [0, 1], self.cnt_step + self.compensation_step
            )

            return self.get_observation()

        def check_status(self):
            return

        def render(self):
            self.render_manager.render()

        def reset(self):
            self.cnt_step = 0
            self.compensation_step = 0
            self.map_.reset()

            if self.dataset == "ngsim":
                self._load_from_ngsim()

            self.render_manager.reset()

            # reset the sensors
            camera = TopDownCamera(
                id_=0,
                map_=self.map_,
                perception_range=(75, 75, 100, 50),
                window_size=self._state_size,
                off_screen=self.off_screen,
            )
            self.render_manager.add_sensor(camera)
            self.render_manager.bind(0, 0)
