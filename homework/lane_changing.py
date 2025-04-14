###! python3
# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# @File: lane_changing.py
# @Description:
# @Author: Tactics2D Team
# @Version:

import logging
import pickle
import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.error import InvalidAction

from tactics2d.dataset_parser import LevelXParser
from tactics2d.map.parser import OSMParser
from tactics2d.participant.trajectory import State
from tactics2d.physics import SingleTrackKinematics
from tactics2d.sensor import RenderManager, TopDownCamera
from tactics2d.traffic import ScenarioManager, ScenarioStatus, TrafficStatus
from tactics2d.utils.common import get_absolute_path

MAX_STEER = 0.5
MAX_ACCEL = 2.0
MIN_ACCEL = -4.0

map_config = {
    "name": "highD location 1",
    "osm_path": "../data/highD/highD_1.osm",
    "country": "DEU",
    "project_rule": {"proj": "utm", "ellps": "WGS84", "zone": 31, "datum": "WGS84"},
    "gps_origin": [0.001, 0.0],
    "trajectory_types": ["Car", "Truck"],
}

ego_candidates = [
    (37, 13, 351),
    (62, 206, 573),
    (87, 447, 832),
    (131, 883, 1259),
    (143, 954, 1324),
    (144, 959, 1334),
    (154, 1032, 1427),
    (172, 1175, 1619),
    (264, 1988, 2345),
    (279, 2113, 2490),
    (283, 2136, 2507),
    (286, 2154, 2580),
    (373, 2933, 3336),
    (390, 3088, 3514),
    (427, 3412, 3820),
    (485, 3807, 4271),
    (502, 3946, 4409),
    (532, 4202, 4512),
    (568, 4560, 4878),
    (595, 4830, 5152),
    (666, 5502, 5882),
    (748, 6227, 6604),
    (791, 6553, 6953),
    (823, 6769, 7114),
    (984, 8007, 8405),
    (1047, 8651, 8959),
    (1059, 8767, 9135),
    (1112, 9228, 9538),
    (1144, 9530, 9925),
    (1164, 9737, 10107),
    (1185, 9953, 10357),
    (1220, 10284, 10592),
    (1230, 10379, 10767),
    (1265, 10778, 11104),
    (1282, 10929, 11234),
    (1329, 11399, 11740),
    (1416, 12213, 12561),
    (1443, 12371, 12750),
    (1466, 12534, 12921),
    (1521, 12943, 13377),
    (1524, 12963, 13290),
    (1615, 13769, 14116),
    (1649, 14064, 14439),
    (1650, 14090, 14466),
    (1685, 14404, 14737),
    (1691, 14448, 14830),
    (1736, 14851, 15277),
]


class LaneChangingEnv(gym.Env):
    """This class provides a simplified environment for the lane-changing task.

    The utility of this environment is mainly for SJTU's course AU7043. The community can refer to this environment as an example about how to call the dataset parser and map parsers to customize your own training environment.

    ## Observation
    `LaneChaning` provides a top-down semantic segmentation image of agent and its surrounding. The observation is represented as an np.ndarray.

    ## Action

    `LaneChangingEnv` accepts either a continuous or a discrete action command for the agent vehicle.

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
        render_mode: str = "human",
        render_fps: int = 25,
        max_step: int = int(1e3),
        continuous: bool = True,
    ):
        """Initialize the racing environment.

        Args:
            render_mode (str, optional): The mode of the rendering. It can be "human" or "rgb_array". Defaults to "human".
            render_fps (int, optional): The frame rate of the rendering. Defaults to 10 Hz.
            max_step (int, optional): The maximum time step of the scenario. Defaults to 1000.
            continuous (bool, optional): Whether to use continuous action space. Defaults to True.

        Raises:
            NotImplementedError: If the render mode is not supported.
        """
        super().__init__()

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

        self.scenario_manager = self._LaneChangingScenarioManager(
            self.max_step,
            40,
            self.render_fps,
            off_screen=self.render_mode != "human",
        )

    def get_surrounding_states(self) -> list:
        return

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

        return observation

    def render(self):
        if self.render_mode == "human":
            self.scenario_manager.render()

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed, options=options)
        observations, infos = self.scenario_manager.reset()
        observation = observations[0]

        return observation, infos

    class _LaneChangingScenarioManager(ScenarioManager):
        _max_steer = MAX_STEER
        _max_accel = MAX_ACCEL
        _min_accel = MIN_ACCEL
        _window_size = (1000, 1000)
        _state_size = (1000, 1000)

        def __init__(
            self,
            max_step: int = None,
            step_size: int = None,
            render_fps: int = None,
            off_screen: bool = False,
        ):
            super().__init__(max_step, step_size, render_fps, off_screen)

            self.map_parser = OSMParser(lanelet2=True)
            self.map_ = self.map_parser.parse(get_absolute_path(map_config["osm_path"]), map_config)

            self.centerlines = self._get_centerlines()

            self.dataset_parser = LevelXParser("highD")

            self.ego_id = None
            self.participants = None
            self.destination = None
            self.cnt_step = 0
            self.compensation_step = 0
            self.status = ScenarioStatus.NORMAL

            self.render_manager = RenderManager(
                fps=self.render_fps, windows_size=self._window_size, off_screen=self.off_screen
            )

        def _get_centerlines(self):
            centerlines = dict()

            def compute_centerline(roadline1, roadline2):
                centerline_points = []
                roadline1_points = roadline1.shape
                roadline2_points = roadline2.shape

                for i in range(len(roadline1_points)):
                    x1, y1 = roadline1_points[i]
                    x2, y2 = roadline2_points[i]
                    centerline_points.append(((x1 + x2) / 2, (y1 + y2) / 2))

                return centerline_points

            centerlines["102000"] = compute_centerline(
                self.map_.roadlines["101899"], self.map_.roadlines["101900"]
            )
            centerlines["102001"] = compute_centerline(
                self.map_.roadlines["101900"], self.map_.roadlines["101901"]
            )
            centerlines["102002"] = compute_centerline(
                self.map_.roadlines["101901"], self.map_.roadlines["101902"]
            )
            centerlines["102003"] = compute_centerline(
                self.map_.roadlines["101903"], self.map_.roadlines["101904"]
            )
            centerlines["102004"] = compute_centerline(
                self.map_.roadlines["101904"], self.map_.roadlines["101905"]
            )
            centerlines["102005"] = compute_centerline(
                self.map_.roadlines["101905"], self.map_.roadlines["101906"]
            )

            return centerlines

        def update(self, steering: float, accel: float):
            self.cnt_step += self.step_size
            step = self.cnt_step + self.compensation_step

            current_state = self.agent.current_state
            next_state, _, _ = self.agent.physics_model.step(current_state, accel, steering)
            self.agent.add_state(next_state)

            logging.debug(f"Step {step}: {self.agent.id_} is at {current_state.location}.")

            participant_ids = []
            for participant_id, participant in self.participants.items():
                if participant.trajectory.has_state(step):
                    participant_ids.append(participant_id)

            self.check_status(step, participant_ids)

            self.render_manager.update(
                self.participants,
                participant_ids,
                step,
            )

            other_states = dict()
            for participant_id in participant_ids:
                if participant_id == self.ego_id:
                    continue
                other_states[participant_id] = self.participants[participant_id].get_state(step)

            infos = {
                "status": self.status,
                "ego_state": self.agent.current_state,
                "other_states": other_states,
                "centerlines": self.centerlines,
            }

            return self.get_observation()[0], infos

        def check_status(self, step, participant_ids):
            ego_locaction = self.agent.current_state.location
            distance = np.sqrt(
                (ego_locaction[0] - self.destination[0]) ** 2
                + (ego_locaction[1] - self.destination[1]) ** 2
            )
            if distance < 10:
                self.status = ScenarioStatus.COMPLETED
                return

            if self.cnt_step > self.max_step * self.step_size:
                self.status = ScenarioStatus.TIME_EXCEEDED

            ego_pose = self.agent.get_pose(step)
            for participant_id in participant_ids:
                if participant_id == self.ego_id:
                    continue

                if ego_pose.intersects(self.participants[participant_id].get_pose(step)):
                    self.status = ScenarioStatus.FAILED
                    break

        def render(self):
            self.render_manager.render()

        def reset(self):
            self.cnt_step = 0
            self.compensation_step = 0
            self.status = ScenarioStatus.NORMAL

            idx = np.random.randint(0, len(ego_candidates))
            # idx = 44 # you can change and fix the index to test a specific scenario
            self.ego_id = ego_candidates[idx][0]
            self.compensation_step = ego_candidates[idx][1] * self.step_size
            print('start parsing')
            # we save the participants to a pickle file for quicker loading
            # if exits, load the participants from the pickle file directly instead of parsing the csv file
            if os.path.exists("../data/highD_participants_ego_id_" + str(idx) + '_' + str(self.ego_id)):
                with open("../data/highD_participants_ego_id_" + str(idx) + '_' + str(self.ego_id), "rb") as f:
                    self.participants = pickle.load(f)
                print('load participants from pickle file: ',f.name)
            else:
                self.participants, _ = self.dataset_parser.parse_trajectory(
                    "11_tracks.csv",
                    "../data/highD",
                    (ego_candidates[idx][1] * self.step_size, ego_candidates[idx][2] * self.step_size),
                )
                # save the participants to a pickle file for quicker loading
                with open("../data/highD_participants_ego_id_" + str(idx) + '_' + str(self.ego_id), "wb") as f:
                    pickle.dump(self.participants, f)
                print('save participants to pickle file: ',f.name)
            print('finish parsing')

            if not self.ego_id in self.participants.keys():
                raise RuntimeError(
                    f"Ego vehicle's trajectory {self.ego_id} not found in the dataset."
                )

            self.agent = self.participants[self.ego_id]
            ego_state = self.agent.trajectory.initial_state
            self.destination = self.agent.trajectory.last_state.location
            logging.info(
                f"Ego vehicle {self.ego_id} is at {ego_state.location} and targeting at {self.destination}."
            )

            self.participants[self.ego_id].trajectory.reset(ego_state)
            self.participants[self.ego_id].color = "light-purple"
            self.participants[self.ego_id].physics_model = SingleTrackKinematics(
                lf=self.agent.length / 2,
                lr=self.agent.length / 2,
                steer_range=(-self._max_steer, self._max_steer),
                speed_range=((-16.67, 55.56)),
                accel_range=(self._min_accel, self._max_accel),
                interval=self.step_size,
            )

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
            self.render_manager.bind(0, self.ego_id)

            other_states = dict()
            for participant_id, participant in self.participants.items():
                if participant_id == self.ego_id:
                    continue
                if participant.trajectory.has_state(self.compensation_step):
                    other_states[participant_id] = participant.get_state(self.compensation_step)

            infos = {
                "status": self.status,
                "ego_state": ego_state,
                "other_states": other_states,
                "centerlines": self.centerlines,
            }

            return self.get_observation(), infos
