##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: parking.py
# @Description: This script defines a parking environment.
# @Author: Yueyuan Li
# @Version: 1.0.0


import logging
from typing import Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.error import InvalidAction
from shapely.geometry import Polygon

from tactics2d.map.element import Map
from tactics2d.map.generator import ParkingLotGenerator
from tactics2d.participant.element import Vehicle
from tactics2d.physics import SingleTrackKinematics
from tactics2d.sensor import RenderManager, SingleLineLidar, TopDownCamera
from tactics2d.traffic import ScenarioManager, ScenarioStatus, TrafficStatus
from tactics2d.traffic.event_detection import (
    Arrival,
    NoAction,
    OutBound,
    StaticCollision,
    TimeExceed,
)

MAX_STEER = 0.524  # 0.75
MAX_ACCEL = 2.0


def truncate_angle(angle: float):
    """Truncate angle to [-pi, pi]"""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi

    return angle


class ParkingEnv(gym.Env):
    """This class provides an environment to train a vehicle to park in a parking lot without dynamic traffic participants, such as pedestrians and vehicles.

    ## Observation

    `ParkingEnv` provides two types of observations:

    - Camera: A top-down semantic segmentation image of the agent vehicle and its surrounding. The perception range is 20 meters. The image is returned as a 3D numpy array with the shape of (200, 200, 3).
    - LiDAR: A single line lidar sensor that scans a full 360-degree view with a range of 20 meters. The lidar data is returned as a 1D numpy array with the shape of (120,).

    ## Action

    `ParkingEnv` accepts either a continuous or a discrete action command for the agent vehicle.

    - The continuous action is a 2D numpy array with the shape of (2,) representing the steering angle and the acceleration.
    - The discrete action is an integer from 1 to 5, representing the following actions:
        1. Do nothing: (0, 0)
        2. Turn left: (-0.5, 0)
        3. Turn right: (0.5, 0)
        4. Move forward: (0, 1)
        5. Move backward: (0, -1)

    The first element of the action tuple is the steering angle, which should be in the range of [-0.75, 0.75]. Its unit is radian. The second element of the action tuple is the acceleration, which should be in the range of [-2.0, 2.0]. Its unit is m/s$^2$.

    ## Status and Reward

    The environment provides a reward for each step. The status check and reward is calculated based on the following rules:

    1. Check time exceed: If the time step exceeds the maximum time step (20000 steps), the scenario status will be set to `TIME_EXCEEDED` and a negative reward -1 will be given.
    2. Check no action: If the agent vehicle does not move for over 100 steps, the scenario status will be set to `NO_ACTION` and a negative reward -1 will be given.
    3. Check out bound: If the agent vehicle goes out of the boundary of the map, the scenario status will be set to `OUT_BOUND` and a negative reward -5 will be given.
    4. Check collision: If the agent vehicle collides with the static obstacles, the traffic status will be set to `COLLISION_STATIC` and a negative reward -5 will be given.
    5. Check completed: If the agent vehicle successfully parks in the target area, the scenario status will be set to `COMPLETED` and a positive reward 5 will be given.
    6. Otherwise, the reward is calculated as the sum of the time penalty and the IoU reward. The time penalty is calculated as -tanh(t / T) * 0.1, where t is the current time step and T is the maximum time step. The IoU reward is calculated as the difference between the current IoU and the maximum IoU.

    If the agent has successfully completed the scenario, the environment will set the terminated flag to True. If the scenario status goes abnormal or the traffic status goes abnormal, the environment will set the truncated flag to True.

    The status information is returned as a dictionary with the following keys:

    - `lidar`: A 1D numpy array with the shape of (120,) representing the lidar data.
    - `state`: The current state of the agent vehicle.
    - `target_area`: The coordinates of the target area.
    - `target_heading`: The heading of the target area.
    - `traffic_status`: The status of the traffic scenario.
    - `scenario_status`: The status of the scenario.
    """

    _metadata = {"render_modes": ["human", "rgb_array"]}
    _max_fps = 200
    _max_steer = MAX_STEER
    _max_accel = MAX_ACCEL
    _discrete_actions = {1: (0, 0), 2: (-0.5, 0), 3: (0.5, 0), 4: (0, 1), 5: (0, -1)}

    def __init__(
        self,
        type_proportion: float = 0.5,
        render_mode: str = "human",
        render_fps: int = 60,
        max_step: int = int(2e4),
        continuous: bool = True,
    ):
        """Initialize the parking environment.

        Args:
            type_proportion (float, optional): The proportion of "bay" parking scenario in all generated scenarios. It should be in the range of [0, 1]. If the input is out of the range, it will be clipped to the range. When it is 0, the generator only generates "parallel" parking scenarios. When it is 1, the generator only generates "bay" parking scenarios.
            render_mode (str, optional): The mode of the rendering. It can be "human" or "rgb_array".
            render_fps (int, optional): The frame rate of the rendering.
            max_step (int, optional): The maximum time step of the scenario.
            continuous (bool, optional): Whether to use continuous action space.

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
                np.array([-self._max_steer, -self._max_accel]),
                np.array([self._max_steer, self._max_accel]),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Discrete(5)

        self._max_iou = -np.inf
        self._min_dist_to_target = np.inf

        self.scenario_manager = self._ParkingScenarioManager(
            type_proportion, self.max_step, 100, self.render_fps, self.render_mode != "human"
        )

    def _get_reward(
        self, scenario_status: ScenarioStatus, traffic_status: TrafficStatus, iou: float
    ):
        if traffic_status == TrafficStatus.COLLISION_STATIC:
            reward = -5
        elif (
            scenario_status == ScenarioStatus.TIME_EXCEEDED
            or scenario_status == ScenarioStatus.NO_ACTION
        ):
            reward = -1
        elif scenario_status == ScenarioStatus.OUT_BOUND:
            reward = -5
        elif scenario_status == ScenarioStatus.COMPLETED:
            reward = 5
        else:
            time_penalty = -np.tanh(self.scenario_manager.cnt_step / self.max_step) * 0.001
            if self._max_iou == -np.inf:
                iou_reward = iou if not iou is None else 0
            else:
                iou_reward = iou - self._max_iou if not iou is None else 0

            reward = time_penalty + iou_reward
            self._max_iou = max(self._max_iou, iou) if not iou is None else self._max_iou

            curr_dist_to_target = np.linalg.norm(
                np.array(
                    [
                        self.scenario_manager.agent.current_state.x,
                        self.scenario_manager.agent.current_state.y,
                    ]
                )
                - np.array(
                    [
                        self.scenario_manager.target_area.geometry.centroid.x,
                        self.scenario_manager.target_area.geometry.centroid.y,
                    ]
                )
            )
            if curr_dist_to_target < self._min_dist_to_target:
                reward += (self._min_dist_to_target - curr_dist_to_target) * 0.1
                self._min_dist_to_target = curr_dist_to_target

        return reward

    def _get_relative_pose(self, state):
        target_pose = np.array(
            [
                self.scenario_manager.target_area.geometry.centroid.x,
                self.scenario_manager.target_area.geometry.centroid.y,
            ]
        )
        target_heading = self.scenario_manager.target_heading
        diff_position = np.linalg.norm(target_pose - np.array(state.location))
        diff_angle = np.arctan2(target_pose[1] - state.y, target_pose[0] - state.x) - state.heading
        diff_heading = target_heading - state.heading

        return diff_position, diff_angle, diff_heading

    def _get_infos(self, state, observations, scenario_status, traffic_status):
        state_infos = dict()
        state_infos["lidar"] = observations[1]
        state_infos["state"] = state
        state_infos["target_area"] = self.scenario_manager.target_area
        state_infos["target_heading"] = self.scenario_manager.target_heading
        state_infos["traffic_status"] = traffic_status
        state_infos["scenario_status"] = scenario_status
        (
            state_infos["diff_position"],
            state_infos["diff_angle"],
            state_infos["diff_heading"],
        ) = self._get_relative_pose(state)
        return state_infos

    def step(self, action: Union[np.ndarray, int]):
        """This function takes a step in the environment.

        Args:
            action (Union[Tuple[float], int]): The action command for the agent vehicle.

        Raises:
            InvalidAction: If the action is not in the action space.

        Returns:
            observation (np.array): The BEV image observation of the environment.
            reward (float): The reward of the environment.
            terminated (bool): Whether the scenario is terminated. If the agent has completed the scenario, the scenario is terminated.
            truncated (bool): Whether the scenario is truncated. If the scenario status goes abnormal or the traffic status goes abnormal, the scenario is truncated.
            infos (dict): The information of the environment.
        """
        if not self.action_space.contains(action):
            raise InvalidAction(f"Action {action} is not in the action space.")
        action = action if self.continuous else self._discrete_action[action]

        steering, accel = action
        observations = self.scenario_manager.update(steering, accel)
        scenario_status, traffic_status, iou = self.scenario_manager.check_status(action)

        terminated = False
        truncated = False
        if scenario_status == ScenarioStatus.COMPLETED:
            terminated = True
        elif (scenario_status != ScenarioStatus.NORMAL) or (traffic_status != TrafficStatus.NORMAL):
            truncated = True

        reward = self._get_reward(scenario_status, traffic_status, iou)

        infos = self._get_infos(
            self.scenario_manager.agent.current_state, observations, scenario_status, traffic_status
        )

        return observations[0], reward, terminated, truncated, infos

    def render(self):
        if self.render_mode == "human":
            self.scenario_manager.render()

    def reset(self, seed: int = None, options: dict = None):
        """This function resets the environment.

        Args:
            seed (int, optional): The random seed.
            options (dict, optional): The options for the environment.
        Returns:
            observation (np.array): The BEV image observation of the environment.
            infos (dict): The information of the environment.
        """
        super().reset(seed=seed, options=options)
        self._max_iou = -np.inf
        self.scenario_manager.reset()

        observations = self.scenario_manager.get_observation()
        infos = self._get_infos(
            self.scenario_manager.agent.current_state,
            observations,
            ScenarioStatus.NORMAL,
            TrafficStatus.NORMAL,
        )
        self._min_dist_to_target = np.linalg.norm(
            np.array(
                [
                    self.scenario_manager.agent.current_state.x,
                    self.scenario_manager.agent.current_state.y,
                ]
            )
            - np.array(
                [
                    self.scenario_manager.target_area.geometry.centroid.x,
                    self.scenario_manager.target_area.geometry.centroid.y,
                ]
            )
        )

        return observations[0], infos

    class _ParkingScenarioManager(ScenarioManager):
        _max_steer = MAX_STEER
        _max_accel = MAX_ACCEL
        _lidar_range = 20
        _lidar_line = 360
        _window_size = (500, 500)
        _state_size = (200, 200)

        def __init__(
            self,
            type_proportion: float = 0.5,
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
            self.participants = {self.agent.id_: self.agent}

            self.map_ = Map(name="ParkingLot", scenario_type="parking")
            self.map_generator = ParkingLotGenerator(
                (self.agent.length, self.agent.width), type_proportion
            )
            self.start_state = None
            self.target_area = None
            self.target_heading = None
            self.cnt_step = 0

            self.render_manager = RenderManager(
                fps=self.render_fps, windows_size=self._window_size, off_screen=self.off_screen
            )

            # traffic event detectors
            self.status_checklist = {
                "time_exceed": TimeExceed(self.max_step),
                "no_action": NoAction(100),
                "out_bound": OutBound(),
                "collision": StaticCollision(),
                "completed": Arrival(),
            }

        def update(self, steering: float, accel: float):
            self.cnt_step += 1
            current_state = self.agent.current_state
            next_state, _, _ = self.agent.physics_model.step(current_state, accel, steering)
            self.agent.add_state(next_state)
            self.render_manager.update(self.participants, [0], self.agent.current_state.frame)

            return self.get_observation()

        def check_status(self, action: np.ndarray):
            scenario_status = ScenarioStatus.NORMAL
            traffic_status = TrafficStatus.NORMAL
            agent_pose = Polygon(self.agent.get_pose())

            is_time_exceed = self.status_checklist["time_exceed"].update()
            if is_time_exceed:
                scenario_status = ScenarioStatus.TIME_EXCEEDED
                return scenario_status, traffic_status, None

            is_no_action = self.status_checklist["no_action"].update(agent_pose)
            if is_no_action:
                traffic_status = ScenarioStatus.NO_ACTION
                return scenario_status, traffic_status, None

            is_out_bound = self.status_checklist["out_bound"].update(agent_pose)
            if is_out_bound:
                scenario_status = ScenarioStatus.OUT_BOUND
                return scenario_status, traffic_status, None

            is_collision = self.status_checklist["collision"].update(agent_pose)
            if is_collision:
                scenario_status = ScenarioStatus.FAILED
                traffic_status = TrafficStatus.COLLISION_STATIC
                return scenario_status, traffic_status, None

            is_completed, iou = self.status_checklist["completed"].update(agent_pose)
            if is_completed:
                scenario_status = ScenarioStatus.COMPLETED
                return scenario_status, traffic_status, iou

            return scenario_status, traffic_status, iou

        def render(self):
            self.render_manager.render()

        def reset(self):
            self.cnt_step = 0

            # reset map
            self.map_.reset()
            (self.start_state, self.target_area, self.target_heading) = self.map_generator.generate(
                self.map_
            )

            # reset agent
            self.agent.reset(self.start_state)

            # reset the sensors
            self.render_manager.reset()

            camera = TopDownCamera(
                id_=0,
                map_=self.map_,
                perception_range=(20, 20, 20, 20),
                window_size=self._state_size,
                off_screen=self.off_screen,
            )
            self.render_manager.add_sensor(camera)
            self.render_manager.bind(0, 0)

            lidar = SingleLineLidar(
                id_=1,
                map_=self.map_,
                perception_range=self._lidar_range,
                freq_detect=self._lidar_line * 10,
                window_size=self._state_size,
                off_screen=self.off_screen,
            )
            self.render_manager.add_sensor(lidar)
            self.render_manager.bind(1, 0)
            self.render_manager.update(self.participants, [0], self.agent.current_state.frame)

            # reset the status check list
            self.status_checklist["time_exceed"].reset()
            self.status_checklist["no_action"].reset()
            self.status_checklist["out_bound"].reset(self.map_.boundary)
            self.status_checklist["collision"].reset(
                [wall for wall in self.map_.areas.values() if wall.subtype != "target_area"]
            )
            self.status_checklist["completed"].reset(self.target_area)

    def close(self):
        """This function closes the environment."""
        self.scenario_manager.render_manager.close()
        super().close()
