##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: racing.py
# @Description: This script defines a racing environment.
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
from tactics2d.map.generator import RacingTrackGenerator
from tactics2d.participant.element import Vehicle
from tactics2d.participant.trajectory import State
from tactics2d.physics import SingleTrackKinematics
from tactics2d.sensor import RenderManager, TopDownCamera
from tactics2d.traffic import ScenarioManager, ScenarioStatus, TrafficStatus
from tactics2d.traffic.event_detection import NoAction, OffLane, OutBound, TimeExceed

MAX_STEER = 1.0
MAX_ACCEL = 2.0
MIN_ACCEL = -4.0


class RacingEnv(gym.Env):
    """This class provides an environment to train a racing car to drive on a racing track.

    ## Observation

    `RacingEnv` provides a top-down semantic segmentation image of agent and its surrounding. The observation is represented as a

    ## Action

    `RacingEnv` accepts either a continuous or a discrete action command for the agent vehicle.

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

    1. Check time exceed: If the time step exceeds the maximum time step (100000 steps), the scenario status will be set to `TIME_EXCEEDED` and a negative reward -1 will be given.
    2. Check no action: If the agent vehicle does not move for over 100 steps, the scenario status will be set to `NO_ACTION` and a negative reward -1 will be given.
    3. Check out bound: If the agent vehicle goes out of the boundary of the map, the scenario status will be set to `OUT_BOUND` and a negative reward -5 will be given.
    4. Check off road: If the agent vehicle drives off the road, the scenario status will be set to `OFF_ROAD` and a negative reward -5 will be given.
    5. Check arrived: If the agent vehicle arrives at the destination, the scenario status will be set to `ARRIVED` and a positive reward 10 will be given.
    6. Otherwise,

    If the agent has successfully completed the scenario, the environment will set the terminated flag to True. If the scenario status goes abnormal or the traffic status goes abnormal, the environment will set the truncated flag to True.

    The status information is returned as a dictionary with the following keys:

    - `state`: The current state of the agent vehicle.
    - `traffic_status`: The status of the traffic scenario.
    - `scenario_status`: The status of the scenario.
    """

    _metadata = {"render_modes": ["human", "rgb_array"]}
    _max_fps = 200
    _max_steer = MAX_STEER
    _max_accel = MAX_ACCEL
    _min_accel = MIN_ACCEL
    _discrete_actions = {1: (0, 0), 2: (-0.5, 0), 3: (0.5, 0), 4: (0, 1), 5: (0, -1)}

    def __init__(
        self,
        render_mode: str = "human",
        render_fps: int = 60,
        max_step: int = int(1e5),
        continuous: bool = True,
    ):
        """Initialize the racing environment.

        Args:
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
                np.array([-self._max_steer, self._min_accel]),
                np.array([self._max_steer, self._max_accel]),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Discrete(5)

        self.scenario_manager = self._RacingScenarioManager(
            self.max_step, 100, self.render_fps, off_screen=self.render_mode != "human"
        )

    def _get_rewards(self, scenario_status: ScenarioStatus, traffic_status: TrafficStatus):
        num_tile = self.scenario_manager.num_tile
        cnt_step = self.scenario_manager.cnt_step

        if (
            scenario_status == ScenarioStatus.TIME_EXCEEDED
            or scenario_status == ScenarioStatus.NO_ACTION
        ):
            reward = -1
        elif traffic_status == ScenarioStatus.OUT_BOUND or traffic_status == TrafficStatus.OFF_LANE:
            reward = -5
        elif scenario_status == ScenarioStatus.COMPLETED:
            reward = (num_tile - 0.1 * cnt_step) / num_tile * 100
        else:
            time_penalty = -0.1 * cnt_step
            tile_reward = 0.1 * self.scenario_manager.num_visited_tile
            reward = time_penalty + tile_reward

        return reward

    def step(self, action: Union[tuple, int]):
        """This function takes a step in the environment.

        Args:
            action (Union[tuple, int]): The action command for the agent vehicle. If the action space is continuous, the input should be a tuple, whose first element controls the steering value and the second controls the acceleration. If the action space is discrete, the input should be an index that points to a pre-defined control command.

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
        observation = self.scenario_manager.update(steering, accel)
        scenario_status, traffic_status = self.scenario_manager.check_status(action)

        terminated = False
        truncated = False
        if scenario_status == ScenarioStatus.COMPLETED:
            terminated = True
        elif (scenario_status != ScenarioStatus.NORMAL) or (traffic_status != TrafficStatus.NORMAL):
            truncated = True

        reward = self._get_rewards(scenario_status, traffic_status)

        infos = {
            "state": self.scenario_manager.agent.current_state,
            "traffic_status": traffic_status,
            "scenario_status": scenario_status,
        }

        return observation, reward, terminated, truncated, infos

    def render(self):
        if self.render_mode == "human":
            self.scenario_manager.render()

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed, options=options)
        self.scenario_manager.reset()
        observations = self.scenario_manager.get_observation()
        observation = observations[0]

        infos = {
            "state": self.scenario_manager.agent.current_state,
            "traffic_status": TrafficStatus.NORMAL,
            "scenario_status": ScenarioStatus.NORMAL,
        }

        return observation, infos

    class _RacingScenarioManager(ScenarioManager):
        _max_steer = MAX_STEER
        _max_accel = MAX_ACCEL
        _min_accel = MIN_ACCEL
        _window_size = (500, 500)
        _state_size = (200, 200)

        def __init__(
            self,
            max_step: int,
            step_size: float = None,
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
                speed_range=self.agent.speed_range,
                accel_range=(self._min_accel, self._max_accel),
                interval=self.step_size,
            )
            self.participants = {self.agent.id_: self.agent}

            self.map_ = Map(name="racing_track", scenario_type="racing")
            self.map_generator = RacingTrackGenerator()

            self.render_manager = RenderManager(
                fps=self.render_fps, windows_size=self._window_size, off_screen=self.off_screen
            )

            self.tile_visited = dict()
            self.tile_visiting = None
            self.start_line = None
            self.end_line = None
            self.cnt_step = 0

            # traffic event detectors
            self.status_checklist = {
                "time_exceed": TimeExceed(max_step),
                "no_action": NoAction(100),
                "out_bound": OutBound(),
                "off_road": OffLane(),
            }

        @property
        def num_tile(self):
            return len(self.map_.lanes)

        @property
        def num_visited_tile(self):
            return sum(self.tile_visited.values())

        def _locate_agent(self) -> float:
            if len(self.tile_visited) == 0 or self.tile_visiting is None:
                raise ValueError("The map is not initialized.")

            agent_pose = Polygon(self.agent.get_pose())
            tile_last_visited = self.tile_visiting
            tile_to_check_id = tile_last_visited
            tiles_visiting = []

            # file all the tiles that the agent is touching
            tile_to_check = self.map_.lanes[tile_to_check_id]
            tile_shape = tile_to_check.geometry
            if tile_shape.intersects(agent_pose) or tile_shape.contains(agent_pose):
                tiles_visiting.append(tile_to_check_id)
            tile_to_check_id = list(tile_to_check.successors)[0]
            while tile_to_check_id != tile_last_visited:
                tile_to_check = self.map_.lanes[tile_to_check_id]
                tile_shape = tile_to_check.geometry
                if tile_shape.intersects(agent_pose) or tile_shape.contains(agent_pose):
                    tiles_visiting.append(tile_to_check_id)
                tile_to_check_id = list(tile_to_check.successors)[0]

                # assume that tiles the agent touches are connected
                if (
                    len(tiles_visiting) != 0
                    and tile_to_check_id not in self.map_.lanes[tiles_visiting[-1]].successors
                ):
                    break

            # assume that all the tiles between the last visited tile and the current visiting tile are visited
            # mark all the tiles that the agent has touched as visiting
            tile_to_check_id = list(self.map_.lanes[tile_last_visited].successors)[0]
            if len(tiles_visiting) > 0:
                while tile_to_check_id not in tiles_visiting:
                    tile_to_check = self.map_.lanes[tile_to_check_id]
                    self.tile_visited[tile_to_check_id] = True
                    tile_to_check_id = list(tile_to_check.successors)[0]
                for tile_id in tiles_visiting:
                    self.tile_visited[tile_id] = True

                self.tile_visiting = tiles_visiting[-1]

        def _reset_map(self):
            self.map_.reset()
            self.map_generator.generate(self.map_)

            start_tile = self.map_.lanes["0000"]
            for tile_id in self.map_.lanes:
                self.tile_visited[tile_id] = False
            self.tile_visiting = start_tile.id_
            self.tile_visited[start_tile.id_] = True
            self.n_tile = len(self.map_.lanes)

        def _reset_agent(self):
            start_line = np.array(self.map_.roadlines["start_line"].shape)
            vec = start_line[1] - start_line[0]
            heading = np.arctan2(vec[0], -vec[1])
            start_loc = np.mean(start_line, axis=0)
            start_loc -= self.agent.length / 2 / np.linalg.norm(vec) * np.array([-vec[1], vec[0]])
            state = State(0, heading=heading, x=start_loc[0], y=start_loc[1], vx=0, vy=0, accel=0)

            self.agent.reset(state)
            logging.info(
                "The racing car starts at (%.3f, %.3f), heading to %.3f rad."
                % (start_loc[0], start_loc[1], heading)
            )

        def update(self, steering: float, accel: float):
            self.cnt_step += 1
            current_state = self.agent.current_state
            next_state, _, _ = self.agent.physics_model.step(current_state, accel, steering)
            self.agent.add_state(next_state)

            self._locate_agent()
            self.render_manager.update(self.participants, [0], self.agent.current_state.frame)

            return self.get_observation()

        def check_status(self, action: np.ndarray):
            scenario_status = ScenarioStatus.NORMAL
            traffic_status = TrafficStatus.NORMAL
            agent_pose = Polygon(self.agent.get_pose())

            is_time_exceed = self.status_checklist["time_exceed"].update()
            if is_time_exceed:
                scenario_status = ScenarioStatus.TIME_EXCEEDED
                return scenario_status, traffic_status

            is_no_action = self.status_checklist["no_action"].update(agent_pose)
            if is_no_action:
                traffic_status = ScenarioStatus.NO_ACTION
                return scenario_status, traffic_status

            is_out_bound = self.status_checklist["out_bound"].update(agent_pose)
            if is_out_bound:
                traffic_status = ScenarioStatus.OUT_BOUND
                return scenario_status, traffic_status

            is_off_road = self.status_checklist["off_road"].update(agent_pose)
            if is_off_road:
                traffic_status = TrafficStatus.OFF_ROAD
                return scenario_status, traffic_status

            # regard the scenario as accomplished if all the tiles are visited
            if all(self.tile_visited.values()):
                scenario_status = ScenarioStatus.COMPLETED
                return scenario_status, traffic_status

            return scenario_status, traffic_status

        def render(self):
            self.render_manager.render()

        def reset(self):
            self.tile_visited.clear()
            self.tile_visiting = None
            self.start_line = None
            self.end_line = None
            self.cnt_step = 0

            self._reset_map()
            self._reset_agent()
            self.render_manager.reset()

            # reset the sensors
            camera = TopDownCamera(
                id_=0,
                map_=self.map_,
                perception_range=(30, 30, 50, 10),
                window_size=self._state_size,
                off_screen=self.off_screen,
            )
            self.render_manager.add_sensor(camera)
            self.render_manager.bind(0, 0)

            # reset the status check list
            self.status_checklist["time_exceed"].reset()
            self.status_checklist["no_action"].reset()
            self.status_checklist["out_bound"].reset(self.map_.boundary)
            self.status_checklist["off_road"].reset(self.map_.lanes)

            return self.scenario_status, self.traffic_status
