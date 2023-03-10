from typing import Union
import logging
logging.basicConfig(level=logging.WARNING)

import numpy as np
from shapely.geometry import Point, LineString, LinearRing

import gymnasium as gym
from gym import spaces

from tactics2d.map.element import Map
from tactics2d.participant.element import Vehicle
from tactics2d.trajectory.element import State
from tactics2d.map.generator import RacingTrackGenerator
from tactics2d.scenario import TrafficEvent

STATE_W = 128
STATE_H = 128
VIDEO_W = 600
VIDEO_H = 400
WIN_W = 1000
WIN_H = 1000

FPS = 60
MAX_FPS = 200
STEP_LIMIT = 20000  # steps
TIME_STEP = 0.01  # state update time step: 0.01 s/step

DISCRETE_ACTION = np.array(
    [
        [0, 0],  # do nothing
        [-0.3, 0],  # steer left
        [0.3, 0],  # steer right
        [0, 0.2],  # accelerate
        [0, -0.8],  # decelerate
    ]
)


class RacingEnv(gym.Env):
    """An improved version of Box2D's CarRacing gym environment.

    Attributes:
        action_space (gym.spaces): The action space is either continuous or discrete.
            When continuous, it is a Box(2,). The first action is steering. Its value range is 
            [-0.5, 0.5]. The unit of steering is radius. The second action is acceleration. 
            Its value range is [-1, 1]. The unit of acceleration is $m^2/s$.
            When discrete, it is a Discrete(5). The action space is defined above:
                -  0: do nothing
                -  1: steer left
                -  2: steer right
                -  3: accelerate
                -  4: decelerate
        observation_space (gym.spaces): The observation space is represented as a top-down
            view 128x128 RGB image of the car and the race track. It is a Box(128, 128, 3).
        render_mode (str, optional): The rendering mode. Possible choices are "human" or 
            "rgb_array". Defaults to "human".
        render_fps (int, optional): The expected FPS for rendering. Defaults to 60.
        continuous (bool, optional): Whether to use continuous action space. Defaults to True.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
    }

    def __init__(
        self, render_mode: str = "human", render_fps: int = FPS, continuous: bool = True
    ):

        if render_mode not in self.metadata["render_modes"]:
            raise NotImplementedError(f"Render_mode {render_mode} is not supported.")
        else:
            self.render_mode = render_mode

        if render_fps > MAX_FPS:
            self.render_fps = MAX_FPS
            logging.warning(f"The input rendering FPS is too high. \
                            Set the FPS with the upper limit {MAX_FPS}.")
        else:
            self.render_fps = render_fps

        self.continuous = continuous

        if self.continuous:
            self.action_space = spaces.Box(
                np.array([-0.5, -1.0]).astype(np.float32),
                np.array([0.5, 1.0]).astype(np.float32),
            )
        else:
            self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(STATE_H, STATE_W, 3), 
            dtype=np.uint8
        )

        self.map_ = Map(name="CarRacing", scenario_type="racing")
        self.map_generator = RacingTrackGenerator()
        self.agent = Vehicle(
            id_=0, type_="sports_car",
            steering_angle_range=(-0.5, 0.5),
            steering_velocity_range=(-0.5, 0.5),
            speed_range=(-10, 100), accel_range=(-1, 1),
        )
        self.step_interval = int(1000 / self.render_fps)

    def _reset_map(self):
        """Generate a new map with random track configurations."""

        self.map_.reset()
        self.map_.lanes = self.map_generator.generate_map()

        self.n_tile = len(self.map_.lanes)
        self.tile_visited = [False] * self.n_tile
        self.tile_visited_cnt = 0
        self.start_tile = self.map.lanes["0000"]
        self.start_line = LineString(self.start_tile.get_ends())
        self.finish_line = LineString(self.start_tile.get_starts())

    def _reset_vehicle(self):
        # get the initial state
        vec = self.start_line[1] - self.start_line[0]
        heading = np.arctan2(-vec[1], vec[0])
        start_loc = np.mean(self.start_line, axis=0) \
            - self.agent.length / 2 / np.linalg.norm(vec) * np.array([-vec[1], vec[0]])
        state = State(
            self.n_step, heading=heading,
            x=start_loc[0], y=start_loc.y[1], vx=0, vy=0,
        )

        #reset the vehicle with the initial state
        self.agent.reset(state)
        logging.info(
            "The racing car starts at (%.3f, %.3f), heading to %.3f rad." \
                % (start_loc.x, start_loc.y, heading)
        )

    def reset(self, seed: int = None):

        super().reset(seed=seed)

        self.n_step = 0

        self._reset_map()
        self._reset_vehicle()

    def _get_reward(self):
        return

    def step(self, action: Union[np.array, int]):
        if not self.continuous:
            action = DISCRETE_ACTION[action]

        self.n_step += 1

        observation = self._get_observation()
        status = self._check_status()
        reward = self._get_reward()

        info = {
            "status": status,
        }

        # return observation, reward, done, info

    def render(self):
        if self.render_mode == "human":
            return
        elif self.render_mode == "rgb_array":
            return


if __name__ == "__main__":
    action = np.array([0.0, 0.0])

    import pygame

    def register_input():
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action[0] = -0.1
                if event.key == pygame.K_RIGHT:
                    action[0] = +0.1
                if event.key == pygame.K_UP:
                    action[1] = -0.1
                if event.key == pygame.K_DOWN:
                    # set 1.0 for wheels to block to zero rotation
                    action[1] = -0.1
                if event.key == pygame.K_RETURN:
                    global restart
                    restart = True
