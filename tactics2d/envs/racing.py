from typing import Union
import logging

import numpy as np
from shapely.geometry import LineString
import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import InvalidAction

from tactics2d.math.geometry import Vector
from tactics2d.map.element import Map
from tactics2d.participant.element import Vehicle
from tactics2d.trajectory.element import State
from tactics2d.sensor import TopDownCamera
from tactics2d.map.generator import RacingTrackGenerator
from tactics2d.scenario import ScenarioManager, RenderManager
from tactics2d.scenario import TrafficEvent


STATE_W = 200
STATE_H = 200
WIN_W = 500
WIN_H = 500

FPS = 60
MAX_FPS = 200
TIME_STEP = 0.01  # state update time step: 0.01 s/step
MAX_STEP = 20000  # steps

DISCRETE_ACTION = np.array(
    [
        [0, 0],  # do nothing
        [-0.3, 0],  # steer left
        [0.3, 0],  # steer right
        [0, 0.2],  # accelerate
        [0, -0.8],  # decelerate
    ]
)

THRESHOLD_NON_DRIVABLE = 0.5


class RacingScenarioManager(ScenarioManager):
    """_summary_

    Attributes:
        render_fps (int): The FPS of the rendering.
        off_screen (bool): Whether to render the scene on the screen.
        map_ (Map): The map of the scenario.
        map_generator (RacingTrackGenerator): The map generator.
        agent (Vehicle): The agent vehicle.
        render_manager (RenderManager): The render manager.
        n_step (int): The number of steps since the beginning of the episode.
        tile_visited (dict): The tiles that the agent has visited.
        tile_visited_cnt (int): The number of tiles that the agent has visited.
        tile_visiting (str): The id of the tile that the agent is visiting. As soon as a new tile
            is touched by the agent, it will be regarded as the visiting tile.
        start_line (LineString): The start line of the track.
        end_line (LineString): The end line of the track.
    """

    def __init__(self, render_fps: int, off_screen: bool, max_step: int):
        super().__init__(max_step)

        self.render_fps = render_fps
        self.off_screen = off_screen
        self.step_size = 1 / self.render_fps

        self.map_ = Map(name="RacingTrack", scenario_type="racing")
        self.map_generator = RacingTrackGenerator()

        self.agent = Vehicle(
            id_=0, type_="sedan", steer_range=(-0.5, 0.5), accel_range=(-1, 1)
        )

        self.render_manager = RenderManager(
            fps=self.render_fps, windows_size=(WIN_W, WIN_H), off_screen=self.off_screen
        )

        self.n_step = 0
        self.tile_visited = dict()
        self.tile_visiting = None
        self.intersect_proportion = 0
        self.start_line = None
        self.end_line = None

        self.status_checklist = [
            self._check_time_exceeded,
            self._check_retrograde,
            self._check_non_drivable,
            self._check_complete,
        ]

    @property
    def tile_visited_cnt(self) -> int:
        return sum(self.tile_visited.values())

    def locate_agent(self) -> float:
        """Locate the agent at a specific tile on the map."""
        tile_last_visited = self.tile_visiting
        agent_pose = self.agent.get_pose()

        if self.map_.lanes[tile_last_visited].contains(agent_pose):
            self.intersect_proportion = 1
        else:
            # mark all the tiles that the agent is touching as visiting
            tile_to_visit = self.map_.lanes[tile_last_visited].successors[0]
            tiles_visiting = []
            intersection_area = 0

            while tile_to_visit != tile_last_visited:
                if self.map_.lanes[tile_to_visit].intersects(
                    agent_pose
                ) or self.map_.lanes[tile_to_visit].contains(agent_pose):
                    tiles_visiting.append(tile_to_visit)
                    intersection_area += (
                        self.map_.lanes[tile_to_visit].intersection(agent_pose).area
                    )
                tile_to_visit = self.map_.lanes[tile_to_visit].successors[0]

            # update the tile_visited and tile_visiting
            if len(tiles_visiting) > 0:
                self.tile_visiting = tiles_visiting[-1]
                tile_to_visit = self.map_.lanes[tile_last_visited].successors[0]
                while tile_to_visit != self.tile_visiting:
                    self.tile_visited[tile_to_visit] = True
                    tile_to_visit = self.map_.lanes[tile_to_visit].successors[0]

            self.intersect_proportion = (
                intersection_area / self.map_.lanes[self.tile_visiting].area
            )

    def update(self, action: np.ndarray) -> TrafficEvent:
        """Update the state of the agent by the action instruction."""
        self.n_step += 1
        frame = self.n_step * 1000 // FPS

        self.agent.update(action)
        self._locate_agent()
        self.render_manager.update(self.participants, [0], frame)

    def _check_retrograde(self):
        successor = self.map_.lanes[self.tile_visiting].successors[0]
        if self.tile_visited[successor] and successor != "0000":
            self.status = TrafficEvent.VIOLATION_RETROGRADE
            return

        agent_pose = list(self.agent.get_pose().coords)
        vec_heading = [np.mean(agent_pose[:4]), np.mean(agent_pose[:2])]
        tile_left_side = list(self.map_.lanes[self.tile_visiting].left_side.coords)
        tile_right_side = list(self.map_.lanes[self.tile_visiting].right_side.coords)

        angle1 = Vector.angle(vec_heading, tile_left_side)
        angle2 = Vector.angle(vec_heading, tile_right_side)

        if angle1 > np.pi / 2 and angle2 > np.pi / 2:
            self.status = TrafficEvent.VIOLATION_RETROGRADE

    def _check_non_drivable(self):
        if self.intersect_proportion < THRESHOLD_NON_DRIVABLE:
            self.status = TrafficEvent.VIOLATION_NON_DRIVABLE

    def _check_completed(self):
        if self.tile_visited_cnt == self.n_tile:
            self.status = TrafficEvent.COMPLETED

    def _reset_map(self):
        self.map_.reset()
        self.map_generator.generate(self.map_)

        start_tile = self.map.lanes["0000"]
        for tile_id in self.map_.lanes:
            self.tile_visited[tile_id] = False
        self.tile_visiting = start_tile.id_
        self.tile_visited[self.tile_visiting] = True
        self.n_tile = len(self.map_.lanes)

        self.start_line = LineString(start_tile.get_ends())
        self.end_line = LineString(start_tile.get_starts())

    def _reset_agent(self):
        vec = self.start_line[1] - self.start_line[0]
        heading = np.arctan2(-vec[1], vec[0])
        start_loc = np.mean(
            self.start_line, axis=0
        ) - self.agent.length / 2 / np.linalg.norm(vec) * np.array([-vec[1], vec[0]])
        state = State(
            self.n_step, heading=heading, x=start_loc[0], y=start_loc.y[1], vx=0, vy=0
        )

        self.agent.reset(state)
        logging.info(
            "The racing car starts at (%.3f, %.3f), heading to %.3f rad."
            % (start_loc.x, start_loc.y, heading)
        )

    def reset(self):
        self.n_step = 0
        self.status = TrafficEvent.NORMAL
        self._reset_map()
        self._reset_agent()
        self.render_manager.reset()

        camera = TopDownCamera(
            id_=0,
            map_=self.scenario_manager.map_,
            perception_range=(60, 60, 100, 20),
            window_size=(STATE_W, STATE_H),
            off_screen=self.off_screen,
        )
        self.render_manager.add_sensor(camera)
        self.render_manager.bind(0, 0)


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

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        render_mode: str = "human",
        render_fps: int = FPS,
        max_step: int = MAX_STEP,
        continuous: bool = True,
    ):
        super().__init__()

        if render_mode not in self.metadata["render_modes"]:
            raise NotImplementedError(f"Render mode {render_mode} is not supported.")
        self.render_mode = render_mode

        if render_fps > MAX_FPS:
            logging.warning(
                f"The input rendering FPS is too high. \
                            Set the FPS with the upper limit {MAX_FPS}."
            )
        self.render_fps = min(render_fps, MAX_FPS)

        self.max_step = max_step
        self.continuous = continuous

        if self.continuous:
            self.action_space = spaces.Box(
                np.array([-0.5, -1.0]), np.array([0.5, 1.0]), dtype=np.float32
            )
        else:
            self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            0, 255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
        )

        self.scenario_manager = RacingScenarioManager(
            self.render_fps, self.render_mode != "human", self.max_step
        )

    def reset(self, seed: int = None):
        super().reset(seed=seed)
        self.scenario_manager.reset()

    def _get_reward(self):
        return

    def step(self, action: Union[np.array, int]):
        if not self.action_space.contains(action):
            raise InvalidAction(f"Action {action} is invalid.")
        action = action if self.continuous else DISCRETE_ACTION[action]

        self.scenario_manager.update(action)
        status = self.scenario_manager.check_status()
        done = status == TrafficEvent.COMPLETED

        observation = self._get_observation()
        reward = self._get_reward()

        info = {"status": status}

        return observation, reward, done, info

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
