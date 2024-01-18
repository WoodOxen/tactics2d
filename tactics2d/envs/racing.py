from typing import Union
import logging

import numpy as np
from shapely.geometry import Polygon
import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import InvalidAction

from tactics2d.map.element import Map
from tactics2d.participant.element import Vehicle
from tactics2d.trajectory.element import State
from tactics2d.sensor import TopDownCamera, RenderManager
from tactics2d.map.generator import RacingTrackGenerator
from tactics2d.traffic import ScenarioManager
from tactics2d.traffic.violation_detection import TrafficEvent


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
        [0, 0.5],  # accelerate
        [0, -2.0],  # decelerate
    ]
)

THRESHOLD_NON_DRIVABLE = 0.5


class RacingScenarioManager(TrafficScenarioManager):
    """_summary_

    Attributes:
        agent (Vehicle): The agent vehicle.
        map_ (Map): The map of the scenario.
        map_generator (RacingTrackGenerator): The map generator.
        render_manager (RenderManager): The render manager.
        render_fps (int): The FPS of the rendering.
        off_screen (bool): Whether to render the scene on the screen.
        n_step (int): The number of steps since the beginning of the episode.
        tile_visited (dict): The tiles that the agent has visited.
        tile_visited_cnt (int): The number of tiles that the agent has visited.
        tiles_visiting (str): The ids of the tiles that the agent is visiting. As soon as a new tile
            is touched by the agent, it will be regarded as the visiting tile.
    """

    def __init__(self, render_fps: int, off_screen: bool, max_step: int):
        super().__init__(render_fps, off_screen, max_step)

        self.map_ = Map(name="RacingTrack", scenario_type="racing")
        self.map_generator = RacingTrackGenerator()

        self.agent = Vehicle(
            id_=0, type_="medium_car", steer_range=(-0.5, 0.5), accel_range=(-2.0, 2.0)
        )
        self.participants = {self.agent.id_: self.agent}

        self.render_manager = RenderManager(
            fps=self.render_fps, windows_size=(WIN_W, WIN_H), off_screen=self.off_screen
        )

        self.tile_visited = dict()
        self.tiles_visiting = None
        self.start_line = None
        self.end_line = None

        self.cnt_still = 0

        self.status_checklist = [
            self._check_still,
            self._check_time_exceeded,
            self._check_retrograde,
            self._check_non_drivable,
            self._check_completed,
        ]

    @property
    def tile_visited_cnt(self) -> int:
        return sum(self.tile_visited.values())

    def locate_agent(self) -> float:
        """Locate the agent at a specific tile on the map."""

        if len(self.tiles_visiting) == 0:
            return

        agent_pose = Polygon(self.agent.get_pose())
        tile_last_visited = self.tiles_visiting[0]
        tile_to_visit = tile_last_visited
        tiles_visiting = []

        # mark all the tiles that the agent has touched as visiting
        for tile_id in self.tiles_visiting:
            tile_shape = Polygon(self.map_.lanes[tile_id].geometry)
            if tile_shape.intersects(agent_pose) or tile_shape.contains(agent_pose):
                tiles_visiting.append(tile_id)

        tiles_visiting_set = set(tiles_visiting)
        tile_to_visit = list(self.map_.lanes[tile_last_visited].successors)[0]
        while tile_to_visit != tile_last_visited:
            tile_shape = Polygon(self.map_.lanes[tile_to_visit].geometry)
            if tile_shape.contains(agent_pose):
                if tile_to_visit not in tiles_visiting_set:
                    tiles_visiting.append(tile_to_visit)
                break
            if tile_shape.intersects(agent_pose):
                if tile_to_visit not in tiles_visiting_set:
                    tiles_visiting.append(tile_to_visit)

            tile_to_visit = list(self.map_.lanes[tile_to_visit].successors)[0]

        # update the tile_visited and tiles_visiting
        self.tiles_visiting = tiles_visiting

        if len(self.tiles_visiting) == 0:
            return

        if len(self.tiles_visiting) > 0 and tile_last_visited == self.tiles_visiting[-1]:
            return

        tile_to_visit = list(self.map_.lanes[tile_last_visited].successors)[0]
        while tile_to_visit != self.tiles_visiting[-1]:
            self.tile_visited[tile_to_visit] = True
            tile_to_visit = list(self.map_.lanes[tile_to_visit].successors)[0]

        self.tile_visited[self.tiles_visiting[-1]] = True

    def update(self, action: np.ndarray) -> TrafficEvent:
        """Update the state of the agent by the action instruction."""
        self.n_step += 1
        self.agent.update(action, self.step_len)
        self.locate_agent()
        self.render_manager.update(self.participants, [0], self.agent.current_state.frame)

        return self.get_observation()

    def _check_still(self):
        if self.agent.speed < 0.1:
            self.cnt_still += 1
        else:
            self.cnt_still = 0

        if self.cnt_still >= self.render_fps:
            self.status = TrafficEvent.NO_ACTION

    def _check_retrograde(self):
        successor = list(self.map_.lanes[self.tiles_visiting[-1]].successors)[0]

        if self.tile_visited[successor] and successor != "0000":
            self.status = TrafficEvent.VIOLATION_RETROGRADE
            return

        heading = self.agent.heading

        idx = int(len(self.tiles_visiting) / 2)
        tile_left_side = np.array(list(self.map_.lanes[self.tiles_visiting[idx]].left_side.coords))
        left_vec = tile_left_side[1] - tile_left_side[0]
        left_angle = np.arctan2(left_vec[1], left_vec[0])
        tile_right_side = np.array(
            list(self.map_.lanes[self.tiles_visiting[idx]].right_side.coords)
        )
        right_vec = tile_right_side[1] - tile_right_side[0]
        right_angle = np.arctan2(right_vec[1], right_vec[0])

        angle1 = np.abs(left_angle - heading)
        angle2 = np.abs(right_angle - heading)

        if all([angle1, angle2]) > np.pi / 2:
            self.status = TrafficEvent.VIOLATION_RETROGRADE

    def _check_non_drivable(self):
        intersection_area = 0
        agent_pose = Polygon(self.agent.get_pose())
        for tile_id in self.tiles_visiting:
            tile_shape = Polygon(self.map_.lanes[tile_id].geometry)
            intersection_area += tile_shape.intersection(agent_pose).area

        intersect_proportion = intersection_area / agent_pose.area
        if intersect_proportion < THRESHOLD_NON_DRIVABLE:
            self.status = TrafficEvent.VIOLATION_NON_DRIVABLE

    def _check_completed(self):
        if self.tile_visited_cnt == self.n_tile:
            self.status = TrafficEvent.COMPLETED

    def _reset_map(self):
        self.map_.reset()
        self.map_generator.generate(self.map_)

        start_tile = self.map_.lanes["0000"]
        for tile_id in self.map_.lanes:
            self.tile_visited[tile_id] = False
        self.tiles_visiting = [start_tile.id_]
        self.tile_visited[start_tile.id_] = True
        self.n_tile = len(self.map_.lanes)

    def _reset_agent(self):
        start_line = np.array(self.map_.roadlines["start_line"].shape)
        vec = start_line[1] - start_line[0]
        heading = np.arctan2(vec[0], -vec[1])
        start_loc = np.mean(start_line, axis=0)
        start_loc -= self.agent.length / 2 / np.linalg.norm(vec) * np.array([-vec[1], vec[0]])
        state = State(
            self.n_step, heading=heading, x=start_loc[0], y=start_loc[1], vx=0, vy=0, accel=0
        )

        self.agent.reset(state)
        logging.info(
            "The racing car starts at (%.3f, %.3f), heading to %.3f rad."
            % (start_loc[0], start_loc[1], heading)
        )

    def reset(self):
        self.n_step = 0
        self.cnt_still = 0
        self.status = TrafficEvent.NORMAL

        self._reset_map()
        self._reset_agent()
        self.render_manager.reset()

        camera = TopDownCamera(
            id_=0,
            map_=self.map_,
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
        render_mode (str, optional): The rendering mode. Possible choices are "human" and
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
            self.action_space = spaces.Box(np.array([-0.5, -2.0]), np.array([0.5, 2.0]))
        else:
            self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(0, 255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)

        self.scenario_manager = RacingScenarioManager(
            self.render_fps, self.render_mode != "human", self.max_step
        )

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed, options=options)
        self.scenario_manager.reset()
        observations = self.scenario_manager.get_observation()
        observation = observations[0]

        info = {
            "velocity": self.scenario_manager.agent.velocity,
            "acceleration": self.scenario_manager.agent.accel,
            "heading": self.scenario_manager.agent.heading,
            "status": self.scenario_manager.status,
            "rewards": dict(),
            "reward": 0,
        }

        return observation, info

    def _get_rewards(self, status: TrafficEvent):
        # penalty for no action
        no_action_penalty = 0 if status != TrafficEvent.NO_ACTION else -200

        # penalty for time exceed
        time_exceeded_penalty = 0 if status != TrafficEvent.TIME_EXCEEDED else -100

        # penalty for driving into non drivable area
        non_drivable_penalty = 0 if status != TrafficEvent.VIOLATION_NON_DRIVABLE else -100

        # penalty for driving in the opposite direction
        retrograde_penalty = 0 if status != TrafficEvent.VIOLATION_RETROGRADE else -50

        # reward for longitudinal speed
        speed = self.scenario_manager.agent.speed
        heading = self.scenario_manager.agent.heading
        speed_reward = speed * np.cos(heading)

        # reward for completion
        complete_reward = 0 if status != TrafficEvent.COMPLETED else 200

        rewards = {
            "no action penalty": no_action_penalty,
            "time exceeded penalty": time_exceeded_penalty,
            "non_drivable": non_drivable_penalty,
            "retrograde": retrograde_penalty,
            "speed_reward": speed_reward,
            "completed": complete_reward,
        }

        if status not in [TrafficEvent.NORMAL, TrafficEvent.COMPLETED]:
            reward = sum(rewards.values()) - speed_reward
        else:
            reward = sum(rewards.values())

        return rewards, reward

    def step(self, action: Union[np.array, int]):
        if not self.action_space.contains(action):
            raise InvalidAction(f"Action {action} is invalid.")
        action = action if self.continuous else DISCRETE_ACTION[action]

        observations = self.scenario_manager.update(action)
        observation = observations[0]

        status = self.scenario_manager.check_status()
        terminated = status == TrafficEvent.COMPLETED
        truncated = status != TrafficEvent.NORMAL

        rewards, reward = self._get_rewards(status)

        info = {
            "velocity": self.scenario_manager.agent.velocity,
            "acceleration": self.scenario_manager.agent.accel,
            "heading": self.scenario_manager.agent.heading,
            "status": status,
            "rewards": rewards,
            "reward": reward,
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self.scenario_manager.render()
