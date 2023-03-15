from typing import Union
import logging

logging.basicConfig(level=logging.WARNING)

import numpy as np
from shapely.geometry import Point
import gymnasium as gym
from gym import spaces, InvalidAction

from tactics2d.map.element import Map
from tactics2d.participant.element import Vehicle, Other
from tactics2d.trajectory.element import State
from tactics2d.map.generator import ParkingLotGenerator
from tactics2d.scenario import ScenarioManager, RenderManager
from tactics2d.scenario import TrafficEvent


STATE_W = 256
STATE_H = 256
VIDEO_W = 600
VIDEO_H = 400
WIN_W = 1000
WIN_H = 1000

FPS = 60
MAX_FPS = 200
TIME_STEP = 0.01  # state update time step: 0.01 s/step
MAX_STEP = 20000  # steps

DISCRETE_ACTION = np.array(
    [
        [0, 0],  # do nothing
        [-0.6, 0],  # steer left
        [0.6, 0],  # steer right
        [0, 0.2],  # accelerate
        [0, -0.2],  # decelerate
    ]
)


def truncate_angle(angle: float):
    """Truncate angle to [-pi, pi]"""

    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi

    return angle


class ParkingScenarioManager(ScenarioManager):
    def __init__(self, max_step: int):
        super().__init__(max_step)

        self.map_ = Map(name="ParkingLot", scenario_type="parking")
        self.map_generator = ParkingLotGenerator()
        self.agent = Vehicle(
            id_=0,
            type_="sedan",
            steering_angle_range=(-0.75, 0.75),
            steering_velocity_range=(-0.5, 0.5),
            speed_range=(-10, 100),
            accel_range=(-1, 1),
        )

        self.n_step = 0
        self.start_state: State = None
        self.target_state: State = None
        self.obstacles: list[Other] = []

        self.iou_threshold = 0.95
        self.status_checklist = [
            self._check_time_exceeded,
            self._check_collision,
            self._check_outbound,
            self._check_completed,
        ]

        self.dist_norm_factor = 10.0
        self.angle_norm_factor = np.pi

    def reset(self):
        self.map_.reset()
        self.obstacles = self.map_generator.generate()

        self.dist_norm_ratio = max(
            Point(self.start_state.location).distance(
                Point(self.target_state.location)
            ),
            10.0,
        )

        self.agent.reset(self.start_state)

    def update(self, action: np.ndarray) -> TrafficEvent:
        self.n_step += 1
        self.agent.update(action)

    def _check_collision(self):
        agent_pose = self.agent.get_pose()
        for obstacle in self.map.obstacles:
            if agent_pose.intersects(obstacle.get_pose()):
                self.status = TrafficEvent.COLLISION
                break

    def _check_completed(self):
        agent_pose = self.agent.get_pose()

        intersection = agent_pose.intersection(self.destination_area)
        union = agent_pose.union(self.destination_area)

        intersection_area = np.sum([p.area for p in intersection])
        union_area = np.sum([p.area for p in union])

        iou = intersection_area / union_area

        if iou >= self.iou_threshold:
            self.status = TrafficEvent.COMPLETED


class ParkingEnv(gym.Env):
    """

    Attributes:
        action_space (gym.spaces): The action space is either continuous or discrete.
            When continuous, it is a Box(2,). The first action is steering. Its value range is
            [-0.75, 0.75]. The second action is acceleration. Its value range is [-1, 1]. The unit of acceleration is $m^2/s$.
            When discrete, it is a Discrete(5). The action value is 0, 1, 2, 3, 4, which means
            - 0: do nothing
            - 1: steer left
            - 2: steer right
            - 3: accelerate
            - 4: decelerate
        observation_space ():
        render_mode
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        scenario: str = "bay",
        render_mode: str = "human",
        render_fps: int = FPS,
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

        self.continuous = continuous

        if self.continuous:
            self.action_space = spaces.Box(
                np.array([-0.75, -1.0]), np.array([0.75, 1.0]), dtype=np.float32
            )
        else:
            self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            0, 255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
        )

        self.scenario_manager = ParkingScenarioManager(MAX_STEP)
        self.render_manager = RenderManager()

    def reset(self, *, seed: int = None):
        super().reset(seed=seed)
        self.scenario_manager.reset()

    def _get_rewards(self):
        time_penalty = -np.tanh(
            self.scenario_manager.n_step / self.scenario_manager.max_step
        )

        curr_state = self.scenario_manager.agent.get_state()
        prev_frame = self.scenario_manager.agent.trajectory.frames[-2]
        prev_state = self.scenario_manager.agent.get_state(prev_frame)

        curr_dist = Point(curr_state.location).distance(
            Point(self.scenario_manager.target_state.location)
        )
        prev_dist = Point(prev_state.location).distance(
            Point(self.scenario_manager.target_state.location)
        )
        distance_reward = (
            prev_dist - curr_dist
        ) / self.scenario_manager.dist_norm_ratio

        curr_angle_diff = truncate_angle(
            curr_state.heading - self.scenario_manager.target_state.heading
        )
        prev_angle_diff = truncate_angle(
            prev_state.heading - self.scenario_manager.target_state.heading
        )
        angle_reward = (
            prev_angle_diff - curr_angle_diff
        ) / self.scenario_manager.angle_norm_factor

        return [time_penalty, distance_reward, angle_reward]

    def step(self, action: Union[np.array, int]):
        if not self.action_space.contains(action):
            raise InvalidAction(f"Action {action} is invalid.")
        action = action if self.continuous else DISCRETE_ACTION[action]

        self.scenario_manager.update(action)

        observation = self.render_manager.render(self.scenario_manager)

        status = self.scenario_manager.check_status()
        terminated = status == TrafficEvent.COMPLETED
        truncated = status != TrafficEvent.NORMAL

        rewards = self._get_rewards()
        total_reward = np.sum(rewards)

        info = {"status": status, "rewards": rewards}

        return observation, total_reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            return
        elif self.render_mode == "rgb_array":
            return


if __name__ == "__main__":
    a = np.array([0.0, 0.0])

    import pygame

    def register_input():
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    a[0] = -0.1
                if event.key == pygame.K_RIGHT:
                    a[0] = +0.1
                if event.key == pygame.K_UP:
                    a[1] = -0.1
                if event.key == pygame.K_DOWN:
                    a[1] = -0.1  # set 1.0 for wheels to block to zero rotation
                if event.key == pygame.K_RETURN:
                    global restart
                    restart = True
