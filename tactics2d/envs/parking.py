from typing import Union
import logging

import numpy as np
from shapely.geometry import Point, Polygon
import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import InvalidAction

from tactics2d.map.element import Map
from tactics2d.map.generator import ParkingLotGenerator
from tactics2d.participant.element import Vehicle
from tactics2d.physics import SingleTrackKinematics
from tactics2d.traffic import TrafficScenarioManager
from tactics2d.traffic.violation_detection import TrafficEvent
from tactics2d.sensor import TopDownCamera, SingleLineLidar, RenderManager
from tactics2d.trajectory.element import State

from tactics2d.participant.element.defaults import VEHICLE_MODEL

STATE_W = 200
STATE_H = 200
WIN_W = 500
WIN_H = 500

FPS = 60
MAX_FPS = 200
TIME_STEP = 0.01
MAX_STEP = 20000
MAX_SPEED = 2.0
MAX_STEER = 0.75
LIDAR_RANGE = 20.0
LIDAR_LINE = 120


def truncate_angle(angle: float):
    """Truncate angle to [-pi, pi]"""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi

    return angle


class ParkingScenarioManager(TrafficScenarioManager):
    """This class provides a parking scenario manager.

    Attributes:
        bay_proportion (float): The proportion of the parking bay in randomly generated
            parking scenarios.
        render_fps (int): The FPS of the rendering.
        off_screen (bool): Whether to render the scene on the screen.
        max_step (int, optional): The maximum number of steps. Defaults to 20000.
        step_size (float): The time duration of each step. Defaults to 0.5.
    """

    max_steer = MAX_STEER
    max_speed = MAX_SPEED
    lidar_line = LIDAR_LINE
    lidar_range = LIDAR_RANGE
    window_size = (WIN_W, WIN_H)
    state_size = (STATE_W, STATE_H)

    def __init__(
        self,
        bay_proportion: float,
        render_fps: int,
        off_screen: bool,
        max_step: float = MAX_STEP,
        step_size: float = 0.5,
    ):
        super().__init__(render_fps, off_screen, max_step, step_size)

        self.vehicle_configs = VEHICLE_MODEL["medium_car"]

        self.agent = Vehicle(
            id_=0,
            type_="medium_car",
            length=self.vehicle_configs["length"],
            width=self.vehicle_configs["width"],
            wheel_base=self.vehicle_configs["wheel_base"],
            speed_range=(-self.max_speed, self.max_speed),
            steer_range=(-self.max_steer, self.max_steer),
            accel_range=(-1.0, 1.0),
            physics_model=SingleTrackKinematics(
                dist_front_hang=0.5 * self.vehicle_configs["length"]
                - self.vehicle_configs["front_overhang"],
                dist_rear_hang=0.5 * self.vehicle_configs["length"]
                - self.vehicle_configs["rear_overhang"],
                steer_range=(-self.max_steer, self.max_steer),
                speed_range=(-self.max_speed, self.max_speed),
            ),
        )
        self.participants = {self.agent.id_: self.agent}

        self.map_ = Map(name="ParkingLot", scenario_type="parking")
        self.map_generator = ParkingLotGenerator(
            (self.agent.length, self.agent.width), bay_proportion
        )

        self.render_manager = RenderManager(
            fps=self.render_fps, windows_size=self.window_size, off_screen=self.off_screen
        )

        self.start_state: State = None
        self.target_area: Polygon = None
        self.target_heading: float = None

        self.cnt_still = 0

        self.iou_threshold = 0.95
        self.status_checklist = [
            # self._check_still,
            self._check_time_exceeded,
            self._check_collision,
            self._check_outbound,
            self._check_completed,
        ]

        self.dist_norm_factor = 10.0
        self.angle_norm_factor = np.pi

    def update(self, action: np.ndarray) -> TrafficEvent:
        self.n_step += 1
        self.agent.update(action, self.step_size)
        self.render_manager.update(self.participants, [0], self.agent.current_state.frame)

        return self.get_observation()

    def _check_still(self):
        if self.agent.speed < 0.1:
            self.cnt_still += 1
        else:
            self.cnt_still = 0

        if self.cnt_still >= self.render_fps:
            self.status = TrafficEvent.NO_ACTION

    def _check_collision(self):
        agent_pose = self.agent.get_pose()
        for _, area in self.map_.areas.items():
            if area.type_ == "obstacle" and agent_pose.intersects(area.geometry):
                self.status = TrafficEvent.COLLISION_STATIC
                break

    def _check_completed(self):
        agent_pose = Polygon(self.agent.get_pose())

        intersection = agent_pose.intersection(self.target_area.geometry)
        union = agent_pose.union(self.target_area.geometry)

        intersection_area = intersection.area
        union_area = union.area
        iou = intersection_area / union_area

        if iou >= self.iou_threshold:
            self.status = TrafficEvent.COMPLETED

    def reset(self):
        self.n_step = 0
        self.cnt_still = 0
        self.status = TrafficEvent.NORMAL
        # reset map
        self.map_.reset()
        (
            self.start_state,
            self.target_area,
            self.target_heading,
        ) = self.map_generator.generate(self.map_)

        # reset agent
        self.agent.reset(self.start_state)

        # reset the sensors
        self.render_manager.reset()

        camera = TopDownCamera(
            id_=0,
            map_=self.map_,
            perception_range=(20, 20, 20, 20),
            window_size=self.state_size,
            off_screen=self.off_screen,
        )
        self.render_manager.add_sensor(camera)
        self.render_manager.bind(0, 0)

        lidar = SingleLineLidar(
            id_=1,
            map_=self.map_,
            perception_range=self.lidar_range,
            freq_detect=self.lidar_line * 10,
            window_size=self.state_size,
            off_screen=self.off_screen,
        )
        self.render_manager.add_sensor(lidar)
        self.render_manager.bind(1, 0)
        self.render_manager.update(self.participants, [0], self.agent.current_state.frame)

        self.dist_norm_ratio = max(
            Point(self.start_state.location).distance(self.target_area.geometry.centroid),
            self.lidar_range,
        )


class ParkingEnv(gym.Env):
    """This class provides an environment to train ego vehicle to park in a parking lot without
    dynamic traffic participants, such as pedestrians and vehicles. The environment is randomly
    generated by calling `tactics2d:map:generator:ParkingLotGenerator`. The agent is
    required to park the vehicle in the target area. When the IoU between the agent and
    the target area is larger than 0.95, the agent is considered to be successfully parked.

    ## Action Space

    The action space is either continuous or discrete.

    When continuous, it is a Box(2,). The first action is steering. Its value range is
    [-0.75, 0.75]. The unit of steering is radian degree. The second action is the expected
    speed. Its value range is [-2, 2]. The unit of speed is $m/s$.

    When discrete, it is a Discrete(5). The action value is 0 (do nothing), 1 (steer left),
    2 (steer right), 3 (accelerate), 4 (decelerate), which means

    ## Observation Space

    The observation output is a RGB image of size [200, 200, 3].

    ## Status

    The status is a TrafficEvent. The possible values are (sorted by detection priority):
    - TrafficEvent.TIME_EXCEEDED: The simulation reaches the maximum time step.
    - TrafficEvent.COLLISION_STATIC: The agent collides with any obstacle.
    - TrafficEvent.OUTSIDE_MAP: The agent is outside the map boundary.
    - TrafficEvent.COMPLETED: The agent is successfully parked.
    - TrafficEvent.NO_ACTION: The agent does not take any action for 1 second.
    - TrafficEvent.NORMAL: The agent is in the normal state.

    ## Reward

    The reward is calculated as follows:
    - If the agent's status is TIME_EXCEEDED, the reward is -1.
    - If the agent's status is in [COLLISION_STATIC, OUTSIDE_MAP], the reward is -5.
    - If the agent's status is COMPLETED, the reward is 5.
    - Otherwise, the reward is calculated as the weighted sum of time penalty and IoU reward.

    Attributes:
        bay_proportion(float, optional): The proportion of the parking bay in the randomly
            generated environment. Defaults to 0.5.
        render_mode (str, optional): The rendering mode. Possible choices are "human" and
            "rgb_array". Defaults to "human".
        render_fps (int, optional): The rendering FPS. Defaults to 60.
        max_step (int, optional): The maximum number of steps. Defaults to 20000.
        continuous (bool, optional): Whether to use continuous action space. Defaults to True.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}
    max_fps = MAX_FPS
    max_steer = MAX_STEER
    max_speed = MAX_SPEED
    state_w = STATE_W
    state_h = STATE_H
    discrete_action = np.array(
        [
            [0, 0],  # do nothing
            [-0.3, 0],  # steer left
            [0.3, 0],  # steer right
            [0, 0.2],  # accelerate
            [0, -0.2],  # decelerate
        ]
    )

    def __init__(
        self,
        bay_proportion: float = 0.5,
        render_mode: str = "human",
        render_fps: int = FPS,
        max_step: int = MAX_STEP,
        continuous: bool = True,
    ):
        super().__init__()

        if render_mode not in self.metadata["render_modes"]:
            raise NotImplementedError(f"Render mode {render_mode} is not supported.")
        self.render_mode = render_mode

        if render_fps > self.max_fps:
            logging.warning(
                f"The input rendering FPS is too high. \
                            Set the FPS with the upper limit {self.max_fps}."
            )
        self.render_fps = min(render_fps, self.max_fps)

        self.max_step = max_step
        self.continuous = continuous

        if self.continuous:
            self.action_space = spaces.Box(
                np.array([-self.max_steer, -self.max_speed]),
                np.array([self.max_steer, self.max_speed]),
            )
        else:
            self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            0, 255, shape=(self.state_h, self.state_w, 3), dtype=np.uint8
        )

        self.max_iou = 0

        self.scenario_manager = ParkingScenarioManager(
            bay_proportion, self.render_fps, self.render_mode != "human", self.max_step
        )

    def _get_relative_pose(self, state: State):
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

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed, options=options)
        self.scenario_manager.reset()
        observations = self.scenario_manager.get_observation()
        observation = observations[0]

        diff_position, diff_angle, diff_heading = self._get_relative_pose(
            self.scenario_manager.agent.current_state
        )

        info = {
            "lidar": observations[1],
            "state": self.scenario_manager.agent.get_state(),
            "target_area": self.scenario_manager.target_area.geometry.exterior.coords[:-1],
            "target_heading": self.scenario_manager.target_heading,
            "diff_position": diff_position,
            "diff_angle": diff_angle,
            "diff_heading": diff_heading,
            "status": self.scenario_manager.status,
        }

        return observation, info

    def _get_reward(self, status: TrafficEvent):
        reward = 0
        if status == TrafficEvent.TIME_EXCEEDED:
            reward = -1
        elif status == TrafficEvent.COLLISION_STATIC:
            reward = -5
        elif status == TrafficEvent.OUTSIDE_MAP:
            reward = -5
        elif status == TrafficEvent.COMPLETED:
            reward = 5
        else:
            # time penalty
            time_penalty = -0.1 * np.tanh(
                self.scenario_manager.n_step / self.scenario_manager.max_step * 0.1
            )

            # IoU reward
            current_pose = Polygon(self.scenario_manager.agent.get_pose())
            target_pose = self.scenario_manager.target_area.geometry
            current_intersection = current_pose.intersection(target_pose).area
            current_union = current_pose.union(target_pose).area
            current_iou = current_intersection / current_union

            iou_reward = max(0, current_iou - self.max_iou)
            self.max_iou = max(self.max_iou, current_iou)

            reward = time_penalty + iou_reward

        return reward

    def step(self, action: Union[np.array, int]):
        if not self.action_space.contains(action):
            raise InvalidAction(f"Action {action} is invalid.")
        action = action if self.continuous else self.discrete_action[action]

        observations = self.scenario_manager.update(action)
        observation = observations[0]
        status = self.scenario_manager.check_status()
        terminated = status == TrafficEvent.COMPLETED
        truncated = status != TrafficEvent.NORMAL

        reward = self._get_reward(self.scenario_manager.status)

        diff_position, diff_angle, diff_heading = self._get_relative_pose(
            self.scenario_manager.agent.current_state
        )

        info = {
            "lidar": observations[1],
            "state": self.scenario_manager.agent.get_state(),
            "target_area": self.scenario_manager.target_area.geometry.exterior.coords[:-1],
            "target_heading": self.scenario_manager.target_heading,
            "diff_position": diff_position,
            "diff_angle": diff_angle,
            "diff_heading": diff_heading,
            "status": self.scenario_manager.status,
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self.scenario_manager.render()
