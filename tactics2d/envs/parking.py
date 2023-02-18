from typing import Optional, Union
import math
from typing import OrderedDict
from enum import Enum

import numpy as np
import gym
from gym import spaces
from gym.error import DependencyNotInstalled, InvalidAction
from shapely.geometry import LineString, LinearRing, Polygon
from shapely.affinity import affine_transform
try:
    # As pygame is necessary for using the environment (reset and step) even without a render mode
    #   therefore, pygame is a necessary import for the environment.
    import pygame
except ImportError:
    raise DependencyNotInstalled(
        "pygame is not installed, run `pip install pygame`"
    )

from tactics2d.map.element.area import Area
from tactics2d.map.element.map import Map
from tactics2d.map.generator.generate_parking_lot import Position, \
    gene_bay_park, gene_parallel_park, VEHICLE_BOX
from tactics2d.render.lidar_simulator import LidarSimlator
# from tactics2d.traffic.traffic_event import TrafficEvent
from tactics2d.participant.element import Vehicle
from tactics2d.trajectory.element import State


WHEEL_BASE = 2.8  # wheelbase
FRONT_HANG = 0.96  # front hang length
REAR_HANG = 0.929  # rear hang length
LENGTH = WHEEL_BASE+FRONT_HANG+REAR_HANG
WIDTH = 1.942  # width

DISCRETE_ACTION = np.array([
    [0, 0], # do nothing
    [-0.6, 0], # steer left
    [0.6, 0], # steer right
    [0, 0.2], # accelerate
    [0, -0.8], # decelerate
])
FPS = 100
MAX_FPS = 200
STATE_W = 256
STATE_H = 256
VIDEO_W = 600
VIDEO_H = 400
K = 12
STEP_LIMIT = 20000 # steps
MAX_STEP = 200

LIDAR_RANGE = 10.0
LIDAR_NUM = 120
VALID_SPEED = [-2.5, 2.5]
VALID_STEER = [-0.75, 0.75]
VALID_ACCEL = [-1.0, 1.0]
VALID_ANGULAR_SPEED = [-0.5, 0.5]

# TODO match the color in color_default.py
BG_COLOR = (255, 255, 255, 255)
START_COLOR = (100, 149, 237, 255)
DEST_COLOR = (69, 139, 0, 255)
OBSTACLE_COLOR = (150, 150, 150, 255)
TRAJ_COLOR = (10, 10, 150, 255)
VEHICLE_COLOR = (30, 144, 255, 255)

class Status(Enum):
    NORMAL = 1
    ARRIVED = 2
    COLLIDED = 3
    OUTBOUND = 4
    OUTTIME = 5

def State2Position(state:State)->Position:
    return Position([state.x, state.y, state.heading])

class ParkingMapNormal(Map):
    def __init__(self,name="CarParking", scenario_type="parking"):
        super().__init__(name, scenario_type)

        self.start: Position = None
        self.dest: Position = None
        self.start_box:LinearRing = None
        self.dest_box:LinearRing = None
        self.xmin, self.xmax = 0, 0
        self.ymin, self.ymax = 0, 0
        self.n_obstacle = 0
        self.obstacles:list[Area] = []
        self.vehicle_box:LinearRing = None
        self.case_id = 0

    def set_vehicle(self, vehicle_box):
        self.vehicle_box = vehicle_box

    def reset(self) -> Position:
        if np.random.random() > 0.5:
            start, dest, obstacles = gene_bay_park(self.vehicle_box)
            self.case_id = 0
        else:
            start, dest, obstacles = gene_parallel_park(self.vehicle_box)
            self.case_id = 1

        self.start = Position(start)
        self.start_box = self.start.create_box(self.vehicle_box)
        self.dest = Position(dest)
        self.dest_box = self.dest.create_box(self.vehicle_box)
        self.xmin = np.floor(min(self.start.loc.x, self.dest.loc.x) - 10)
        self.xmax = np.ceil(max(self.start.loc.x, self.dest.loc.x) + 10)
        self.ymin = np.floor(min(self.start.loc.y, self.dest.loc.y) - 10)
        self.ymax = np.ceil(max(self.start.loc.y, self.dest.loc.y) + 10)
        # self.set_boundary([self.xmin, self.ymin, self.xmax, self.ymax])
        idx = 0
        for obs in obstacles:
            idx += 1
            obs_area = Area('%s'%idx, Polygon(obs),line_ids=None, subtype='obstcale')
            self.add_area(obs_area)
            self.obstacles.append(obs_area)
        self.n_obstacle = len(self.obstacles)

        return self.start

class CarParking(gym.Env):
    """
    Description:
        Unconstructed parking environment.
    """

    metadata = {
        "render_mode": [
            "human",
            "rgb_array",
        ]
    }

    def __init__(
        self,
        render_mode: Optional[str] = "human",
        render_fps: int = FPS,
        verbose: bool = True,
        continuous: bool = True,
        use_lidar_observation: bool = True,
        use_img_observation: bool = True,
    ):
        super().__init__()

        self.render_mode = render_mode
        if self.render_mode not in self.metadata["render_mode"]:
            raise NotImplementedError
        self.render_fps = render_fps
        if render_fps > MAX_FPS:
            raise UserWarning()

        self.verbose = verbose
        self.continuous = continuous
        if not continuous:
            raise NotImplementedError('parking environment only support continuous action space !')
        self.use_lidar_observation = use_lidar_observation
        self.use_img_observation = use_img_observation
        self.screen: Optional[pygame.Surface] = None
        self.matrix = None
        self.clock = None
        self.is_open = True
        self.n_step = 0.0
        self.k = None
        self.tgt_repr_size = 5 # relative_distance, cos(theta), sin(theta), cos(phi), sin(phi)

        self.map = ParkingMapNormal()
        self.agent = Vehicle(id_=0, type_="vehicle:racing",
            width=WIDTH, length=LENGTH, height=1.5, # TODO make the hyperparameters
            steering_angle_range=(-0.5, 0.5), steering_velocity_range=(-0.5, 0.5),
            speed_range=(-10, 100), accel_range=(-1, 1)
        )
        self.raw_vehicle_box = VEHICLE_BOX
        self.curr_vehicle_box = None
        self.map.set_vehicle(self.raw_vehicle_box)
        self.lidar = LidarSimlator(LIDAR_RANGE, LIDAR_NUM)
        self.reward = 0.0

        if self.continuous:
            # self.action_space = spaces.Box(
            #     np.array([VALID_ANGULAR_SPEED[0], VALID_ACCEL[0]]).astype(np.float32),
            #     np.array([VALID_ANGULAR_SPEED[1], VALID_ACCEL[1]]).astype(np.float32),
            # ) # steer, acceleration
            self.action_space = spaces.Box(
                np.array([VALID_STEER[0], VALID_SPEED[0]]).astype(np.float32),
                np.array([VALID_STEER[1], VALID_SPEED[1]]).astype(np.float32),
            ) # steer, speed
        else:
            self.action_space = spaces.Discrete(5) # do nothing, left, right, gas, brake
        self.observation_space = {'lidar':None, 'img':None, 'target':None}
        if self.use_img_observation:
            self.observation_space['img'] = spaces.Box(low=0, high=255,
                shape=(STATE_W, STATE_H, 3), dtype=np.uint8)
        if self.use_lidar_observation:
            # the observation is composed of lidar points and target representation
            # the target representation is (relative_distance, cos(theta), sin(theta), cos(phi), sin(phi))
            # where the theta indicates the relative angle of parking lot, and phi means the heading of
            # parking lit in the polar coordinate of the ego car's view
            low_bound, high_bound = np.zeros((LIDAR_NUM)), np.ones((LIDAR_NUM))*LIDAR_RANGE
            self.observation_space['lidar'] = spaces.Box(
                low=low_bound, high=high_bound, shape=(LIDAR_NUM,), dtype=np.float64
            )
        low_bound = np.array([0,-1,-1,-1,-1])
        high_bound = np.array([50,1,1,1,1]) # TODO the hyper-param high_bound[0], the max distance to target
        self.observation_space['target'] = spaces.Box(
            low=low_bound, high=high_bound, shape=(self.tgt_repr_size,), dtype=np.float64
        )

    def reset(self,) -> np.ndarray:
        self.n_step = 0
        self.reward = 0.0

        initial_pos = self.map.reset()
        state = State(
            frame=self.n_step, heading=initial_pos.heading,
            x = initial_pos.loc.x, y = initial_pos.loc.y
        )
        self.agent.reset(state)
        self.matrix = self._coord_transform_matrix()
        self.curr_vehicle_box = initial_pos.create_box(self.raw_vehicle_box)
        return self.step()[0]

    def _coord_transform_matrix(self) -> list:
        """Get the transform matrix that convert the real world coordinate to the pygame coordinate.
        """
        k = K
        bx = 0.5 * (VIDEO_W - k * (self.map.xmax + self.map.xmin))
        by = 0.5 * (VIDEO_H - k * (self.map.ymax + self.map.ymin))
        self.k = k
        return [k, 0, 0, k, bx, by]

    def _coord_transform(self, obj) -> list:
        transformed = affine_transform(obj, self.matrix)
        return list(transformed.coords)

    def _detect_collision(self):
        for obstacle in self.map.obstacles:
            if self.curr_vehicle_box.intersects(obstacle.polygon.exterior):
                return True
        return False

    def _detect_outbound(self):
        vehicle_box = np.array(self._coord_transform(self.curr_vehicle_box))
        if vehicle_box[:, 0].min() < 0:
            return True
        if vehicle_box[:, 0].max() > VIDEO_W:
            return True
        if vehicle_box[:, 1].min() < 0:
            return True
        if vehicle_box[:, 1].max() > VIDEO_H:
            return True
        return False

    def _check_arrived(self):
        vehicle_box = Polygon(self.curr_vehicle_box)
        dest_box = Polygon(self.map.dest_box)
        union_area = vehicle_box.intersection(dest_box).area
        if union_area / dest_box.area > 0.95:
            # print('arrive!!: ',union_area / dest_box.area)
            return True
        return False

    def _check_time_exceeded(self):
        return self.n_step < STEP_LIMIT

    def _check_status(self):
        # TODO: merge the status into traffic event
        if self._detect_collision():
            return Status.COLLIDED
        if self._detect_outbound():
            return Status.OUTBOUND
        if self._check_arrived():
            return Status.ARRIVED
        if self._check_time_exceeded():
            return Status.OUTTIME
        return Status.NORMAL

    def _get_reward(self, prev_state: Position, curr_state: Position):
        # time penalty
        time_cost = - np.tanh(self.n_step / (10*MAX_STEP))

        # Euclidean distance reward & angle reward
        def get_angle_diff(angle1, angle2):
            # norm to 0 ~ pi/2
            angle_dif = math.acos(math.cos(angle1 - angle2)) # 0~pi
            return angle_dif if angle_dif<math.pi/2 else math.pi-angle_dif
        dist_diff = curr_state.loc.distance(self.map.dest.loc)
        angle_diff = get_angle_diff(curr_state.heading, self.map.dest.heading)
        prev_dist_diff = prev_state.loc.distance(self.map.dest.loc)
        prev_angle_diff = get_angle_diff(prev_state.heading, self.map.dest.heading)
        dist_norm_ratio = max(self.map.dest.loc.distance(self.map.start.loc),10)
        angle_norm_ratio = math.pi
        dist_reward = math.exp(-dist_diff/dist_norm_ratio) - \
            math.exp(-prev_dist_diff/dist_norm_ratio)
        angle_reward = math.exp(-angle_diff/angle_norm_ratio) - \
            math.exp(-prev_angle_diff/angle_norm_ratio)

        return time_cost+dist_reward+angle_reward

    def step(self, action: Union[np.ndarray, int] = None):
        '''
        Parameters:
        ----------
        `action`: `np.ndarra`y if continous action space else `int`

        Returns:
        ----------
        ``obsercation`` (Dict):
            the observation of image based surroundings, lidar view and target representation.

        ``reward`` (float): the reward considering the distance and angle difference between
                current state and destination.
        `status` (`Status`): represent the state of vehicle, including:
                `NORMAL`, `ARRIVED`, `COLLIDED`, `OUTBOUND`, `OUT_TIME`
        `info` (`OrderedDict`): other information.
        '''

        assert self.agent is not None
        self.n_step += 1
        prev_state = self.agent.current_state
        if action is not None:
            if self.continuous:
                self.agent.update_state(action)
            else:
                if not self.action_space.contains(action):
                    raise InvalidAction(
                        f"you passed the invalid action `{action}`. "
                        f"The supported action_space is `{self.action_space}`"
                    )
                self.agent.update_state(DISCRETE_ACTION[action])

        # update vehicle box
        self.curr_vehicle_box = \
            State2Position(self.agent.current_state).create_box(self.raw_vehicle_box)

        observation = self.render(self.render_mode)

        status = self._check_status()
        done = (status != Status.NORMAL)

        reward = self._get_reward(State2Position(prev_state), \
            State2Position(self.agent.current_state))

        # done = False if status==Status.CONTINUE else True
        info = OrderedDict({'reward_info':None,
            'path_to_dest':None,
            'status':status})

        return observation, reward, done, info

    def _render(self, surface: pygame.Surface):
        surface.fill(BG_COLOR)
        for obstacle in self.map.obstacles:
            pygame.draw.polygon(
                surface, OBSTACLE_COLOR, self._coord_transform(obstacle.polygon.exterior))

        pygame.draw.polygon(
            surface, START_COLOR, self._coord_transform(self.map.start_box), width=1)
        pygame.draw.polygon(
            surface, DEST_COLOR, self._coord_transform(self.map.dest_box))#, width=1

        pygame.draw.polygon(
            surface, VEHICLE_COLOR, self._coord_transform(self.curr_vehicle_box))

        if len(self.agent.trajectory) > 1: # TODO undefined trajectory
            pygame.draw.lines(
                surface, TRAJ_COLOR, False,
                self._coord_transform(LineString(self.agent.trajectory))
            )

    def _get_img_observation(self, surface: pygame.Surface):
        angle = self.agent.current_state.heading
        old_center = surface.get_rect().center

        # Rotate and find the center of the vehicle
        capture = pygame.transform.rotate(surface, np.rad2deg(angle))
        rotate = pygame.Surface((VIDEO_W, VIDEO_H))
        rotate.blit(capture, capture.get_rect(center=old_center))

        vehicle_center = np.array(self._coord_transform(self.curr_vehicle_box.centroid)[0])
        dx = (vehicle_center[0]-old_center[0])*np.cos(angle) \
            + (vehicle_center[1]-old_center[1])*np.sin(angle)
        dy = -(vehicle_center[0]-old_center[0])*np.sin(angle) \
            + (vehicle_center[1]-old_center[1])*np.cos(angle)

        # align the center of the observation with the center of the vehicle
        observation = pygame.Surface((VIDEO_W, VIDEO_H))

        observation.fill(BG_COLOR)
        observation.blit(rotate, (int(-dx), int(-dy)))
        observation = observation.subsurface((
            (VIDEO_W-STATE_W)/2, (VIDEO_H-STATE_H)/2), (STATE_W, STATE_H))

        obs_str = pygame.image.tostring(observation, "RGB")
        observation = np.frombuffer(obs_str, dtype=np.uint8)
        observation = observation.reshape(self.observation_space['img'].shape)
        return observation

    def _get_lidar_observation(self,):
        obs_list = [obs.polygon.exterior for obs in self.map.obstacles]
        ego_pos = (self.agent.current_state.x, self.agent.current_state.y,\
             self.agent.current_state.heading)
        lidar_view = self.lidar.get_observation(ego_pos, obs_list)
        return lidar_view

    def _get_targt_repr(self,):
        # target position representation
        dest_pos = (self.map.dest.loc.x, self.map.dest.loc.y, self.map.dest.heading)
        ego_pos = (self.agent.current_state.x, self.agent.current_state.y,\
             self.agent.current_state.heading)
        rel_distance = math.sqrt((dest_pos[0]-ego_pos[0])**2 + (dest_pos[1]-ego_pos[1])**2)
        rel_angle = math.atan2(dest_pos[1]-ego_pos[1], dest_pos[0]-ego_pos[0]) - ego_pos[2]
        rel_dest_heading = dest_pos[2] - ego_pos[2]
        tgt_repr = np.array([rel_distance, math.cos(rel_angle), math.sin(rel_angle),\
            math.cos(rel_dest_heading), math.cos(rel_dest_heading)])
        return tgt_repr

    def render(self, mode: str = "human"):
        assert mode in self.metadata["render_mode"]

        if mode == "human":
            display_flags = pygame.SHOWN
        else:
            display_flags = pygame.HIDDEN
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIDEO_W, VIDEO_H), flags = display_flags)
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self._render(self.screen)
        observation = {'img':None, 'lidar':None, 'target':None}
        if self.use_img_observation:
            observation['img'] = self._get_img_observation(self.screen)
        if self.use_lidar_observation:
            observation['lidar'] = self._get_lidar_observation()
        observation['target'] = self._get_targt_repr()
        pygame.display.update()
        self.clock.tick(self.render_fps)

        return observation

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.is_open = False
            pygame.quit()


if __name__ == "__main__":
    a = np.array([0.0, 0.0])

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

    env = CarParking()
    is_open = True
    while is_open:
        env.reset()
        total_reward = 0.0
        n_step = 0
        restart = False
        status = Status.NORMAL
        while status == Status.NORMAL and is_open:
            register_input()
            observation, reward, status = env.step(a)
            total_reward += reward
            n_step += 1

        is_open = False

    env.close()

