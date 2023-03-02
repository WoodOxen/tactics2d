from typing import Tuple, List
import time
import logging

import numpy as np
from shapely.geometry import Point, LineString, LinearRing
from shapely.affinity import affine_transform

from tactics2d.math import Bezier, Circle
from tactics2d.map.element import Lane, Map


# Track related configurations
N_CHECKPOINT = (10, 20)  # the number of turns is ranging in 10-20
TRACK_WIDTH = 15  # the width of the track is ranging in 15m
TRACK_RAD = 800  # the maximum curvature radius
CURVE_RAD = (10, 150)  # the curvature radius is ranging in 10-150m
TILE_LENGTH = 10  # the length of each tile


class RacingTrackGenerator:
    """Generate a racing track with random configurations.

    Attributes:
        bezier_generator (Bezier): The Bezier line generator.
        verbose (bool, optional): Whether to print the generation process. Defaults to False.
    """

    def __init__(self, bezier_param: Tuple[int, int] = (2, 50), verbose: bool = False):

        self.bezier_generator = Bezier(*bezier_param)
        self.verbose = verbose

    def _get_checkpoints(self) -> Tuple[List[np.ndarray], List[np.ndarray], bool]:

        n_checkpoint = np.random.randint(*N_CHECKPOINT)
        noise = np.random.uniform(0, 2 * np.pi / n_checkpoint, n_checkpoint)
        alpha = 2 * np.pi * np.arange(n_checkpoint) / n_checkpoint + noise
        rad = np.random.uniform(TRACK_RAD / 5, TRACK_RAD, n_checkpoint)

        checkpoints = np.array([rad * np.cos(alpha), rad * np.sin(alpha)])
        success = False
        control_points = []

        # try generating checkpoints for 100
        for _ in range(100):
            glued_cnt = 0
            control_points.clear()

            for i in range(n_checkpoint):
                pt1 = checkpoints[:, i - 1]
                pt2 = checkpoints[:, i]
                next_i = 0 if i + 1 == n_checkpoint else i + 1
                pt3 = checkpoints[:, next_i]

                t1 = np.random.uniform(low=1 / 4, high=1 / 2)
                t2 = np.random.uniform(low=1 / 4, high=1 / 2)
                pt1_ = (1 - t1) * pt2 + t1 * pt1
                pt3_ = (1 - t2) * pt2 + t2 * pt3
                _, radius = Circle.from_three_points(pt1_, pt2, pt3_)
                if radius < CURVE_RAD[0]:
                    if rad[i] > rad[next_i]:
                        rad[next_i] += np.random.uniform(0.0, 10.0)
                    else:
                        rad[next_i] -= np.random.uniform(0.0, 10.0)
                    alpha[next_i] += np.random.uniform(0.0, 0.05)
                    checkpoints[:, next_i] = [
                        rad[next_i] * np.cos(alpha[next_i]),
                        rad[next_i] * np.sin(alpha[next_i]),
                    ]
                elif radius > CURVE_RAD[1]:
                    if rad[i] > rad[next_i]:
                        rad[next_i] -= np.random.uniform(0.0, 10.0)
                    else:
                        rad[next_i] += np.random.uniform(0.0, 10.0)
                    alpha[next_i] -= np.random.uniform(0.0, 0.05)
                    checkpoints[:, next_i] = [
                        rad[next_i] * np.cos(alpha[next_i]),
                        rad[next_i] * np.sin(alpha[next_i]),
                    ]
                else:
                    glued_cnt += 1
                    control_points.append([pt1_, pt3_])

            if glued_cnt == n_checkpoint:
                success = True
                break

        success = success and all(alpha == sorted(alpha))

        return checkpoints, control_points, success
    
    def _get_start_point(
            self, n_checkpoint: int, control_points: List[np.ndarray]
        ) -> Tuple[Point, int]:

        # get the lengths of the straight track
        straight_lens = []
        for i in range(n_checkpoint):
            straight_lens.append(
                np.linalg.norm([control_points[i][0], control_points[i-1][1]]))
        # find the first three longest straight track
        sorted_id = sorted(
            range(n_checkpoint), key=lambda i: straight_lens[i], reverse=True)
        start_id = None
        for i in range(3):
            start_id = sorted_id[i]
            if straight_lens[start_id] < 200:
                break
        # get the start point from the track[start_id] at 1/3
        start_point = LineString(
            [control_points[start_id][0], control_points[start_id - 1][1]]
        ).interpolate(straight_lens[start_id] / 3)

        return start_point, start_id
    
    def _get_center_line(
            self, start_point: Point, start_id: int, 
            checkpoints: List[np.ndarray], control_points: List[np.ndarray]
        ) -> LineString:

        points = []
        # get the center line by Bezier curve generator
        for i in range(checkpoints.shape[1]):
            points += self.bezier_generator.get_points(
                np.array([
                    control_points[start_id - i - 1][1],
                    checkpoints[:, start_id - i - 1],
                    control_points[start_id - i - 1][0],
                ])
            )
        # create the new map by the centerline
        center_line = LineString([start_point] + points + [start_point])

        return center_line
    
    def _get_tiles(self, n_tile: int, center_line: LineString) -> List[]:
        tiles = []
        center_points = []
        left_points = []
        right_points = []

        # refresh the center line to a new  equidistance distance 

        for i in range(n_tile + 1):
            tiles.append(
                center_line.interpolate(TILE_LENGTH * i).buffer(TRACK_WIDTH / 2))
        return tiles

    def _create_map(self, n_tile: int, center_line: LineString):

        for i in range(n_tile + 1):
            center_points.append(center.interpolate(TILE_LENGTH * i))
            if i > 0:
                pt1 = center_points[i - 1]
                pt2 = center_points[i]
                x_diff = pt2.x - pt1.x
                y_diff = pt2.y - pt1.y
                k = TRACK_WIDTH / 2 / Point(pt1).distance(Point(pt2))
                left_points.append([pt1.x - k * y_diff, pt1.y + k * x_diff])
                right_points.append([pt1.x + k * y_diff, pt1.y - k * x_diff])
        left_points.append(left_points[0])
        right_points.append(right_points[0])

        # generate map
        for i in range(n_tile):
            tile = Lane(
                id_="%04d" % i,
                left_side=LineString([left_points[i], left_points[i + 1]]),
                right_side=LineString([right_points[i], right_points[i + 1]]),
                subtype="lane",
                inferred_participants=["sports_car"],
            )
            self.map.add_lane(tile)

        return n_tile

    def generate_track_tiles(self) -> List[LinearString]:
        """_summary_
        """
        t1 = time.time()

        # generate the checkpoints
        success = False
        while not success:
            checkpoints, control_points, success = self._get_checkpoints()

        n_checkpoints = checkpoints.shape[1]
        logging.info(f"Generated a new track with {n_checkpoints} checkpoints.")

        start_point, start_id = self._get_start_point(n_checkpoints, control_points)

        center_line = self._get_center_line(start_point, start_id, checkpoints, control_points)

        distance = center_line.length
        n_tile = int(np.ceil(distance / TILE_LENGTH))
        tiles = self._get_tiles(n_tile, center_line)
        logging.info(f"The track is {int(distance)}m long and has {n_tile} tiles.")

        # record time cost
        t2 = time.time()
        logging.info("The generation process takes %.4fs." % (t2 - t1))

        return tiles