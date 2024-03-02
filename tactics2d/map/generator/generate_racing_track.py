##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: generate_racing_track.py
# @Description: This file defines a class for generating random racing tracks.
# @Author: Yueyuan Li
# @Version: 1.0.0


import logging
import time
from typing import Dict, List, Tuple

import numpy as np
from shapely.geometry import LineString, Point

from tactics2d.map.element import Lane, LaneRelationship, Map, RoadLine
from tactics2d.math.geometry import Circle
from tactics2d.math.interpolate import Bezier
from tactics2d.participant.trajectory import State


class RacingTrackGenerator:
    """This class generates random racing tracks.

    The generating process can be described as follows:

    1. Generate a circle with a radius = 800 m. Ideally, the time needed for a vehicle to make a full turn in this circle is around 3 minutes if it drives at a speed of 100 km/h.
    2. Generate some checkpoints that are deviated from the circle.
    2. Generate the center line by interpolating the checkpoints with Bezier curves.
    3. Check if the center line has extremely sharp turns. If so, adjust the checkpoints.
    4. Iterate until the center line is valid.
    5. Generate the tiles by interpolating the center line.
    6. Generate the road bounds by adding the track width to the left and right of the center line.

    Attributes:
        bezier_order (int): The order of the Bezier curve. Defaults to 2.
        bezier_interpolation (int): The number of interpolation points for each Bezier curve. Defaults to 50.
    """

    _n_checkpoint = (10, 20)  # the number of turns is ranging in 10-20
    _track_width = 15  # the width of the track is varying around 15m
    _track_rad = 800  # the maximum curvature radius
    _curve_rad = (10, 150)  # the curvature radius is ranging in 10-150m
    _tile_length = 10  # the length of each tile

    def __init__(self, bezier_order=2, bezier_interpolation=50):
        """Initialize the attributes in the class."""
        self.bezier_generator = Bezier(bezier_order)
        self.bezier_interpolation = bezier_interpolation

    def _get_checkpoints(self) -> Tuple[List[np.ndarray], List[np.ndarray], bool]:
        n_checkpoint = np.random.randint(*self._n_checkpoint)
        noise = np.random.uniform(0, 2 * np.pi / n_checkpoint, n_checkpoint)
        alpha = 2 * np.pi * np.arange(n_checkpoint) / n_checkpoint + noise
        rad = np.random.uniform(self._track_rad / 5, self._track_rad, n_checkpoint)

        checkpoints = np.array([rad * np.cos(alpha), rad * np.sin(alpha)])
        success = False
        control_points = []

        # try generating the checkpoints for 100 times
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
                _, radius = Circle.get_circle(Circle.ConstructBy.ThreePoints, pt1_, pt2, pt3_)
                if radius < self._curve_rad[0]:
                    if rad[i] > rad[next_i]:
                        rad[next_i] += np.random.uniform(0.0, 10.0)
                    else:
                        rad[next_i] -= np.random.uniform(0.0, 10.0)
                    alpha[next_i] += np.random.uniform(0.0, 0.05)
                    checkpoints[:, next_i] = [
                        rad[next_i] * np.cos(alpha[next_i]),
                        rad[next_i] * np.sin(alpha[next_i]),
                    ]
                elif radius > self._curve_rad[1]:
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
            straight_lens.append(np.linalg.norm([control_points[i][0], control_points[i - 1][1]]))

        # find the first three longest straight track
        sorted_id = sorted(range(n_checkpoint), key=lambda i: straight_lens[i], reverse=True)
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
        self,
        start_point: Point,
        start_id: int,
        checkpoints: List[np.ndarray],
        control_points: List[np.ndarray],
    ) -> LineString:
        points = []
        # get the center line by Bezier curve generator
        for i in range(checkpoints.shape[1]):
            new_points = self.bezier_generator.get_curve(
                np.array(
                    [
                        control_points[start_id - i - 1][1],
                        checkpoints[:, start_id - i - 1],
                        control_points[start_id - i - 1][0],
                    ]
                ),
                self.bezier_interpolation,
            )
            points.extend(new_points)

        # create the new map by the centerline
        center_line = LineString([start_point] + points + [start_point])

        return center_line

    def _get_tiles(self, n_tile: int, center_line: LineString) -> Dict[str, Lane]:
        center_points = [center_line.interpolate(self._tile_length * i) for i in range(n_tile)]

        # generate tracks with the same length
        left_points = []
        right_points = []
        tiles = {}

        for i in range(n_tile):
            pt0 = center_points[i - 1]
            pt1 = center_points[i]
            x_diff = pt1.x - pt0.x
            y_diff = pt1.y - pt0.y
            k = self._track_width / 2 / np.linalg.norm([x_diff, y_diff])
            left_points.append([pt1.x - k * y_diff, pt1.y + k * x_diff])
            right_points.append([pt1.x + k * y_diff, pt1.y - k * x_diff])

        left_points.append(left_points[0])
        right_points.append(right_points[0])

        for i in range(n_tile):
            tile = Lane(
                id_="%04d" % i,
                left_side=LineString([left_points[i], left_points[i + 1]]),
                right_side=LineString([right_points[i], right_points[i + 1]]),
                subtype="road",
                inferred_participants=["sports_car"],
            )

            if i > 0:
                tile.add_related_lane("%04d" % (i - 1), LaneRelationship.PREDECESSOR)
            if i < n_tile - 1:
                tile.add_related_lane("%04d" % (i + 1), LaneRelationship.SUCCESSOR)

            tiles[tile.id_] = tile

        tiles["%04d" % 0].add_related_lane("%04d" % (n_tile - 1), LaneRelationship.PREDECESSOR)
        tiles["%04d" % (n_tile - 1)].add_related_lane("%04d" % 0, LaneRelationship.SUCCESSOR)

        return tiles

    def generate(self, map_: Map):
        """Generate a random racing scenario.

        Args:
            map_ (Map): The map instance to store the generated racing scenario.
        """
        t1 = time.time()

        # generate the checkpoints
        success = False
        while not success:
            checkpoints, control_points, success = self._get_checkpoints()

        n_checkpoints = checkpoints.shape[1]
        logging.info(f"Start generating a track with {n_checkpoints} checkpoints.")

        start_point, start_id = self._get_start_point(n_checkpoints, control_points)
        map_.customs["start_state"] = State(frame=0, x=start_point.x, y=start_point.y, vx=0, vy=0)

        # generate center line
        center_line = self._get_center_line(start_point, start_id, checkpoints, control_points)

        # generate tiles
        distance = center_line.length
        n_tile = int(np.ceil(distance / self._tile_length))
        map_.lanes = self._get_tiles(n_tile, center_line)

        map_.roadlines = {
            "start_line": RoadLine(
                id_="0",
                geometry=LineString(map_.lanes["0000"].ends),
                type_="solid",
                color=(255, 0, 0),
            ),
            "end_line": RoadLine(
                id_="1",
                geometry=LineString(map_.lanes["0000"].starts),
                type_="solid",
                color=(0, 255, 0),
            ),
            "center_line": RoadLine(id_="2", geometry=center_line),
        }

        logging.info(f"The track is {int(distance)}m long and has {n_tile} tiles.")

        # record time cost
        t2 = time.time()
        logging.info("The generating process takes %.4fs." % (t2 - t1))
