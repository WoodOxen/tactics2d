##! python3
# -*- coding: utf-8 -*-
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: parse_nuplan.py
# @Description: This file implements a parser for NuPlan dataset.
# @Author: Yueyuan Li
# @Version: 1.0.0

import os
import datetime
import json
from typing import Tuple, List

import geopandas as gpd
import sqlite3
import numpy as np
from shapely.geometry import Point, LineString, Polygon
from shapely.affinity import affine_transform

from tactics2d.participant.element import Vehicle, Pedestrian, Cyclist, Other
from tactics2d.participant.trajectory import State, Trajectory
from tactics2d.map.element import RoadLine, Lane, LaneRelationship, Area, Regulatory, Map


class NuPlanParser:
    """This class implements a parser for NuPlan dataset.

    !!! info "Reference"
        Caesar, Holger, et al. "nuplan: A closed-loop ml-based planning benchmark for autonomous vehicles." arXiv preprint arXiv:2106.11810 (2021).
    """

    _CLASS_MAPPING = {
        "vehicle": Vehicle,
        "bicycle": Cyclist,
        "pedestrian": Pedestrian,
        "traffic_cone": Other,
        "barrier": Other,
        "czone_sign": Other,
        "generic_object": Other,
    }

    # millisecond-level time stamp at 2021-01-01 00:00:00
    _DATETIME = datetime.datetime(2021, 1, 1, 0, 0, 0).timestamp() * 1000

    def __init__(self):
        self.transform_matrix = np.zeros((6, 1))

    def get_location(self, file: str, folder: str) -> str:
        """This function gets the location of a single trajectory data file.

        Args:
            file (str): The name of the trajectory data file. The file is expected to be a sqlite3 database file (.db).
            folder (str): The path to the folder containing the trajectory file.

        Returns:
            str: The location of the trajectory data file.
        """
        file_path = os.path.join(folder, file)

        with sqlite3.connect(file_path) as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT location FROM log;")
            location = cursor.fetchone()[0]

        return location

    def update_transform_matrix(self, file: str, folder: str):
        """This function updates the transform matrix of the map.

        Args:
            file (str): The name of the trajectory data file. The file is expected to be a sqlite3 database file (.db).
            folder (str): The path to the folder containing the trajectory file.
        """
        with open("./tactics2d/data/map/NuPlan/nuplan-maps-v1.0.json", "r") as f:
            configs = json.load(f)

        location = self.get_location(file, folder)
        transform_matrix = configs[location]["layers"]["Intensity"]["transform_matrix"]
        self.transform_matrix = np.array(
            [
                1 / transform_matrix[0][0],
                transform_matrix[1][0],
                transform_matrix[0][1],
                1 / transform_matrix[1][1],
                -transform_matrix[0][3] / transform_matrix[0][0],
                -transform_matrix[1][3] / transform_matrix[1][1],
            ]
        )

    def parse_trajectory(
        self, file: str, folder: str, stamp_range: Tuple[float, float] = None
    ) -> Tuple[dict, List[int]]:
        """This function parses trajectories from a single NuPlan database file.

        Args:
            file (str): The name of the trajectory data file. The file is expected to be a sqlite3 database file (.db).
            folder (str): The path to the folder containing the trajectory file.
            stamp_range (Tuple[float, float], optional): The time range of the trajectory data to parse. If the stamp range is not given, the parser will parse the whole trajectory data.

        Returns:
            dict: A dictionary of participants. The keys are the ids of the participants. The values are the participants.
            List[int]: The actual time range of the trajectory data. Because NuPlan collects data at an unstable frequency, the parser will return a list of time stamps.
        """
        participants = dict()
        time_stamps = set()

        file_path = os.path.join(folder, file)

        if stamp_range is None:
            stamp_range = (-float("inf"), float("inf"))

        with sqlite3.connect(file_path) as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT * FROM track;")
            rows = cursor.fetchall()

            for row in rows:
                category_token = row[1]
                cursor.execute("SELECT * FROM category WHERE token=?;", (category_token,))
                category_name = cursor.fetchone()[1]
                participants[row[0]] = self._CLASS_MAPPING[category_name](
                    id_=row[0],
                    type_=category_name,
                    trajectory=Trajectory(id_=row[0], fps=20, stable_freq=False),
                    length=row[3],
                    width=row[2],
                    height=row[4],
                )

            cursor.execute("SELECT * FROM lidar_box;")
            rows = cursor.fetchall()

            for row in rows:
                cursor.execute("SELECT * FROM lidar_pc WHERE token=?;", (row[1],))
                time_stamp = int(cursor.fetchone()[7] / 1000 - self._DATETIME)

                if time_stamp < stamp_range[0] or time_stamp > stamp_range[1]:
                    continue
                time_stamps.add(time_stamp)

                state = State(
                    frame=time_stamp, x=row[5], y=row[6], heading=row[14], vx=row[11], vy=row[12]
                )
                participants[row[2]].trajectory.append_state(state)

        cursor.close()
        connection.close()
        actual_time_range = sorted(list(time_stamps))

        return participants, actual_time_range

    def parse_map(self, file: str, folder: str) -> Map:
        """This function parses a map from a single NuPlan map file. The map file is expected to be a geopackage file (.gpkg).

        TODO: the parsing of lane connectors is not implemented yet.

        A NuPlan map includes the following layers: 'baseline_paths', 'carpark_areas', 'generic_drivable_areas', 'dubins_nodes', 'lane_connectors', 'intersections', 'boundaries', 'crosswalks', 'lanes_polygons', 'lane_group_connectors', 'lane_groups_polygons', 'road_segments', 'stop_polygons', 'traffic_lights', 'walkways', 'gen_lane_connectors_scaled_width_polygons', 'meta'. In this parser, we only parse the following layers: 'boundaries', 'lanes_polygons', 'lane_connectors', 'carpark_areas', 'crosswalks', 'walkways', 'stop_polygons', 'traffic_lights'
        """
        map_file = os.path.join(folder, file)
        map_ = Map(name="nuplan_" + file.split(".")[0])

        boundaries = gpd.read_file(map_file, layer="boundaries")
        for _, row in boundaries.iterrows():
            boundary_ids = [int(s) for s in row["boundary_segment_fids"].split(",") if s.isdigit()]
            boundary_id = boundary_ids[0] - 1
            boundary = RoadLine(
                id_=str(boundary_id),
                linestring=affine_transform(LineString(row["geometry"]), self.transform_matrix),
            )
            map_.add_roadline(boundary)

        id_cnt = max(np.array(list(map_.ids.keys()), np.int64)) + 1

        # lane_polygons = gpd.read_file(map_file, layer="lanes_polygons")
        # for _, row in lane_polygons.iterrows():
        #     lane_polygon = Lane(
        #         id_=str(row["lane_fid"]),
        #         left_side=map_.get_by_id(str(row["left_boundary_fid"])).geometry,
        #         right_side=map_.get_by_id(str(row["right_boundary_fid"])).geometry,
        #         line_ids=set([str(row["left_boundary_fid"]), str(row["right_boundary_fid"])]),
        #         subtype=None,
        #         speed_limit=row["speed_limit_mps"],
        #         speed_limit_unit="m/s",
        #     )
        #     lane_polygon.add_related_lane(str(row["from_edge_fid"]), LaneRelationship.PREDECESSOR)
        #     lane_polygon.add_related_lane(str(row["to_edge_fid"]), LaneRelationship.SUCCESSOR)
        #     map_.add_lane(lane_polygon)

        lane_connectors = gpd.read_file(map_file, layer="lane_connectors")
        for _, row in lane_connectors.iterrows():
            # TODO: parse the polygon in lane to left and right side
            pass

        area_dict = {
            "crosswalks": "crosswalk",
            "carpark_areas": "parking",
            "walkways": "walkway",
            "stop_polygons": "stop",
        }

        for key, value in area_dict.items():
            df_areas = gpd.read_file(map_file, layer=key)
            for _, row in df_areas.iterrows():
                area = Area(
                    id_=str(id_cnt),
                    geometry=affine_transform(Polygon(row["geometry"]), self.transform_matrix),
                    subtype=value,
                    custom_tags={"heading": row["heading"]} if key == "carpark_areas" else None,
                )
                id_cnt += 1
                map_.add_area(area)

        traffic_lights = gpd.read_file(map_file, layer="traffic_lights")
        for _, row in traffic_lights.iterrows():
            traffic_light = Regulatory(
                id_=str(id_cnt),
                subtype="traffic_light",
                position=affine_transform(
                    Point(row["geometry"].x, row["geometry"].y), self.transform_matrix
                ),
                custom_tags={"heading": row["ori_mean_yaw"]},
            )
            id_cnt += 1
            map_.add_regulatory(traffic_light)

        return map_
