##! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: parse_nuplan.py
# @Description: This file implements a parser for NuPlan dataset.
# @Author: Yueyuan Li
# @Version: 1.0.0

import datetime
import os
import sqlite3
from typing import List, Tuple

import geopandas as gpd
import numpy as np
import pyogrio
from shapely.geometry import LinearRing, LineString, Point, Polygon

from tactics2d.map.element import Area, Lane, Map, Regulatory, RoadLine
from tactics2d.participant.element import Cyclist, Other, Pedestrian, Vehicle
from tactics2d.participant.trajectory import State, Trajectory


class NuPlanParser:
    """This class implements a parser for NuPlan dataset.

    !!! quote "Reference"
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

    _ROADLINE_MAPPING = {0: "dashed", 1: "virtual", 2: "solid", 3: "virtual"}

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
            location (str): The location of the trajectory data file.
        """
        file_path = os.path.join(folder, file)

        with sqlite3.connect(file_path) as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT location FROM log;")
            location = cursor.fetchone()[0]

        return location

    def parse_trajectory(
        self, file: str, folder: str, stamp_range: Tuple[float, float] = None
    ) -> Tuple[dict, List[int]]:
        """This function parses trajectories from a single NuPlan database file.

        Args:
            file (str): The name of the trajectory data file. The file is expected to be a sqlite3 database file (.db).
            folder (str): The path to the folder containing the trajectory file.
            stamp_range (Tuple[float, float], optional): The time range of the trajectory data to parse. If the stamp range is not given, the parser will parse the whole trajectory data.

        Returns:
            participants (dict): A dictionary of participants. The keys are the ids of the participants. The values are the participants.
            stamps (List[int]): The actual time range of the trajectory data. Because NuPlan collects data at an unstable frequency, the parser will return a list of time stamps.
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
                id_ = int.from_bytes(row[0], byteorder="big")
                participants[row[0]] = self._CLASS_MAPPING[category_name](
                    id_=id_,
                    type_=category_name,
                    trajectory=Trajectory(id_=id_, fps=20, stable_freq=False),
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
                participants[row[2]].trajectory.add_state(state)

        cursor.close()
        connection.close()
        stamps = sorted(list(time_stamps))

        return participants, stamps

    def parse_map(self, file_path: str) -> Map:
        map_meta = gpd.read_file(file_path, layer="meta", engine="pyogrio")
        projection_system = map_meta[map_meta["key"] == "projectedCoordSystem"]["value"].iloc[0]
        type_mapping = self._ROADLINE_MAPPING
        id_cnt = 0

        map_ = Map(name="nuplan_" + file_path.split("/")[-1].split(".")[0])

        def load_utm_coords(layer_name):
            gdf_in_pixel_coords = pyogrio.read_dataframe(file_path, layer=layer_name)
            gdf_in_utm_coords = gdf_in_pixel_coords.to_crs(projection_system)
            return gdf_in_utm_coords

        boundaries = load_utm_coords("boundaries")
        for _, row in boundaries.iterrows():
            roadline = RoadLine(
                id_=int(row["boundary_segment_fids"].split(",")[0]),
                type_=type_mapping[row["boundary_type_fid"]],
                geometry=LineString(row["geometry"]),
            )
            map_.add_roadline(roadline)

        # load lands
        lane_dict = {"lanes_polygons": "lane"}

        for key, value in lane_dict.items():
            df_lanes = load_utm_coords(key)
            for _, row in df_lanes.iterrows():
                lane = Lane(
                    id_=row["lane_fid"],
                    geometry=LinearRing(Polygon(row["geometry"]).exterior),
                    subtype=value,
                )
                map_.add_lane(lane)

        # load areas
        area_dict = {
            "carpark_areas": "parking",
            "crosswalks": "crosswalk",
            "intersections": "lane",
            "walkways": "walkway",
        }
        id_cnt = max(np.array(list(map_.ids.keys()), np.int64)) + 1

        for key, value in area_dict.items():
            df_areas = load_utm_coords(key)
            for _, row in df_areas.iterrows():
                area = Area(
                    id_=id_cnt,
                    geometry=Polygon(row["geometry"]),
                    subtype=value,
                    custom_tags={"heading": row["heading"]} if key == "carpark_areas" else None,
                )
                id_cnt += 1
                map_.add_area(area)

        traffic_lights = load_utm_coords("traffic_lights")
        for _, row in traffic_lights.iterrows():
            traffic_light = Regulatory(
                id_=id_cnt,
                subtype="traffic_light",
                position=Point(row["geometry"].x, row["geometry"].y),
                custom_tags={"heading": row["ori_mean_yaw"]},
            )
            id_cnt += 1
            map_.add_regulatory(traffic_light)

        return map_
