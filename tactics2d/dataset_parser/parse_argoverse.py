##! python3
# -*- coding: utf-8 -*-
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: parse_argoverse.py
# @Description: This file implements a parser for Argoverse 2 dataset.
# @Author: Yueyuan Li
# @Version: 1.0.0

import os
import json
from typing import Tuple

import pandas as pd
import numpy as np
from shapely.geometry import LineString, Polygon

from tactics2d.participant.element import Vehicle, Pedestrian, Cyclist, Other
from tactics2d.trajectory.element import State, Trajectory
from tactics2d.map.element import Area, RoadLine, Lane, LaneRelationship, Map


class ArgoverseParser:
    """This class implements a parser for Argoverse dataset. The default size of the participants are referred to the [official visualization toolkit](https://github.com/argoverse/av2-api/blob/main/src/av2/datasets/motion_forecasting/viz/scenario_visualization.py).

    !!! info "Reference"
        Wilson, Benjamin, et al. "Argoverse 2: Next generation datasets for self-driving perception and forecasting." arXiv preprint arXiv:2301.00493 (2023).
    """

    _TYPE_MAPPING = {
        "vehicle": "car",
        "bus": "bus",
        "motorcyclist": "motorcycle",
        "cyclist": "bicycle",
        "riderless_bicycle": "bicycle",
        "pedestrian": "pedestrian",
        "background": "background",
        "static": "static",
    }

    _CLASS_MAPPING = {
        "vehicle": Vehicle,
        "bus": Vehicle,
        "motorcyclist": Cyclist,
        "cyclist": Cyclist,
        "riderless_bicycle": Cyclist,
        "pedestrian": Pedestrian,
        "background": Other,
        "static": Other,
    }

    _DEFAULT_SIZE = {
        "vehicle": (4.0, 2.0),
        "bus": (4.0, 2.0),
        "motorcyclist": (2.0, 0.7),
        "cyclist": (2.0, 0.7),
        "riderless_bicycle": (2.0, 0.7),
        "pedestrian": (0.5, 0.5),
        "background": (None, None),
        "static": (None, None),
    }

    _LANE_TYPE_MAPPING = {"VEHICLE": "road", "BIKE": "bicycle_lane"}

    _ROADLINE_TYPE_MAPPING = {
        "SOLID_WHITE": ["line_thin", "solid", "white"],
        "SOLID_YELLOW": ["line_thin", "solid", "yellow"],
        "SOLID_BLUE": ["line_thin", "solid", "blue"],
        "DASHED_WHITE": ["line_thin", "dashed", "white"],
        "DASHED_YELLOW": ["line_thin", "dashed", "yellow"],
        "SOLID_DASH_WHITE": ["line_thin", "solid_dash", "white"],
        "SOLID_DASH_YELLOW": ["line_thin", "solid_dash", "yellow"],
        "DASH_SOLID_WHITE": ["line_thin", "dash_solid", "white"],
        "DASH_SOLID_YELLOW": ["line_thin", "dash_solid", "yellow"],
        "DOUBLE_SOLID_WHITE": ["line_thick", "solid", "white"],
        "DOUBLE_SOLID_YELLOW": ["line_thick", "solid", "yellow"],
        "DOUBLE_DASH_WHITE": ["line_thick", "dashed", "white"],
        "DOUBLE_DASH_YELLOW": ["line_thick", "dashed", "yellow"],
        "NONE": ["virtual", None, None],
        "UNKNOWN": ["virtual", None, None],
    }

    def parse_trajectory(self, file: str, folder: str) -> Tuple[dict, Tuple[int, int]]:
        """This function parses trajectories from a single Argoverse parquet file. Because the duration of the scenario has been well articulated, the parser will not provide an option to select time range within a single scenario. The states were collected at 10Hz.

        Args:
            file (str): The name of the trajectory data file. The file is expected to be a parquet file.
            folder (str): The path to the folder containing the trajectory data.

        Returns:
            dict: A dictionary of participants. The keys are the ids of the participants. The values are the participants.
            Tuple[int, int]: The actual time range of the trajectory data. The first element is the start time. The second element is the end time. The unit of time stamp is millisecond.
        """
        participants = dict()
        actual_stamp_range = (np.inf, -np.inf)

        file_path = os.path.join(folder, file)
        df = pd.read_parquet(file_path, engine="fastparquet")

        for _, state_info in df.iterrows():
            if state_info["track_id"] not in participants:
                object_type = state_info["object_type"]
                participants[state_info["track_id"]] = self._CLASS_MAPPING[object_type](
                    id_=state_info["track_id"],
                    type_=self._TYPE_MAPPING[object_type],
                    length=self._DEFAULT_SIZE[object_type][0],
                    width=self._DEFAULT_SIZE[object_type][1],
                    trajectory=Trajectory(id_=state_info["track_id"], fps=10.0),
                )

            time_stamp = int(state_info["timestep"] * 100)
            actual_stamp_range = (
                min(actual_stamp_range[0], time_stamp),
                max(actual_stamp_range[1], time_stamp),
            )

            state = State(
                frame=time_stamp,
                x=state_info["position_x"],
                y=state_info["position_y"],
                heading=state_info["heading"],
                vx=state_info["velocity_x"],
                vy=state_info["velocity_y"],
            )

            participants[state_info["track_id"]].trajectory.append_state(state)

        return participants, actual_stamp_range

    def parse_map(self, file: str, folder: str) -> Map:
        """This function parses a map from a single Argoverse json file.

        Args:
            file (str): The name of the map file. The file is expected to be a json file (.json).
            folder (str): The path to the folder containing the map data.

        Returns:
            Map: A map object.
        """
        file_path = os.path.join(folder, file)

        with open(file_path, "r") as f:
            map_data = json.load(f)

        map_ = Map(name="argoverse_" + file.split(".")[0])

        if "drivable_areas" in map_data:
            for road_element in map_data["drivable_areas"].values():
                map_.add_area(
                    Area(
                        id_=str(road_element["id"]),
                        geometry=Polygon(
                            [[point["x"], point["y"]] for point in road_element["area_boundary"]]
                        ),
                        subtype="drivable_area",
                    )
                )

        roadline_id_counter = 0
        if "lane_segments" in map_data:
            for road_element in map_data["lane_segments"].values():
                left_type, left_subtype, left_color = self._ROADLINE_TYPE_MAPPING[
                    road_element["left_lane_mark_type"]
                ]
                left_road_line = RoadLine(
                    id_="%05d" % roadline_id_counter,
                    linestring=LineString(
                        [[point["x"], point["y"]] for point in road_element["left_lane_boundary"]]
                    ),
                    type_=left_type,
                    subtype=left_subtype,
                    color=left_color,
                )

                roadline_id_counter += 1

                right_type, right_subtype, right_color = self._ROADLINE_TYPE_MAPPING[
                    road_element["right_lane_mark_type"]
                ]
                right_road_line = RoadLine(
                    id_="%05d" % roadline_id_counter,
                    linestring=LineString(
                        [[point["x"], point["y"]] for point in road_element["right_lane_boundary"]]
                    ),
                    type_=right_type,
                    subtype=right_subtype,
                    color=right_color,
                )

                roadline_id_counter += 1

                lane = Lane(
                    id_=str(road_element["id"]),
                    left_side=left_road_line.geometry,
                    right_side=right_road_line.geometry,
                    line_ids=set([left_road_line.id_, right_road_line.id_]),
                    subtype=self._LANE_TYPE_MAPPING[road_element["lane_type"]],
                    location="urban",
                    custom_tags={"is_intersection": road_element["is_intersection"]},
                )
                lane.add_related_lane(road_element["predecessors"], LaneRelationship.PREDECESSOR)
                lane.add_related_lane(road_element["successors"], LaneRelationship.SUCCESSOR)
                lane.add_related_lane(
                    road_element["left_neighbor_id"], LaneRelationship.LEFT_NEIGHBOR
                )
                lane.add_related_lane(
                    road_element["right_neighbor_id"], LaneRelationship.RIGHT_NEIGHBOR
                )

                map_.add_roadline(left_road_line)
                map_.add_roadline(right_road_line)
                map_.add_lane(lane)

        if "pedestrian_crossings" in map_data:
            for road_element in map_data["pedestrian_crossings"].values():
                edge1 = [[point["x"], point["y"]] for point in road_element["edge1"]]
                edge2 = [[point["x"], point["y"]] for point in road_element["edge2"]]

                polygon = Polygon(edge1 + edge2)
                if not polygon.is_simple:
                    polygon = Polygon(edge1 + list(reversed(edge2)))

                map_.add_area(
                    Area(id_=str(road_element["id"]), geometry=polygon, subtype="crosswalk")
                )

        return map_
