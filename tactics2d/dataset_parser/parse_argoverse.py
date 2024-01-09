import os
import json

import sys

sys.path.append("../../")

import pandas as pd
from shapely.geometry import Polygon, LineString

from tactics2d.participant.element import Vehicle, Pedestrian, Cyclist
from tactics2d.trajectory.element import State, Trajectory
from tactics2d.map.element import Area, RoadLine, Lane, LaneRelationship, CrossWalk, Map


class ArgoverseParser:
    """This class implements a parser for Argoverse dataset. The default size of the participants are referred to the [official visualization toolkit](https://github.com/argoverse/av2-api/blob/main/src/av2/datasets/motion_forecasting/viz/scenario_visualization.py).

    Wilson, Benjamin, et al. "Argoverse 2: Next generation datasets for self-driving perception and forecasting." arXiv preprint arXiv:2301.00493 (2023).
    """

    TYPE_MAPPING = {
        "vehicle": "car",
        "bus": "bus",
        "motorcycle": "motorcycle",
        "cyclist": "bicycle",
        "riderless_bicycle": "bicycle",
        "pedestrian": "pedestrian",
    }

    CLASS_MAPPING = {
        "vehicle": Vehicle,
        "bus": Vehicle,
        "motorcycle": Cyclist,
        "cyclist": Cyclist,
        "riderless_bicycle": Cyclist,
        "pedestrian": Pedestrian,
    }

    DEFAULT_SIZE = {
        "car": (4.0, 2.0),
        "bus": (4.0, 2.0),
        "motorcycle": (2.0, 0.7),
        "bicycle": (2.0, 0.7),
        "rideless_bicycle": (2.0, 0.7),
    }

    LANE_TYPE_MAPPING = {"VEHICLE": ["vehicle"], "BICYCLE": ["bicycle"]}

    MARK_TYPE_MAPPING = {
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

    def parse_trajectory(self, file_name: str, folder_path: str):
        """This function parses trajectories from a single Argoverse parquet file. Because the duration of the scenario is well articulated, the parser will not provide an option to parse a subset of time range of the scenario. The states were collected at 10Hz.

        Args:
            file_name (str): _description_
            folder_path (str): _description_
        """
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_parquet(file_path, engine="fastparquet")

        participants = dict()

        for _, state_info in df.iterrows():
            if state_info["track_id"] not in participants:
                participants[state_info["track_id"]] = self.CLASS_MAPPING(
                    id_=state_info["track_id"],
                    type_=self.TYPE_MAPPING[state_info["object_type"]],
                    length=self.DEFAULT_SIZE[state_info["object_type"]][0],
                    width=self.DEFAULT_SIZE[state_info["object_type"]][1],
                    trajectory=Trajectory(id_=state_info["track_id"], fps=10.0),
                )

            state = State(
                frame=state_info["timestep"],
                x=state_info["position_x"],
                y=state_info["position_y"],
                heading=state_info["heading"],
                vx=state_info["velocity_x"],
                vy=state_info["velocity_y"],
            )

            participants[state_info["track_id"]].trajectory.append_state(state)

        return participants

    def parse_map(self, file_name: str, folder_path: str) -> Map:
        """This function parses a map from a single Argoverse json file.

        Args:
            file_name (str): _description_
            folder_path (str): _description_

        Returns:
            Map: _description_
        """
        file_path = os.path.join(folder_path, file_name)

        with open(file_path, "r") as f:
            map_data = json.load(f)

        map_ = Map()

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
                left_type, left_subtype, left_color = self.MARK_TYPE_MAPPING["left_lane_mark_type"]
                left_road_line = RoadLine(
                    id_="%04d" % roadline_id_counter,
                    linestring=LineString(
                        [[point["x"], point["y"]] for point in road_element["left_lane_boundary"]]
                    ),
                    type_=left_type,
                    subtype=left_subtype,
                    color=left_color,
                )

                right_type, right_subtype, right_color = self.MARK_TYPE_MAPPING[
                    "right_lane_mark_type"
                ]
                right_road_line = RoadLine(
                    id_="%04d" % roadline_id_counter,
                    linestring=LineString(
                        [[point["x"], point["y"]] for point in road_element["right_lane_boundary"]]
                    ),
                    type_=right_type,
                    subtype=right_subtype,
                    color=right_color,
                )

                lane = Lane(
                    id_=str(road_element["id"]),
                    left_side=left_road_line,
                    right_side=right_road_line,
                    line_ids=set(left_road_line.id_, right_road_line.id_),
                    subtype="road",
                    location="urban",
                    inferred_participants=self.LANE_TYPE_MAPPING[road_element["lane_type"]],
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
                left_edge = LineString(
                    [[point["x"], point["y"]] for point in road_element["left_edge"]]
                )
                right_edge = LineString(
                    [[point["x"], point["y"]] for point in road_element["right_edge"]]
                )
                map_.add_regulatory_element(
                    CrossWalk(
                        id_=str(road_element["id"]), left_side=left_edge, right_side=right_edge
                    )
                )

        return map_
