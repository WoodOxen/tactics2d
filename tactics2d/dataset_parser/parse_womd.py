##! python3
# -*- coding: utf-8 -*-
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: parse_womd.py
# @Description: This file implements a parser for Waymo Open Motion Dataset v1.2.
# @Author: Yueyuan Li
# @Version: 1.0.0

import os
from typing import Tuple, List, Union

import numpy as np
from shapely.geometry import Point, LineString, Polygon
import tensorflow as tf

from tactics2d.participant.element import Vehicle, Pedestrian, Cyclist, Other
from tactics2d.participant.trajectory import State, Trajectory
from tactics2d.map.element import RoadLine, Lane, LaneRelationship, Area, Regulatory, Map
from tactics2d.dataset_parser.womd_proto import scenario_pb2


class WOMDParser:
    """This class implements a parser for Waymo Open Motion Dataset (WOMD).

    !!! info "Reference"
        Ettinger, Scott, et al. "Large scale interactive motion forecasting for autonomous driving: The waymo open motion dataset." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.

    Because loading the tfrecord file is time consuming, the trajectory and the map parsers provide two ways to load the file. The first way is to load the file directly from the given file path. The second way is to load the file from a tf.data.TFRecordDataset object. If the tf.data.TFRecordDataset object is given, the parser will ignore the file path.
    """

    _TYPE_MAPPING = {0: "unknown", 1: "vehicle", 2: "pedestrian", 3: "cyclist", 4: "other"}

    _CLASS_MAPPING = {0: Other, 1: Vehicle, 2: Pedestrian, 3: Cyclist, 4: Other}

    _ROADLINE_TYPE_MAPPING = {
        0: ["virtual", None, None],
        1: ["line_thin", "dashed", "white"],
        2: ["line_thin", "solid", "white"],
        3: ["line_thin", "solid_solid", "white"],
        4: ["line_thin", "dashed", "yellow"],
        5: ["line_thin", "dashed_dashed", "yellow"],
        6: ["line_thin", "solid", "yellow"],
        7: ["line_thin", "solid_solid", "yellow"],
        8: [None, "dashed", "yellow"],
    }

    _LANE_TYPE_MAPPING = {0: "road", 1: "highway", 2: "road", 3: "bicycle_lane"}

    def get_scenario_ids(self, dataset) -> List[str]:
        """This function get the list of scenario ids from the given tfrecord file.

        Args:
            dataset (tf.data.TFRecordDataset): The dataset to parse.

        Returns:
            List[str]: A list of scenario ids looking like ["637f20cafde22ff8", ...].
        """

        id_list = []

        for data in dataset:
            proto_string = data.numpy()
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(proto_string)
            id_list.append(scenario.scenario_id)

        return id_list

    def parse_trajectory(
        self, scenario_id: Union[str, int] = None, **kwargs
    ) -> Tuple[dict, List[int]]:
        """This function parses trajectories from a single WOMD file. Because the duration of the scenario has been well articulated, the parser will not provide an option to select time range within a single scenario. The states were collected at 10Hz.

        Args:
            scenario_id (Union[str, int], optional): The id of the scenario to parse. If the scenario id is a string, the parser will search for the scenario id in the file. If the scenario id is an integer, the parser will parse `scenario_id`-th scenario in the file. If the scenario id is None or is not found, the first scenario in the file will be parsed.

        Keyword Args:
            dataset (tf.data.TFRecordDataset, optional): The dataset to parse.
            file (str, optional): The name of the trajectory file. The file is expected to be a tfrecord file (.tfrecord).
            folder (str, optional): The path to the folder containing the tfrecord file.

        Returns:
            dict: A dictionary of participants. If the scenario id is not found, return None.
            List[int]: The actual time range of the trajectory data. Because WOMD collects data at an unstable frequency, the parser will return a list of time stamps.

        Raises:
            KeyError: Either dataset or file and folder should be given as keyword arguments.
        """
        participants = dict()
        time_stamps = set()

        if "dataset" in kwargs:
            dataset = kwargs["dataset"]
        elif "file" in kwargs and "folder" in kwargs:
            file_path = os.path.join(kwargs["folder"], kwargs["file"])
            dataset = tf.data.TFRecordDataset(file_path, compression_type="")
        else:
            raise KeyError(
                "Either dataset or file and folder should be given as keyword arguments."
            )

        scenario_ids = self.get_scenario_ids(dataset)

        data_id = 0
        if isinstance(scenario_id, str):
            if scenario_id in scenario_ids:
                data_id = scenario_ids.index(scenario_id)
        elif isinstance(scenario_id, int):
            data_id = scenario_id % len(scenario_ids)

        data = None
        cnt = 0
        for data in dataset:
            if cnt == data_id:
                break
            cnt += 1

        proto_string = data.numpy()
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(proto_string)

        timestamps = scenario.timestamps_seconds
        for track in scenario.tracks:
            trajectory = Trajectory(id_=track.id, fps=10, stable_freq=False)
            width = 0
            length = 0
            height = 0
            cnt = 0
            for i, state_ in enumerate(track.states):
                if not state_.valid:
                    continue
                state = State(
                    frame=int(timestamps[i] * 1000),
                    x=state_.center_x,
                    y=state_.center_y,
                    heading=state_.heading,
                    vx=state_.velocity_x,
                    vy=state_.velocity_y,
                )
                time_stamps.add(state.frame)

                trajectory.append_state(state)

                width += state_.width
                length += state_.length
                height += state_.height
                cnt += 1

            participant = self._CLASS_MAPPING[track.object_type](
                id_=track.id,
                type_=self._TYPE_MAPPING[track.object_type],
                trajectory=trajectory,
                length=length / cnt,
                width=width / cnt,
                height=height / cnt,
            )
            participants[track.id] = participant

        actual_time_range = sorted(list(time_stamps))

        return participants, actual_time_range

    def _join_lane_boundary(self, ids, map_):
        points = []
        for id_ in ids:
            line = map_.roadlines[id_].shape
            if len(points) == 0:
                points = line
            else:
                if points[-1] == line[0]:
                    points.extend(line[1:])
                elif points[-1] == line[-1]:
                    points.extend(line[::-1][1:])
                elif np.linalg.norm(points[-1] - line[0]) < np.linalg.norm(points[-1] - line[-1]):
                    points.extend(line[1:])
                else:
                    points.extend(line[::-1][1:])

        return LineString(points)

    def _parse_map_features(self, map_feature, map_: Map):
        if map_feature.HasField("lane"):
            lane = map_feature.lane
            left_side_ids = set()
            right_side_ids = set()
            line_ids = set()

            for boundary in lane.left_boundaries:
                id_ = "%05d" % boundary.boundary_feature_id
                line_ids.add(id_)
                left_side_ids.add(id_)

            for boundary in map_feature.lane.right_boundaries:
                id_ = "%05d" % boundary.boundary_feature_id
                line_ids.add(id_)
                right_side_ids.add(id_)

            lane = Lane(
                id_="%05d" % map_feature.id,
                left_side=self._join_lane_boundary(left_side_ids, map_),
                right_side=self._join_lane_boundary(right_side_ids, map_),
                subtype=self._LANE_TYPE_MAPPING[lane.type],
                speed_limit=lane.speed_limit_mph,
                speed_limit_unit="mi/h",
            )

            if getattr(lane, "entry_lanes"):
                for entry_lane in lane.entry_lanes:
                    lane.add_related_lane("%05d" % entry_lane, LaneRelationship.PREDECESSOR)
            if getattr(lane, "exit_lanes"):
                for exit_lane in lane.exit_lanes:
                    lane.add_related_lane("%05d" % exit_lane, LaneRelationship.SUCCESSOR)
            if getattr(lane, "left_neighbors"):
                for left_neighbor in lane.left_neighbors:
                    lane.add_related_lane("%05d" % left_neighbor, LaneRelationship.LEFT_NEIGHBOR)
            if getattr(lane, "right_neighbors"):
                for right_neighbor in lane.right_neighbors:
                    lane.add_related_lane("%05d" % right_neighbor, LaneRelationship.RIGHT_NEIGHBOR)

        elif map_feature.HasField("road_line"):
            type_, subtype, color = self._ROADLINE_TYPE_MAPPING[map_feature.road_line.type]
            points = [[point.x, point.y] for point in map_feature.road_line.polyline]
            if len(points) > 2:
                roadline = RoadLine(
                    id_="%05d" % map_feature.id,
                    linestring=LineString(points),
                    type_=type_,
                    subtype=subtype,
                    color=color,
                )
                map_.add_roadline(roadline)

        elif map_feature.HasField("road_edge"):
            roadline = RoadLine(
                id_="%05d" % map_feature.id,
                linestring=LineString(
                    [[point.x, point.y] for point in map_feature.road_edge.polyline]
                ),
                type_="road_boarder",
            )
            map_.add_roadline(roadline)

        elif map_feature.HasField("stop_sign"):
            stop_sign = Regulatory(
                id_="%05d" % map_feature.id,
                way_ids=set(["%05d" % way_id for way_id in map_feature.stop_sign.lane]),
                subtype="stop_sign",
                position=Point(map_feature.stop_sign.position.x, map_feature.stop_sign.position.y),
            )
            map_.add_regulatory(stop_sign)

        elif map_feature.HasField("crosswalk"):
            crosswalk = Area(
                id_="%05d" % map_feature.id,
                geometry=Polygon([[point.x, point.y] for point in map_feature.crosswalk.polygon]),
                subtype="crosswalk",
            )
            map_.add_area(crosswalk)

        elif map_feature.HasField("speed_bump"):
            speed_bump = Area(
                id_="%05d" % map_feature.id,
                geometry=Polygon([[point.x, point.y] for point in map_feature.speed_bump.polygon]),
                subtype="speed_bump",
            )
            map_.add_area(speed_bump)

        elif map_feature.HasField("driveway"):
            area = Area(
                id_="%05d" % map_feature.id,
                geometry=Polygon([[point.x, point.y] for point in map_feature.driveway.polygon]),
                subtype="free_space",
            )
            map_.add_area(area)

    def _parse_dynamic_map_features(self, map_feature, map_: Map):
        return

    def parse_map(self, scenario_id=None, **kwargs) -> Map:
        """This function parses the map from a single WOMD file.

        Args:
            scenario_id (str, optional): The id of the scenario to parse. If the scenario id is not given, the first scenario in the file will be parsed.

        Keyword Args:
            dataset (tf.data.TFRecordDataset, optional): The dataset to parse.
            file (str, optional): The name of the trajectory file. The file is expected to be a tfrecord file (.tfrecord).
            folder (str, optional): The path to the folder containing the tfrecord file.

        Returns:
            Map: A map object.

        Raises:
            KeyError: Either dataset or file and folder should be given as keyword arguments.
        """

        if "dataset" in kwargs:
            dataset = kwargs["dataset"]
        elif "file" in kwargs and "folder" in kwargs:
            file_path = os.path.join(kwargs["folder"], kwargs["file"])
            dataset = tf.data.TFRecordDataset(file_path, compression_type="")
        else:
            raise KeyError(
                "Either dataset or file and folder should be given as keyword arguments."
            )

        for data in dataset:
            proto_string = data.numpy()
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(proto_string)

            if scenario_id is None:
                scenario_id = scenario.scenario_id

            if scenario_id == scenario.scenario_id:
                map_ = Map(name="womd_" + scenario.scenario_id)
                for map_feature in scenario.map_features:
                    if map_feature.HasField("road_line") or map_feature.HasField("road_edge"):
                        self._parse_map_features(map_feature, map_)
                for map_feature in scenario.map_features:
                    if not map_feature.HasField("road_line") and not map_feature.HasField(
                        "road_edge"
                    ):
                        self._parse_dynamic_map_features(map_feature, map_)
                for dynamic_map_state in scenario.dynamic_map_states:
                    self._parse_dynamic_map_features(dynamic_map_state, map_)

        return map_
