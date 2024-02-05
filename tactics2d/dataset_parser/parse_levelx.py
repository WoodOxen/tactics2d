##! python3
# -*- coding: utf-8 -*-
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: parse_levelx.py
# @Description: This file implements a parser for highD, inD, rounD, exiD, uniD datasets.
# @Author: Yueyuan Li
# @Version: 1.0.0

import os
from typing import Tuple, Union
import re
import math

# import xml.etree.ElementTree as ET
from pyproj import Proj
import pandas as pd
import numpy as np

from tactics2d.participant.element import Vehicle, Pedestrian, Cyclist
from tactics2d.trajectory.element import State, Trajectory

# from tactics2d.map.parser import Lanelet2Parser


class LevelXParser:
    """This class implements a parser for the series of datasets collected by the Institute for Automotive Engineering (ika) of RWTH Aachen University. Because the commercial version of the datasets are held by LevelXData, we call this series of datasets LevelX-series datasets. The datasets include: highD, inD, rounD, exiD, uniD.

    !!! info "Reference"
        Krajewski, Robert, et al. "The highd dataset: A drone dataset of naturalistic vehicle trajectories on german highways for validation of highly automated driving systems." 2018 21st international conference on intelligent transportation systems (ITSC). IEEE, 2018.

        Bock, Julian, et al. "The ind dataset: A drone dataset of naturalistic road user trajectories at german intersections." 2020 IEEE Intelligent Vehicles Symposium (IV). IEEE, 2020.

        Krajewski, Robert, et al. "The round dataset: A drone dataset of road user trajectories at roundabouts in germany." 2020 IEEE 23rd International Conference on Intelligent Transportation Systems (ITSC). IEEE, 2020.

        Moers, Tobias, et al. "The exiD dataset: A real-world trajectory dataset of highly interactive highway scenarios in Germany." 2022 IEEE Intelligent Vehicles Symposium (IV). IEEE, 2022.

        Bock, Julian, et al. "Highly accurate scenario and reference data for automated driving." ATZ worldwide 123.5 (2021): 50-55.
    """

    _REGISTERED_DATASET = ["highD", "inD", "rounD", "exiD", "uniD"]

    _TYPE_MAPPING = {
        "car": "car",
        "Car": "car",
        "van": "van",
        "truck": "truck",
        "Truck": "truck",
        "truck_bus": "bus",
        "bus": "bus",
        "trailer": "trailer",
        "motorcycle": "motorcycle",
        "bicycle": "bicycle",
        "cycle": "bicycle",
        "pedestrian": "pedestrian",
    }

    _CLASS_MAPPING = {
        "car": Vehicle,
        "Car": Vehicle,
        "van": Vehicle,
        "truck": Vehicle,
        "Truck": Vehicle,
        "truck_bus": Vehicle,
        "bus": Vehicle,
        "trailer": Vehicle,
        "motorcycle": Cyclist,
        "bicycle": Cyclist,
        "cycle": Cyclist,
        "pedestrian": Pedestrian,
    }

    _HIGHD_BOUNDS = {
        1: [-0.00025899967, 0],
        2: [-0.00018397412, 0],
        3: [-0.00021942279, 0],
        4: [-0.00024320481, 0],
        5: [-0.00018558951, 0],
        6: [-0.00024051251, 0.0000336538],
    }

    def __init__(self, dataset: str = ""):
        """Initialize the parser.

        Args:
            dataset (str, optional): The dataset you want to parse. The available choices are: highD, inD, rounD, exiD, uniD. Defaults to "".
        """
        if dataset not in self._REGISTERED_DATASET:
            raise KeyError(
                f"{dataset} is not an available LevelX-series dataset. The available datasets are {self._REGISTERED_DATASET}."
            )

        self.dataset = dataset
        self.id_key = "id" if self.dataset == "highD" else "trackId"
        self.key_length = "width" if self.dataset == "highD" else "length"
        self.key_width = "height" if self.dataset == "highD" else "width"
        self.highd_projector = Proj(proj="utm", ellps="WGS84", zone=31, datum="WGS84")
        self.highd_origin = [0.01, 0]

    def _get_calibrate_params(self, df_meta: pd.DataFrame):
        location = int(df_meta.iloc[0]["locationId"])
        _, lower_bound = self.highd_projector(0, self._HIGHD_BOUNDS[location][0])
        _, upper_bound = self.highd_projector(0, self._HIGHD_BOUNDS[location][1])
        lower_lane_markings = [float(x) for x in df_meta.iloc[0]["lowerLaneMarkings"].split(";")]
        upper_lane_markings = [float(x) for x in df_meta.iloc[0]["upperLaneMarkings"].split(";")]
        local_lower = lower_lane_markings[-1]
        local_upper = upper_lane_markings[0]

        k = (upper_bound - lower_bound) / (local_upper - local_lower)
        b = upper_bound - k * local_upper

        return k, b

    def _get_file_id(self, file: Union[int, str]):
        if isinstance(file, str):
            file_id = int(re.findall(r"\d+", file)[0])
        elif isinstance(file, int):
            file_id = file
        else:
            raise TypeError("The input file must be an integer or a string.")

        return file_id

    def get_location(self, file: Union[int, str], folder: str) -> int:
        """This function retrieves the location from which a trajectory data file is obtained.

        Args:
            file (Union[int, str]): The id or the name of the trajectory file. If the input is an integer, the parser will parse the trajectory data from the following files: `%02d_recordingMeta.csv % file`. If the input is a string, the parser will extract the integer id first and repeat the above process.
            folder (str): The path to the folder containing the trajectory data.

        Returns:
            int: The id of the location.
        """
        file_id = self._get_file_id(file)
        df_meta = pd.read_csv(os.path.join(folder, "%02d_recordingMeta.csv" % file_id))

        return df_meta.iloc[0]["locationId"]

    def get_stamp_range(self, file: Union[int, str], folder: str) -> Tuple[int, int]:
        """This function gets the time range of a single trajectory data file.

        Args:
            file (Union[int, str]): The id or the name of the trajectory file. If the input is an integer, the parser will parse the trajectory data from the following files: `%02d_tracksMeta.csv" % file`. If the input is a string, the parser will extract the integer id first and repeat the above process.
            folder (str): The path to the folder containing the trajectory data.

        Returns:
            Tuple[int, int]: The time range of the trajectory data. The first element is the start time. The second element is the end time. The unit of time stamp is millisecond.
        """
        file_id = self._get_file_id(file)
        df_track_meta = pd.read_csv(os.path.join(folder, "%02d_tracksMeta.csv" % file_id))
        start_frame = int(min(df_track_meta["initialFrame"]) * 40)
        end_frame = int(max(df_track_meta["finalFrame"]) * 40)

        return start_frame, end_frame

    def parse_trajectory(
        self, file: Union[int, str], folder: str, stamp_range: Tuple[int, int] = None
    ) -> Tuple[dict, Tuple[int, int]]:
        """This function parses the trajectory data of LevelX-series datasets. The states were collected at 25Hz.

        Args:
            file (int): The id or the name of the trajectory file. If the input is an integer, the parser will parse the trajectory data from the following files: `%02d_tracks.csv % file` and `%02d_tracksMeta.csv % file`. If the input is a string, the parser will extract the integer id first and repeat the above process.
            folder (str): The path to the folder containing the trajectory data.
            stamp_range (Tuple[int, int], optional): The time range of the trajectory data to parse. The unit of time stamp is millisecond. If the stamp range is not given, the parser will parse the whole trajectory data. Defaults to None.

        Returns:
            dict: A dictionary of participants. The keys are the ids of the participants. The values are the participants.
            Tuple[int, int]: The actual time range of the trajectory data. The first element is the start time. The second element is the end time. The unit of time stamp is millisecond.
        """
        if stamp_range is None:
            stamp_range = (-np.inf, np.inf)

        # load the vehicles that have frame in the arbitrary range
        participants = dict()
        actual_stamp_range = (np.inf, -np.inf)

        file_id = self._get_file_id(file)

        df_track_chunk = pd.read_csv(
            os.path.join(folder, "%02d_tracks.csv" % file_id), iterator=True, chunksize=10000
        )
        df_track_meta = pd.read_csv(os.path.join(folder, "%02d_tracksMeta.csv" % file_id))
        df_meta = pd.read_csv(os.path.join(folder, "%02d_recordingMeta.csv" % file_id))

        # highD is record in a special way and needs to be calibrated
        # first get the calibration parameters
        if self.dataset == "highD":
            k, b = self._get_calibrate_params(df_meta)

        for _, participant_info in df_track_meta.iterrows():
            first_stamp = participant_info["initialFrame"] * 40  # ms
            last_stamp = participant_info["finalFrame"] * 40  # ms

            if last_stamp < stamp_range[0] or first_stamp > stamp_range[1]:
                continue

            id_ = participant_info[self.id_key]
            class_ = self._CLASS_MAPPING[participant_info["class"]]
            type_ = self._TYPE_MAPPING[participant_info["class"]]

            participant = class_(
                id_=id_,
                type_=type_,
                length=participant_info[self.key_length],
                width=participant_info[self.key_width],
            )

            participants[id_] = participant

        participant_ids = set(participants.keys())

        # parse the corresponding trajectory to each participant and bind them
        trajectories = dict()

        for chunk in df_track_chunk:
            chunk_ids = set(pd.unique(chunk[self.id_key]))
            if len(chunk_ids.union(participant_ids)) == 0:
                continue

            if self.dataset == "highD":
                chunk["heading_"] = np.round(np.arctan2(-chunk["yVelocity"], chunk["xVelocity"]), 5)
            else:
                chunk["heading_"] = chunk["heading"] * 2 * math.pi / 360

            chunk["time_stamp"] = chunk["frame"] * 40

            # calibrate the coordinates of highD
            if self.dataset == "highD":
                headings = np.round(
                    np.arctan(chunk["yVelocity"].copy(), chunk["xVelocity"].copy()), 5
                )
                xCenter = (
                    chunk["x"]
                    + chunk[self.key_length] * np.cos(headings) / 2
                    - chunk[self.key_width] * np.sin(headings) / 2
                )
                yCenter = (
                    chunk["y"]
                    + chunk[self.key_length] * np.sin(headings) / 2
                    + chunk[self.key_width] * np.cos(headings) / 2
                )

                chunk["xCenter"] = xCenter
                chunk["yCenter"] = k * yCenter + b

            for _, state_info in chunk.iterrows():
                time_stamp = state_info["time_stamp"]

                if time_stamp < stamp_range[0] or time_stamp > stamp_range[1]:
                    continue

                actual_stamp_range = (
                    min(actual_stamp_range[0], time_stamp),
                    max(actual_stamp_range[1], time_stamp),
                )

                trajectory_id = int(state_info[self.id_key])
                if trajectory_id not in trajectories:
                    trajectories[trajectory_id] = Trajectory(id_=trajectory_id, fps=25.0)

                state = State(
                    time_stamp,
                    x=state_info["xCenter"],
                    y=state_info["yCenter"],
                    heading=state_info["heading_"],
                    vx=state_info["xVelocity"],
                    vy=state_info["yVelocity"],
                    ax=state_info["xAcceleration"],
                    ay=state_info["yAcceleration"],
                )

                trajectories[trajectory_id].append_state(state)

        for participant_id in participants.keys():
            participants[participant_id].bind_trajectory(trajectories[participant_id])

        return participants, actual_stamp_range

    def parse_map(self, **kwargs):
        """TODO: provide an API similar to other parsers to parse the map data. At present the map data are self-built and can be parsed by the Lanelet2Parser."""
        return

    # map_ = Lanelet2Parser.parse()
