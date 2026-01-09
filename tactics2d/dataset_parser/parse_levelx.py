# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Levelx datasets parser implementation."""


import os
import re
from typing import Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
from pyproj import Proj

from tactics2d.participant.element import Cyclist, Pedestrian, Vehicle
from tactics2d.participant.trajectory import State, Trajectory


class LevelXParser:
    """This class implements a parser for the series of datasets collected by the Institute for Automotive Engineering (ika) of RWTH Aachen University. Because the commercial version of the datasets are held by LevelXData, we call this series of datasets LevelX-series datasets. The datasets include: highD, inD, rounD, exiD, uniD.

    !!! quote "Reference"
        Krajewski, Robert, et al. "The highd dataset: A drone dataset of naturalistic vehicle trajectories on german highways for validation of highly automated driving systems." 2018 21st international conference on intelligent transportation systems (ITSC). IEEE, 2018.

        Bock, Julian, et al. "The ind dataset: A drone dataset of naturalistic road user trajectories at german intersections." 2020 IEEE Intelligent Vehicles Symposium (IV). IEEE, 2020.

        Krajewski, Robert, et al. "The round dataset: A drone dataset of road user trajectories at roundabouts in germany." 2020 IEEE 23rd International Conference on Intelligent Transportation Systems (ITSC). IEEE, 2020.

        Moers, Tobias, et al. "The exiD dataset: A real-world trajectory dataset of highly interactive highway scenarios in Germany." 2022 IEEE Intelligent Vehicles Symposium (IV). IEEE, 2022.
    """

    _REGISTERED_DATASET = ["highd", "ind", "round", "exid", "unid"]

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

    _EXID_SPECIAL_COLS = [
        "latLaneCenterOffset",
        "laneWidth",
        "laneletId",
        "laneChange",
        "lonLaneletPos",
        "laneletLength",
        "leadDHW",
        "leadDV",
        "leadTHW",
        "leadTTC",
        "leadId",
        "rearId",
        "leftLeadId",
        "leftRearId",
        "leftAlongsideId",
        "rightLeadId",
        "rightRearId",
        "rightAlongsideId",
        "odrRoadId",
        "odrSectionNo",
        "odrLaneId",
    ]

    def __init__(self, dataset: str):
        """Initialize the parser.

        Args:
            dataset (str): The dataset you want to parse. The available choices are: highD, inD, rounD, exiD, uniD, ignorant of letter case.
        """
        self.dataset = dataset.lower()

        if self.dataset not in self._REGISTERED_DATASET:
            raise KeyError(
                f"{dataset} is not an available LevelX-series dataset. The available datasets are {self._REGISTERED_DATASET}."
            )

        self.id_key = "id" if self.dataset == "highd" else "trackId"
        self.key_length = "width" if self.dataset == "highd" else "length"
        self.key_width = "height" if self.dataset == "highd" else "width"
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
            location_id (int): The id of the location.
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
            actual_stamp_range (Tuple[int, int]): The time range of the trajectory data. The first element is the start time. The second element is the end time. The unit of time stamp is millisecond.
        """
        file_id = self._get_file_id(file)
        df_track_meta = pd.read_csv(os.path.join(folder, "%02d_tracksMeta.csv" % file_id))
        start_frame = int(min(df_track_meta["initialFrame"]) * 40)
        end_frame = int(max(df_track_meta["finalFrame"]) * 40)

        actual_stamp_range = (start_frame, end_frame)
        return actual_stamp_range

    def parse_trajectory(
        self,
        file: Union[int, str],
        folder: str,
        stamp_range: Tuple[int, int] = None,
        ids: list = None,
    ) -> Tuple[dict, Tuple[int, int]]:
        """This function parses the trajectory data of LevelX-series datasets. The states were collected at 25Hz.

        Args:
            file (int): The id or the name of the trajectory file. If the input is an integer, the parser will parse the trajectory data from the following files: `%02d_tracks.csv % file` and `%02d_tracksMeta.csv % file`. If the input is a string, the parser will extract the integer id first and repeat the above process.
            folder (str): The path to the folder containing the trajectory data.
            stamp_range (Tuple[int, int], optional): The time range of the trajectory data to parse. The unit of time stamp is millisecond. If the stamp range is not given, the parser will parse the whole trajectory data. Defaults to None.
            ids (list): The list of trajectory ids that needs to parse. If this value is not specified, the parser will parse all the trajectories within the time range. Defaults to None.

        Returns:
            participants (dict): A dictionary of participants. The keys are the ids of the participants. The values are the participants.
            actual_stamp_range (Tuple[int, int]): The actual time range of the trajectory data. The first element is the start time. The second element is the end time. The unit of time stamp is millisecond.
        """
        if stamp_range is None:
            stamp_range = (-np.inf, np.inf)

        if ids is not None:
            ids = {int(x) for x in ids}

        # load the vehicles that have frame in the arbitrary range
        participants = dict()

        file_id = self._get_file_id(file)

        schema_overrides = (
            None
            if self.dataset != "exid"
            else {name: pl.String for name in self._EXID_SPECIAL_COLS}
        )
        df_track = pl.read_csv(
            os.path.join(folder, "%02d_tracks.csv" % file_id), schema_overrides=schema_overrides
        )
        df_track_meta = pd.read_csv(os.path.join(folder, "%02d_tracksMeta.csv" % file_id))
        df_meta = pd.read_csv(os.path.join(folder, "%02d_recordingMeta.csv" % file_id))

        # highD is record in a special way and needs to be calibrated
        # first get the calibration parameters
        if self.dataset == "highd":
            k, b = self._get_calibrate_params(df_meta)

        for _, participant_info in df_track_meta.iterrows():
            first_stamp = participant_info["initialFrame"] * 40  # ms
            last_stamp = participant_info["finalFrame"] * 40  # ms

            if last_stamp < stamp_range[0] or first_stamp > stamp_range[1]:
                continue

            id_ = participant_info[self.id_key]
            class_ = self._CLASS_MAPPING[participant_info["class"]]
            type_ = self._TYPE_MAPPING[participant_info["class"]]

            if ids is not None and id_ not in ids:
                continue

            participant = class_(
                id_=id_,
                type_=type_,
                length=participant_info[self.key_length],
                width=participant_info[self.key_width],
            )

            participants[id_] = participant

        participant_ids = set(participants.keys())

        # Filter trajectories following requirements
        df_track_filtered = df_track.filter(pl.col(self.id_key).is_in(participant_ids))
        df_track_filtered = df_track_filtered.with_columns(
            (pl.col("frame") * 40).alias("time_stamp")
        )
        df_track_filtered = df_track_filtered.filter(
            (pl.col("time_stamp") >= stamp_range[0]) & (pl.col("time_stamp") <= stamp_range[1])
        )

        if self.dataset == "highd":
            # This heading is for Tactics2D, using the common coordinate system.
            df_track_filtered = df_track_filtered.with_columns(
                pl.arctan2(pl.col("yVelocity").neg(), pl.col("xVelocity"))
                .round(5)
                .alias("heading_")
            )
        else:
            df_track_filtered = df_track_filtered.with_columns(
                (pl.col("heading") * 2 * pl.lit(np.pi) / 360).alias("heading_")
            )

        # Calibrate the coordinates of highD
        if self.dataset == "highd":
            # This theta is only for computing the center of bounding box.
            # theta is in [-pi/2, pi/2], because the x, y always denotes the upper left corner, and the coordinate system of highD is downward.
            df_track_filtered = df_track_filtered.with_columns(
                (pl.col("yVelocity") / pl.col("xVelocity")).arctan().round(5).alias("theta")
            )
            df_track_filtered = df_track_filtered.with_columns(
                (
                    pl.col("x")
                    + (pl.col(self.key_length) * pl.col("theta").cos()) / 2
                    - (pl.col(self.key_width) * pl.col("theta").sin()) / 2
                ).alias("xCenter"),
                (
                    pl.col("y")
                    + (pl.col(self.key_length) * pl.col("theta").sin()) / 2
                    + (pl.col(self.key_width) * pl.col("theta").cos()) / 2
                ).alias("yCenter"),
            )
            df_track_filtered = df_track_filtered.with_columns(
                (pl.col("yCenter") * k + b).alias("yCenter")
            )

        actual_stamp_range = (
            df_track_filtered.select(pl.col("time_stamp").min()).to_numpy()[0][0],
            df_track_filtered.select(pl.col("time_stamp").max()).to_numpy()[0][0],
        )

        grouped = df_track_filtered.group_by(self.id_key)
        trajectories = dict()

        for trajectory_id, group in grouped:
            trajectory_id = int(trajectory_id[0])
            if trajectory_id not in trajectories:
                trajectories[trajectory_id] = Trajectory(id_=trajectory_id, fps=25.0)

            time_stamp_idx = group.columns.index("time_stamp")
            x_center_idx = group.columns.index("xCenter")
            y_center_idx = group.columns.index("yCenter")
            heading_idx = group.columns.index("heading_")
            vx_idx = group.columns.index("xVelocity")
            vy_idx = group.columns.index("yVelocity")
            ax_idx = group.columns.index("xAcceleration")
            ay_idx = group.columns.index("yAcceleration")

            for state_info in group.rows():
                state = State(
                    state_info[time_stamp_idx],
                    x=state_info[x_center_idx],
                    y=state_info[y_center_idx],
                    heading=state_info[heading_idx],
                    vx=state_info[vx_idx],
                    vy=state_info[vy_idx],
                    ax=state_info[ax_idx],
                    ay=state_info[ay_idx],
                )

                trajectories[trajectory_id].add_state(state)

        for participant_id in participants.keys():
            participants[participant_id].bind_trajectory(trajectories[participant_id])

        return participants, actual_stamp_range

    def parse_map(self, **kwargs):
        """TODO: provide an API similar to other parsers to parse the map data. At present the map data are self-built and can be parsed by the Lanelet2Parser."""
        return
