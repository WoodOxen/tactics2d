# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""NGSIM dataset parser implementation."""


import os
from typing import Tuple

import numpy as np
import pandas as pd

from tactics2d.participant.element import Vehicle
from tactics2d.participant.trajectory import State, Trajectory


class NGSIMParser:
    """
    TODO: The support of NGSIM dataset is planned to be added before version 1.1.0.
    """

    def extract_meta_data(self, file: str, folder: str):
        df_track_chunk = pd.read_csv(os.path.join(folder, file), iterator=True, chunksize=10000)

        meta_data_dict = dict()

        for chunk in df_track_chunk:
            for _, info in chunk.iterrows():
                id_ = info["Vehicle_ID"]

                if id_ not in meta_data_dict:
                    meta_data_dict[id_] = dict()
                    meta_data_dict[id_]["v_Length"] = info["v_Length"]
                    meta_data_dict[id_]["v_Width"] = info["v_Width"]
                    meta_data_dict[id_]["v_Class"] = info["v_Class"]
                    meta_data_dict[id_]["v_Vel_Max"] = info["v_Vel"]
                    meta_data_dict[id_]["v_Vel_Min"] = info["v_Vel"]
                    meta_data_dict[id_]["v_Acc_Max"] = info["v_Acc"]
                    meta_data_dict[id_]["v_Acc_Min"] = info["v_Acc"]
                    meta_data_dict[id_]["Start_Frame"] = info["Frame_ID"]
                    meta_data_dict[id_]["End_Frame"] = info["Frame_ID"]
                    meta_data_dict[id_]["num_Lane_Change"] = 0
                    last_lane = info["Lane_ID"]
                else:
                    meta_data_dict[id_]["v_Vel_Max"] = np.max(
                        [meta_data_dict[id_]["v_Vel_Max"], info["v_Vel"]]
                    )
                    meta_data_dict[id_]["v_Vel_Min"] = np.min(
                        [meta_data_dict[id_]["v_Vel_Min"], info["v_Vel"]]
                    )
                    meta_data_dict[id_]["v_Acc_Max"] = np.max(
                        [meta_data_dict[id_]["v_Acc_Max"], info["v_Acc"]]
                    )
                    meta_data_dict[id_]["v_Acc_Min"] = np.min(
                        [meta_data_dict[id_]["v_Acc_Min"], info["v_Acc"]]
                    )
                    meta_data_dict[id_]["Start_Frame"] = np.min(
                        [meta_data_dict[id_]["Start_Frame"], info["Frame_ID"]]
                    )
                    meta_data_dict[id_]["End_Frame"] = np.max(
                        [meta_data_dict[id_]["End_Frame"], info["Frame_ID"]]
                    )

                    if last_lane != info["Lane_ID"]:
                        meta_data_dict[id_]["num_Lane_Change"] += 1
                        last_lane = info["Lane_ID"]

        meta_data_df = pd.DataFrame.from_dict(meta_data_dict, orient="index")
        meta_data_df.index.name = "Vehicle_ID"
        meta_data_df.index = meta_data_df.index.astype(int)

        file_meta = file.split(".")[0] + "-meta" + "." + file.split(".")[1]
        meta_data_df.to_csv(os.path.join(folder, file_meta))

    def parse_trajectory(
        self, file: str, folder: str, stamp_range: Tuple[int, int] = None, ids: list = None
    ) -> Tuple[dict, Tuple[int, int]]:
        """This function parses the trajectory data of the NGSIM dataset. The states were collected at 10Hz.

        Args:
            file (int):
            folder (str): The path to the folder containing the trajectory data.
            stamp_range (Tuple[int, int], optional): The time range of the trajectory data to parse. The unit of time stamp is millisecond. If the stamp range is not given, the parser will parse the whole trajectory data. Defaults to None.
            ids (list): The list of trajectory ids that needs to parse. If this value is not specified, the parser will parse all the trajectories within the time range. Defaults to None.

        Returns:
            participants (dict): A dictionary of participants. The keys are the ids of the participants. The values are the participants.
            actual_stamp_range (Tuple[int, int]): The actual time range of the trajectory data. The first element is the start time. The second element is the end time. The unit of time stamp is millisecond."
        """
        if stamp_range is None:
            stamp_range = (-np.inf, np.inf)

        # load the vehicles that have frame in the arbitrary range
        participants = dict()
        actual_stamp_range = (np.inf, -np.inf)

        if ids is not None:
            ids = {int(x) for x in ids}

        df_track_chunk = pd.read_csv(os.path.join(folder, file), iterator=True, chunksize=10000)

        for chunk in df_track_chunk:
            for idx, info in chunk.iterrows():
                time_stamp = info["Frame_ID"]

                if time_stamp < stamp_range[0] or time_stamp > stamp_range[1]:
                    continue

                actual_stamp_range = (
                    min(actual_stamp_range[0], time_stamp),
                    max(actual_stamp_range[1], time_stamp),
                )

                id_ = int(info["Vehicle_ID"])

                if ids is not None and id_ not in ids:
                    continue

                if id_ not in participants:
                    participants[id_] = Vehicle(
                        id_=info["Vehicle_ID"],
                        length=info["v_Length"] * 0.3048,  # feet -> meter
                        width=info["v_Width"] * 0.3048,  # feet -> meter
                        trajectory=Trajectory(id_=id_, fps=10),
                    )

                state = State(
                    frame=info["Frame_ID"] * 100,  # 10 Hz, ms
                    x=info["Global_X"] * 0.3048,  # feet -> meter
                    y=info["Global_Y"] * 0.3048,  # feet -> meter
                    speed=info["v_Vel"] * 0.3048,  # feet/s -> meter/s
                    accel=info["v_Acc"] * 0.3048,  # feet/s^2 -> meter/s^2
                )

                participants[id_].trajectory.add_state(state)

        return participants, actual_stamp_range
