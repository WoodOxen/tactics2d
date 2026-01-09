# Copyright (C) 2025, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""CitySim dataset parser implementation."""


import logging
import os
from typing import Tuple

import numpy as np
import polars as pl

from tactics2d.participant.element import Vehicle
from tactics2d.participant.trajectory import State, Trajectory


class CitySimParser:
    """This class implements a parser of the CitySim Dataset.

    !!! quote "Reference"
        Zheng, Ou, et al. "CitySim: A drone-based vehicle trajectory dataset for safety-oriented research and digital twins." Transportation research record 2678.4 (2024): 606-621.
    """

    def get_stamp_range(self, file: str, folder: str) -> Tuple[int, int]:
        """This function gets the time range of a single trajectory data file.

        Args:
            file (Union[int, str]): The name of the trajectory data file. The file is expected to be a csv file.
            folder (str): The path to the folder containing the trajectory data.

        Returns:
            actual_stamp_range (Tuple[int, int]): The time range of the trajectory data. The first element is the start time. The second element is the end time. The unit of time stamp is millisecond.
        """
        df_track = pl.read_csv(os.path.join(folder, file))

        start_frame = min(df_track["frameNum"])
        end_frame = max(df_track["frameNum"])

        actual_stamp_range = (int(start_frame * 1000 / 30), int(end_frame * 1000 / 30))
        return actual_stamp_range

    def parse_trajectory(
        self, file: str, folder: str, stamp_range: Tuple[int, int] = None, ids: list = None
    ) -> Tuple[dict, Tuple[int, int]]:
        """This function parses the trajectory data of CitySim datasets. The states were collected at 30Hz.

        Args:
            file (str): The name of the trajectory data file. The file is expected to be a csv file.
            folder (str): The path to the folder containing the trajectory data.
            stamp_range (Tuple[int, int], optional): The time range of the trajectory data to parse. The unit of time stamp is millisecond. If the stamp range is not given, the parser will parse the whole trajectory data. Defaults to None.
            ids (list, optional): The list of trajectory ids that needs to parse. If this value is not specified, the parser will parse all the trajectories within the time range. Defaults to None.

        Returns:
            participants (dict): A dictionary of participants. The keys are the ids of the participants. The values are the participants.
            actual_stamp_range (Tuple[int, int]): The actual time range of the trajectory data. The first element is the start time. The second element is the end time. The unit of time stamp is millisecond.
        """
        if stamp_range is None:
            stamp_range = (-np.inf, np.inf)

        df_track = pl.read_csv(os.path.join(folder, file))

        # Frame number to milliseconds (30Hz -> 1000ms/30 = frame duration)
        df_track = df_track.with_columns(
            (pl.col("frameNum") * 1000 / 30).cast(pl.Int64).alias("frameNum (ms)")
        )

        # Filter participant IDs
        if ids is not None:
            participant_ids = {int(x) for x in ids}
        else:
            participant_ids = set(df_track.select("carId").unique().to_series().to_list())

        # Filter by carId and time range
        df_track_filtered = df_track.filter(
            (pl.col("carId").is_in(participant_ids))
            & (pl.col("frameNum (ms)") >= stamp_range[0])
            & (pl.col("frameNum (ms)") <= stamp_range[1])
        )

        # Add converted columns
        df_track_filtered = df_track_filtered.with_columns(
            [
                (pl.col("carCenterXft") * 0.3048).alias("carCenterX (m)"),
                (pl.col("carCenterYft") * 0.3048).alias("carCenterY (m)"),
                (pl.col("boundingBox1Xft") * 0.3048).alias("boundingBox1X (m)"),
                (pl.col("boundingBox1Yft") * 0.3048).alias("boundingBox1Y (m)"),
                (pl.col("boundingBox2Xft") * 0.3048).alias("boundingBox2X (m)"),
                (pl.col("boundingBox2Yft") * 0.3048).alias("boundingBox2Y (m)"),
                (pl.col("boundingBox3Xft") * 0.3048).alias("boundingBox3X (m)"),
                (pl.col("boundingBox3Yft") * 0.3048).alias("boundingBox3Y (m)"),
                (pl.col("boundingBox4Xft") * 0.3048).alias("boundingBox4X (m)"),
                (pl.col("boundingBox4Yft") * 0.3048).alias("boundingBox4Y (m)"),
                (pl.col("course") * np.pi / 180).alias("heading (radian)"),
                (pl.col("speed") * 0.447).alias("speed (m/s)"),
            ]
        )

        # Extract actual timestamp range
        actual_stamp_range = (
            df_track_filtered.select(pl.col("frameNum (ms)").min())[0, 0],
            df_track_filtered.select(pl.col("frameNum (ms)").max())[0, 0],
        )

        # Initialize participants and trajectories
        participants = dict()
        trajectories = dict()

        grouped = df_track_filtered.group_by("carId")

        for trajectory_id, group in grouped:
            trajectory_id = int(trajectory_id[0])

            # Initialize vehicle
            if trajectory_id not in participants:
                bx1 = group["boundingBox1X (m)"][0]
                by1 = group["boundingBox1Y (m)"][0]
                bx2 = group["boundingBox2X (m)"][0]
                by2 = group["boundingBox2Y (m)"][0]
                bx3 = group["boundingBox3X (m)"][0]
                by3 = group["boundingBox3Y (m)"][0]
                bx4 = group["boundingBox4X (m)"][0]
                by4 = group["boundingBox4Y (m)"][0]

                b1 = np.sqrt((bx2 - bx1) ** 2 + (by2 - by1) ** 2)
                b2 = np.sqrt((bx3 - bx2) ** 2 + (by3 - by2) ** 2)
                b3 = np.sqrt((bx4 - bx3) ** 2 + (by4 - by3) ** 2)
                b4 = np.sqrt((bx1 - bx4) ** 2 + (by1 - by4) ** 2)

                if np.abs(b3 - b1) > 0.1 or np.abs(b4 - b2) > 0.1:
                    logging.warning(f"The shape of Vehicle {trajectory_id} is not a rectangle.")

                a = (b1 + b3) / 2
                b = (b2 + b4) / 2

                participants[trajectory_id] = Vehicle(
                    id_=trajectory_id, length=max(a, b), width=min(a, b)
                )

            # Initialize trajectory
            if trajectory_id not in trajectories:
                trajectories[trajectory_id] = Trajectory(
                    id_=trajectory_id, fps=30.0, stable_freq=False
                )

            time_stamp_idx = group.columns.index("frameNum (ms)")
            x_center_idx = group.columns.index("carCenterX (m)")
            y_center_idx = group.columns.index("carCenterY (m)")
            heading_idx = group.columns.index("heading (radian)")
            speed_idx = group.columns.index("speed (m/s)")

            for state_info in group.rows():
                state = State(
                    state_info[time_stamp_idx],
                    x=state_info[x_center_idx],
                    y=state_info[y_center_idx],
                    heading=state_info[heading_idx],
                    speed=state_info[speed_idx],
                )

                trajectories[trajectory_id].add_state(state)

        for participant_id in participants.keys():
            participants[participant_id].bind_trajectory(trajectories[participant_id])

        return participants, actual_stamp_range
