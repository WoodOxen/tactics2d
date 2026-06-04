# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""INTERACTION dataset parser implementation."""


import os
import re
from typing import Tuple, Union

import numpy as np
import polars as pl

from tactics2d.participant.element import Cyclist, Pedestrian, Vehicle
from tactics2d.participant.guess_type import GuessType
from tactics2d.participant.trajectory import State, Trajectory


class InteractionParser:
    """This class implements a parser for INTERACTION dataset.

    !!! quote "Reference"
        Zhan, Wei, et al. "Interaction dataset: An international, adversarial and cooperative motion dataset in interactive driving scenarios with semantic maps." arXiv preprint arXiv:1910.03088 (2019).
    """

    _type_guesser = GuessType()

    def _get_file_id(self, file: Union[int, str]):
        if isinstance(file, str):
            file_id = int(re.findall(r"\d+", file)[0])
        elif isinstance(file, int):
            file_id = file
        else:
            raise TypeError("The input file must be an integer or a string.")

        return file_id

    def get_time_range(self, file: str, folder: str) -> Tuple[int, int]:
        """This function gets the time range of a single trajectory data file.

        Args:
            file (str): The name of the trajectory file or the id as a string. The parser will extract the integer id from the string and parse the trajectory data from the following files: `vehicle_tracks_%03d.csv % file_id`, `pedestrian_tracks_%03d.csv % file_id`.
            folder (str): The path to the folder containing the trajectory data.

        Returns:
            actual_time_range (Tuple[int, int]): The time range of the trajectory data. The first element is the start time. The second element is the end time. The unit of time stamp is millisecond (ms).
        """
        # Convert to string if integer
        if isinstance(file, int):
            file = str(file)
        file_id = self._get_file_id(file)
        vehicle_file_path = os.path.join(folder, "vehicle_tracks_%03d.csv" % file_id)

        df_vehicle = pl.read_csv(vehicle_file_path)
        start_frame = int(df_vehicle["timestamp_ms"].min())
        end_frame = int(df_vehicle["timestamp_ms"].max())

        pedestrian_file_path = os.path.join(folder, "pedestrian_tracks_%03d.csv" % file_id)
        if os.path.exists(pedestrian_file_path):
            df_pedestrian = pl.read_csv(pedestrian_file_path)
            ped_start = int(df_pedestrian["timestamp_ms"].min())
            ped_end = int(df_pedestrian["timestamp_ms"].max())
            start_frame = min(start_frame, ped_start)
            end_frame = max(end_frame, ped_end)

        actual_time_range = (start_frame, end_frame)
        return actual_time_range

    def parse_vehicle(
        self, file_path: str, time_range: Tuple[int, int] = None
    ) -> Tuple[dict, Tuple[int, int]]:
        """This function parses the vehicle trajectory file in INTERACTION dataset.

        Args:
            file_path: The path to the vehicle trajectory file.
            time_range (Tuple[int, int], optional): The time range of the trajectory data to parse. The unit of time stamp is millisecond (ms). If the time range is not given, the parser will parse the whole trajectory data.

        Returns:
            vehicles (dict): A dictionary of vehicles. The keys are the ids of the vehicles. The values are the vehicles.
            actual_stamp_range (Tuple[int, int]): The actual time range of the trajectory data. The first element is the start time. The second element is the end time. The unit of time stamp is millisecond (ms).
        """
        if time_range is None:
            time_range = (-np.inf, np.inf)

        vehicles = dict()
        trajectories = dict()

        # Read and filter data using polars
        df = pl.read_csv(file_path)
        # Filter by time range
        if time_range[0] != -np.inf or time_range[1] != np.inf:
            df = df.filter(
                (pl.col("timestamp_ms") >= time_range[0])
                & (pl.col("timestamp_ms") <= time_range[1])
            )

        # Calculate actual time range
        if len(df) > 0:
            actual_stamp_range = (int(df["timestamp_ms"].min()), int(df["timestamp_ms"].max()))
        else:
            actual_stamp_range = (np.inf, -np.inf)

        # Group by track_id and process each vehicle
        for track_id, group in df.group_by("track_id", maintain_order=False):
            # Convert track_id to integer (polars may return tuple for single column group by)
            vehicle_id = int(track_id[0]) if isinstance(track_id, tuple) else int(track_id)
            # Get vehicle metadata from first row
            vehicle = Vehicle(
                id_=vehicle_id,
                type_=group["agent_type"][0],
                length=group["length"][0],
                width=group["width"][0],
            )
            vehicles[vehicle_id] = vehicle

            # Create trajectory and add states
            trajectory = Trajectory(vehicle_id, fps=10)
            # Iterate over rows in the group
            for row in group.iter_rows(named=True):
                state = State(
                    frame=row["timestamp_ms"],
                    x=row["x"],
                    y=row["y"],
                    heading=row["psi_rad"],
                    vx=row["vx"],
                    vy=row["vy"],
                )
                trajectory.add_state(state)
            trajectories[vehicle_id] = trajectory

            # Bind trajectory to vehicle
            vehicle.bind_trajectory(trajectory)

        return vehicles, actual_stamp_range

    def parse_pedestrians(
        self, participants: dict, file_path: str, time_range: Tuple[int, int] = None
    ) -> Tuple[dict, Tuple[int, int]]:
        """This function parses the pedestrian trajectory file in INTERACTION dataset. Because the original dataset does not distinguish cyclist and pedestrian, this function calls a type guesser, which is built from other datasets, to guess the type of the participants.

        Args:
            participants (dict): A dictionary of participants.
            file_path (str): The path to the pedestrian trajectory file.
            time_range (Tuple[int, int], optional): The time range of the trajectory data to parse. The unit of time stamp is millisecond (ms). If the time range is not given, the parser will parse the whole trajectory data.

        Returns:
            participants (dict): A dictionary of participants. The keys are the ids of the participants. The values are the participants.
            actual_stamp_range (Tuple[int, int]): The actual time range of the trajectory data. The first element is the start time. The second element is the end time. The unit of time stamp is millisecond (ms).
        """
        if time_range is None:
            time_range = (-np.inf, np.inf)

        trajectories = {}
        pedestrian_ids = {}
        if participants:
            # Extract integer IDs from keys (may be tuples from polars group by)
            keys = []
            for k in participants.keys():
                if isinstance(k, tuple):
                    keys.append(k[0])
                else:
                    keys.append(k)
            id_cnt = max(keys) + 1
        else:
            id_cnt = 0

        # Read and filter data using polars
        df = pl.read_csv(file_path)
        # Filter by time range
        if time_range[0] != -np.inf or time_range[1] != np.inf:
            df = df.filter(
                (pl.col("timestamp_ms") >= time_range[0])
                & (pl.col("timestamp_ms") <= time_range[1])
            )

        # Calculate actual time range
        if len(df) > 0:
            actual_stamp_range = (int(df["timestamp_ms"].min()), int(df["timestamp_ms"].max()))
        else:
            actual_stamp_range = (np.inf, -np.inf)

        # Group by track_id and process each pedestrian/cyclist
        for track_id, group in df.group_by("track_id", maintain_order=False):
            # Unpack tuple if polars returned tuple for single column group by
            if isinstance(track_id, tuple):
                track_id = track_id[0]
            # Map original track_id to new consecutive id
            if track_id not in pedestrian_ids:
                pedestrian_ids[track_id] = id_cnt
                trajectories[id_cnt] = Trajectory(id_cnt, fps=10)
                id_cnt += 1

            mapped_id = pedestrian_ids[track_id]
            # Add all states for this track
            for row in group.iter_rows(named=True):
                state = State(
                    frame=row["timestamp_ms"], x=row["x"], y=row["y"], vx=row["vx"], vy=row["vy"]
                )
                trajectories[mapped_id].add_state(state)

        # Guess type for each trajectory and create participant
        for trajectory_id, trajectory in trajectories.items():
            type_ = self._type_guesser.guess_by_trajectory(trajectory)
            if type_ == "pedestrian":
                participants[trajectory_id] = Pedestrian(
                    trajectory_id, type_, trajectory=trajectory
                )
            elif type_ == "bicycle":
                participants[trajectory_id] = Cyclist(
                    trajectory_id, type_, trajectory=trajectory, length=2.0, width=0.7
                )

        return participants, actual_stamp_range

    def parse_trajectory(
        self, file: str, folder: str, time_range: Tuple[int, int] = None
    ) -> Tuple[dict, Tuple[int, int]]:
        """Parse the trajectory data of INTERACTION dataset. The states were collected at 10Hz.

        Args:
            file (str): The name of the trajectory file or the id as a string. The parser will extract the integer id from the string and parse the trajectory data from the following files: `vehicle_tracks_%03d.csv % file_id`, `pedestrian_tracks_%03d.csv % file_id`.
            folder (str): The path to the folder containing the trajectory data.
            time_range (Tuple[int, int], optional): The time range of the trajectory data to parse. The unit of time stamp is millisecond (ms). If the time range is not given, the parser will parse the whole trajectory data.

        Returns:
            participants (dict): A dictionary of participants. The keys are the ids of the participants. The values are the participants.
            actual_stamp_range (Tuple[int, int]): The actual time range of the trajectory data. The first element is the start time. The second element is the end time. The unit of time stamp is millisecond (ms).
        """
        # Convert to string if integer
        if isinstance(file, int):
            file = str(file)
        file_id = self._get_file_id(file)

        vehicle_file_path = os.path.join(folder, "vehicle_tracks_%03d.csv" % file_id)
        participants, actual_time_range = self.parse_vehicle(vehicle_file_path, time_range)

        pedestrian_file_path = os.path.join(folder, "pedestrian_tracks_%03d.csv" % file_id)
        if os.path.exists(pedestrian_file_path):
            participants, actual_time_range_ = self.parse_pedestrians(
                participants, pedestrian_file_path, time_range
            )
            actual_time_range = (
                min(actual_time_range[0], actual_time_range_[0]),
                max(actual_time_range[1], actual_time_range_[1]),
            )

        return participants, actual_time_range
