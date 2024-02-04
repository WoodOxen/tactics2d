##! python3
# -*- coding: utf-8 -*-
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: parse_interaction.py
# @Description: This file implements a parser for INTERACTION dataset.
# @Author: Yueyuan Li
# @Version: 1.0.0

import os
from typing import Tuple, Union
import re

import pandas as pd
import numpy as np

from tactics2d.participant.element import Vehicle, Pedestrian, Cyclist
from tactics2d.participant.guess_type import GuessType
from tactics2d.trajectory.element import State, Trajectory


class InteractionParser:
    """This class implements a parser for INTERACTION dataset.

    !!! info "Reference"
        Zhan, Wei, et al. "Interaction dataset: An international, adversarial and cooperative motion dataset in interactive driving scenarios with semantic maps." arXiv preprint arXiv:1910.03088 (2019).
    """

    type_guesser = GuessType()

    def _get_file_id(self, file: Union[int, str]):
        if isinstance(file, str):
            file_id = int(re.findall(r"\d+", file)[0])
        elif isinstance(file, int):
            file_id = file
        else:
            raise TypeError("The input file must be an integer or a string.")

        return file_id

    def get_time_range(self, file: Union[int, str], folder: str) -> Tuple[float, float]:
        """This function gets the time range of a single trajectory data file.

        Args:
            file (Union[int, str]): The id or the name of the trajectory file. If the input is an integer, the parser will parse the trajectory data from the following files:  `vehicle_tracks_%03d.csv % file`, `pedestrian_tracks_%03d.csv % file`. If the input is a string, the parser will extract the integer id first and repeat the above process.
            folder (str): The path to the folder containing the trajectory data.

        Returns:
            Tuple[int, int]: The time range of the trajectory data. The first element is the start time. The second element is the end time. The unit of time stamp is millisecond.
        """
        file_id = self._get_file_id(file)
        vehicle_file_path = os.path.join(folder, "vehicle_tracks_%03d.csv" % file_id)
        df_vehicle = pd.read_csv(vehicle_file_path)
        start_frame = int(min(df_vehicle["timestamp_ms"]))
        end_frame = int(max(df_vehicle["timestamp_ms"]))

        pedestrian_file_path = os.path.join(folder, "pedestrian_tracks_%03d.csv" % file_id)
        if os.path.exists(pedestrian_file_path):
            df_pedestrian = pd.read_csv(pedestrian_file_path)
            start_frame = min(start_frame, int(min(df_pedestrian["timestamp_ms"])))
            end_frame = max(end_frame, int(max(df_pedestrian["timestamp_ms"])))

        return start_frame, end_frame

    def parse_vehicle(
        self, file_path: str, time_range: Tuple[float, float] = None
    ) -> Tuple[dict, Tuple[int, int]]:
        """This function parses the vehicle trajectory file in INTERACTION dataset.

        Args:
            file_path: The path to the vehicle trajectory file.
            time_range (Tuple[float, float], optional): The time range of the trajectory data to parse. The unit of time stamp is millisecond. If the stamp range is not given, the parser will parse the whole trajectory data. Defaults to None.

        Returns:
            dict: A dictionary of vehicles. The keys are the ids of the vehicles. The values are the vehicles.
            Tuple[int, int]: The actual time range of the trajectory data. The first element is the start time. The second element is the end time. The unit of time stamp is millisecond.
        """
        if time_range is None:
            time_range = (-np.inf, np.inf)

        vehicles = dict()
        trajectories = dict()

        df_vehicle = pd.read_csv(file_path)
        actual_stamp_range = (np.inf, -np.inf)

        for _, state_info in df_vehicle.iterrows():
            time_stamp = state_info["timestamp_ms"]
            if time_stamp < time_range[0] or time_stamp > time_range[1]:
                continue

            actual_stamp_range = (
                min(actual_stamp_range[0], time_stamp),
                max(actual_stamp_range[1], time_stamp),
            )

            vehicle_id = state_info["track_id"]
            if vehicle_id not in vehicles:
                vehicle = Vehicle(
                    id_=vehicle_id,
                    type_=state_info["agent_type"],
                    length=state_info["length"],
                    width=state_info["width"],
                )
                vehicles[vehicle_id] = vehicle

            if vehicle_id not in trajectories:
                trajectories[vehicle_id] = Trajectory(vehicle_id, fps=10)

            state = State(
                frame=time_stamp,
                x=state_info["x"],
                y=state_info["y"],
                heading=state_info["psi_rad"],
                vx=state_info["vx"],
                vy=state_info["vy"],
            )
            trajectories[vehicle_id].append_state(state)

        for vehicle_id, vehicle in vehicles.items():
            vehicles[vehicle_id].bind_trajectory(trajectories[vehicle_id])

        return vehicles, actual_stamp_range

    def parse_pedestrians(
        self, participants: dict, file_path: str, time_range: Tuple[float, float] = None
    ) -> Tuple[dict, Tuple[int, int]]:
        """This function parses the pedestrian trajectory file in INTERACTION dataset. Because the original dataset does not distinguish cyclist and pedestrian, this function calls a type guesser, which is built from other datasets, to guess the type of the participants.

        Args:
            participants (dict): A dictionary of participants.
            file_path (str): The path to the pedestrian trajectory file.
            time_range (Tuple[float, float], optional): The time range of the trajectory data to parse. The unit of time stamp is millisecond. If the stamp range is not given, the parser will parse the whole trajectory data. Defaults to None.

        Returns:
            dict: A dictionary of participants. The keys are the ids of the participants. The values are the participants.
            Tuple[int, int]: The actual time range of the trajectory data. The first element is the start time. The second element is the end time. The unit of time stamp is millisecond.
        """
        if time_range is None:
            time_range = (-np.inf, np.inf)

        trajectories = {}
        pedestrian_ids = {}
        id_cnt = max(list(participants.keys())) + 1

        df_pedestrian = pd.read_csv(file_path)
        actual_stamp_range = (np.inf, -np.inf)

        for _, state_info in df_pedestrian.iterrows():
            time_stamp = state_info["timestamp_ms"]
            if time_stamp < time_range[0] or time_stamp > time_range[1]:
                continue

            actual_stamp_range = (
                min(actual_stamp_range[0], time_stamp),
                max(actual_stamp_range[1], time_stamp),
            )

            if state_info["track_id"] not in pedestrian_ids:
                pedestrian_ids[state_info["track_id"]] = id_cnt
                trajectories[id_cnt] = Trajectory(id_cnt, fps=10)
                id_cnt += 1

            state = State(
                frame=state_info["timestamp_ms"],
                x=state_info["x"],
                y=state_info["y"],
                vx=state_info["vx"],
                vy=state_info["vy"],
            )
            trajectories[pedestrian_ids[state_info["track_id"]]].append_state(state)

        for trajectory_id, trajectory in trajectories.items():
            type_ = self.type_guesser.guess_by_trajectory(trajectory)
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
        self, file: Union[int, str], folder: str, time_range: Tuple[float, float] = None
    ) -> Tuple[dict, Tuple[int, int]]:
        """Parse the trajectory data of INTERACTION dataset. The states were collected at 10Hz.

        Args:
            file (Union[int, str]): The id or the name of the trajectory file. If the input is an integer, the parser will parse the trajectory data from the following files: `vehicle_tracks_%03d.csv % file`, `pedestrian_tracks_%03d.csv % file`. If the input is a string, the parser will extract the integer id first and repeat the above process.
            folder (str): The path to the folder containing the trajectory data.
            time_range (Tuple[float, float], optional): The time range of the trajectory data to parse. The unit of time stamp is millisecond. If the stamp range is not given, the parser will parse the whole trajectory data. Defaults to None.

        Returns:
            dict: A dictionary of participants. The keys are the ids of the participants. The values are the participants.
            Tuple[int, int]: The actual time range of the trajectory data. The first element is the start time. The second element is the end time. The unit of time stamp is millisecond.
        """
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
