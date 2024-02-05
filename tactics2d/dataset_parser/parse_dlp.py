##! python3
# -*- coding: utf-8 -*-
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: parse_dlp.py
# @Description: This file implements a parser of the Dragon Lake Parking Dataset.
# @Author: Yueyuan Li
# @Version: 1.0.0

from typing import Tuple, Union
import json

import numpy as np

from tactics2d.participant.element import Vehicle, Pedestrian, Cyclist, Other
from tactics2d.trajectory.element import State, Trajectory


class DLPParser:
    """This class implements a parser of the Dragon Lake Parking Dataset.

    !!! info "Reference"
        Shen, Xu, et al. "Parkpredict: Motion and intent prediction of vehicles in parking lots." 2020 IEEE Intelligent Vehicles Symposium (IV). IEEE, 2020.
    """

    _TYPE_MAPPING = {
        "Car": "car",
        "Medium Vehicle": "car",
        "Bus": "bus",
        "Motorcycle": "motorcycle",
        "Bicycle": "bicycle",
        "Pedestrian": "pedestrian",
        "Undefined": "other",
    }

    _CLASS_MAPPING = {
        "Car": Vehicle,
        "Medium Vehicle": Vehicle,
        "Bus": Vehicle,
        "Motorcycle": Cyclist,
        "Bicycle": Cyclist,
        "Pedestrian": Pedestrian,
        "Undefined": Other,
    }

    def _generate_participant(self, instance, id_):
        type_ = self._TYPE_MAPPING[instance["type"]]
        class_ = self._CLASS_MAPPING[instance["type"]]
        participant = class_(
            id_=id_,
            type_=type_,
            length=instance["size"][0],
            width=instance["size"][1],
            trajectory=Trajectory(id_=id_, fps=25.0, stable_freq=False),
        )

        return participant

    def parse_trajectory(
        self, file: Union[int, str], folder: str, stamp_range: Tuple[float, float] = None
    ) -> Tuple[dict, Tuple[int, int]]:
        """This function parses trajectories from a series of DLP dataset files. The states were collected at 25Hz.

        Args:
            file (Union[int, str]): The id or the name of the trajectory file. The file is expected to be a json file (.json). If the input is an integer, the parser will parse the trajectory data from the following files: `DJI_%04d_agents.json % file`, `DJI_%04d_frames.json % file`, `DJI_%04d_instances.json % file`, `DJI_%04d_obstacles.json % file`. If the input is a string, the parser will extract the integer id first and repeat the above process.
            folder (str): The path to the folder containing the trajectory data.
            stamp_range (Tuple[float, float], optional): The time range of the trajectory data to parse. The unit of time stamp is millisecond. If the stamp range is not given, the parser will parse the whole trajectory data. Defaults to None.

        Returns:
            dict: A dictionary of vehicles. The keys are the ids of the vehicles. The values are the vehicles.
            Tuple[int, int]: The actual time range of the trajectory data. The first element is the start time. The second element is the end time. The unit of time stamp is millisecond.
        """
        if stamp_range is None:
            stamp_range = (-np.inf, np.inf)

        participants = {}
        actual_stamp_range = (np.inf, -np.inf)
        id_cnt = 0

        if isinstance(file, str):
            file_id = [int(s) for s in file.split("_") if s.isdigit()][0]
        elif isinstance(file, int):
            file_id = file
        else:
            raise TypeError("The input file must be an integer or a string.")

        with open("%s/DJI_%04d_agents.json" % (folder, file_id), "r") as f_agent:
            df_agent = json.load(f_agent)
        with open("%s/DJI_%04d_frames.json" % (folder, file_id), "r") as f_frame:
            df_frame = json.load(f_frame)
        with open("%s/DJI_%04d_instances.json" % (folder, file_id), "r") as f_instance:
            df_instance = json.load(f_instance)
        with open("%s/DJI_%04d_obstacles.json" % (folder, file_id), "r") as f_obstacle:
            df_obstacle = json.load(f_obstacle)

        for frame in df_frame.values():
            time_stamp = int(frame["timestamp"] * 1000)
            if time_stamp < stamp_range[0] or time_stamp > stamp_range[1]:
                continue
            actual_stamp_range = (
                min(actual_stamp_range[0], time_stamp),
                max(actual_stamp_range[1], time_stamp),
            )

            for obstacle in df_obstacle.values():
                state = State(
                    frame=time_stamp,
                    x=obstacle["coords"][0],
                    y=obstacle["coords"][1],
                    heading=obstacle["heading"],
                    vx=0,
                    vy=0,
                    ax=0,
                    ay=0,
                )

                if obstacle["obstacle_token"] not in participants:
                    participants[obstacle["obstacle_token"]] = self._generate_participant(
                        obstacle, id_cnt
                    )
                    id_cnt += 1

                participants[obstacle["obstacle_token"]].trajectory.append_state(state)

            for instance_token in frame["instances"]:
                instance = df_instance[instance_token]
                state = State(
                    frame=round(frame["timestamp"] * 1000),
                    x=instance["coords"][0],
                    y=instance["coords"][1],
                    heading=instance["heading"],
                    speed=instance["speed"],
                    ax=instance["acceleration"],
                    ay=instance["acceleration"],
                )

                if instance["agent_token"] not in participants:
                    participants[instance["agent_token"]] = self._generate_participant(
                        df_agent[instance["agent_token"]], id_cnt
                    )
                    id_cnt += 1

                participants[instance["agent_token"]].trajectory.append_state(state)

        return participants, actual_stamp_range
