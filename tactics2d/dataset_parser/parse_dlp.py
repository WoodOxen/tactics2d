# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Dragon Lake Parking dataset parser implementation."""


import os
from typing import Tuple, Union

import numpy as np
import orjson
import pandas as pd
import polars as pl

from tactics2d.participant.element import Cyclist, Obstacle, Other, Pedestrian, Vehicle
from tactics2d.participant.trajectory import State, Trajectory


class DLPParser:
    """This class implements a parser of the Dragon Lake Parking Dataset.

    !!! quote "Reference"
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

    def _generate_participant(self, row, id_):
        type_ = self._TYPE_MAPPING[row["type"]]
        class_ = self._CLASS_MAPPING[row["type"]]
        participant = class_(
            id_=id_,
            type_=type_,
            length=row["size_0"],
            width=row["size_1"],
            trajectory=Trajectory(id_=id_, fps=25.0, stable_freq=False),
        )

        return participant

    def _generate_obstacle(self, row, id_, frame):
        type_ = self._TYPE_MAPPING[row["type"]]
        state = State(
            frame=frame,
            x=row["coords_0"],
            y=row["coords_1"],
            heading=row["heading"],
            vx=0,
            vy=0,
            ax=0,
            ay=0,
        )
        trajectory = Trajectory(id_=id_, fps=25.0, stable_freq=False)
        trajectory.add_state(state)

        participant = Obstacle(
            id_=id_, type_=type_, length=row["size_0"], width=row["size_1"], trajectory=trajectory
        )

        return participant

    @staticmethod
    def _flatten_list_columns(df, list_columns) -> pd.DataFrame:
        for col in list_columns:
            expanded = pd.DataFrame(df[col].tolist(), index=df.index)
            expanded.columns = [f"{col}_{i}" for i in expanded.columns]
            df = df.drop(columns=[col]).join(expanded)
        return df

    def parse_trajectory(
        self, file: Union[int, str], folder: str, stamp_range: Tuple[float, float] = None
    ) -> Tuple[dict, Tuple[int, int]]:
        """This function parses trajectories from a series of DLP dataset files. The states were collected at 25Hz.

        Args:
            file (Union[int, str]): The id or the name of the trajectory file. The file is expected to be a json file (.json). If the input is an integer, the parser will parse the trajectory data from the following files: `DJI_%04d_agents.json % file`, `DJI_%04d_frames.json % file`, `DJI_%04d_instances.json % file`, `DJI_%04d_obstacles.json % file`. If the input is a string, the parser will extract the integer id first and repeat the above process.
            folder (str): The path to the folder containing the trajectory data.
            stamp_range (Tuple[float, float], optional): The time range of the trajectory data to parse. The unit of time stamp is millisecond (ms). If the stamp range is not given, the parser will parse the whole trajectory data.

        Returns:
            participants (dict): A dictionary of vehicles. The keys are the ids of the vehicles. The values are the vehicles.
            actual_stamp_range (Tuple[int, int]): The actual time range of the trajectory data. The first element is the start time. The second element is the end time. The unit of time stamp is millisecond (ms).
        """
        if stamp_range is None:
            stamp_range = (-np.inf, np.inf)

        participants = {}
        id_cnt = 0

        if isinstance(file, str):
            file_id = [int(s) for s in file.split("_") if s.isdigit()][0]
        elif isinstance(file, int):
            file_id = file
        else:
            raise TypeError("The input file must be an integer or a string.")

        with open(os.path.join(folder, "DJI_%04d_frames.json" % file_id), "rb") as f_frame:
            json_frame = orjson.loads(f_frame.read())
        with open(os.path.join(folder, "DJI_%04d_agents.json" % file_id), "rb") as f_agent:
            json_agent = orjson.loads(f_agent.read())
        with open(os.path.join(folder, "DJI_%04d_instances.json" % file_id), "rb") as f_instance:
            json_instance = orjson.loads(f_instance.read())
        with open(os.path.join(folder, "DJI_%04d_obstacles.json" % file_id), "rb") as f_obstacle:
            json_obstacle = orjson.loads(f_obstacle.read())

        df_frame = pd.json_normalize(json_frame.values(), sep=".")
        df_frame = df_frame.explode("instances")
        df_frame = pl.from_pandas(df_frame).rename({"instances": "instance_token"})

        df_agent = pd.json_normalize(json_agent.values(), sep=".")
        df_agent = self._flatten_list_columns(df_agent, ["size"])
        df_agent = pl.from_pandas(df_agent)

        df_instance = pd.json_normalize(json_instance.values(), sep=".")
        df_instance = self._flatten_list_columns(df_instance, ["coords", "acceleration"])
        df_instance = pl.from_pandas(df_instance)

        df_obstacle = pd.json_normalize(json_obstacle.values(), sep=".")
        df_obstacle = self._flatten_list_columns(df_obstacle, ["coords", "size"])
        df_obstacle = pl.from_pandas(df_obstacle)

        df_frame = df_frame.with_columns(
            (pl.col("timestamp") * 1000).cast(pl.Int64).alias("timestamp (ms)")
        )
        df_frame = df_frame.filter(
            pl.col("timestamp (ms)").is_between(stamp_range[0], stamp_range[1], closed="both")
        )

        actual_stamp_range = (df_frame["timestamp (ms)"].min(), df_frame["timestamp (ms)"].max())

        df_instance = df_instance.filter(pl.col("instance_token").is_in(df_frame["instance_token"]))
        df_instance = df_instance.join(df_frame, on="instance_token", how="inner")
        df_agent = df_agent.filter(pl.col("agent_token").is_in(df_instance["agent_token"]))

        participants = {
            row["agent_token"]: self._generate_participant(row, idx)
            for idx, row in enumerate(df_agent.iter_rows(named=True))
        }

        for row in df_instance.iter_rows(named=True):
            state = State(
                frame=int(row["timestamp (ms)"]),
                x=row["coords_0"],
                y=row["coords_1"],
                heading=row["heading"],
                speed=row["speed"],
                ax=row["acceleration_0"],
                ay=row["acceleration_1"],
            )

            participants[row["agent_token"]].trajectory.add_state(state)

        id_cnt = len(participants)
        obstacles = {
            row["obstacle_token"]: self._generate_obstacle(row, id_cnt + idx, actual_stamp_range[0])
            for idx, row in enumerate(df_obstacle.iter_rows(named=True))
        }

        participants.update(obstacles)

        return participants, actual_stamp_range
