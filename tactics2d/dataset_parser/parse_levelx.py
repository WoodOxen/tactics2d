from typing import Tuple
import os
import math

import pandas as pd

from tactics2d.participant.element import Vehicle, Pedestrian, Cyclist
from tactics2d.trajectory.element import State, Trajectory


class LevelXParser:
    """This class implements a parser for the series of datasets collected by the Institute for Automotive Engineering (ika) of RWTH Aachen University. Because the commercial version of the datasets are held by LevelXData, we call this series of datasets LevelX-series datasets. The datasets include: highD, inD, rounD, exiD, uniD.

    Krajewski, Robert, et al. "The highd dataset: A drone dataset of naturalistic vehicle trajectories on german highways for validation of highly automated driving systems." 2018 21st international conference on intelligent transportation systems (ITSC). IEEE, 2018.

    Bock, Julian, et al. "The ind dataset: A drone dataset of naturalistic road user trajectories at german intersections." 2020 IEEE Intelligent Vehicles Symposium (IV). IEEE, 2020.

    Krajewski, Robert, et al. "The round dataset: A drone dataset of road user trajectories at roundabouts in germany." 2020 IEEE 23rd International Conference on Intelligent Transportation Systems (ITSC). IEEE, 2020.

    Moers, Tobias, et al. "The exiD dataset: A real-world trajectory dataset of highly interactive highway scenarios in Germany." 2022 IEEE Intelligent Vehicles Symposium (IV). IEEE, 2022.

    Bock, Julian, et al. "Highly accurate scenario and reference data for automated driving." ATZ worldwide 123.5 (2021): 50-55.
    """

    REGISTERED_DATASET = ["highD", "inD", "rounD", "exiD", "uniD"]

    TYPE_MAPPING = {
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

    CLASS_MAPPING = {
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

    def __init__(self, dataset: str = ""):
        """Initialize the parser.

        Args:
            dataset (str, optional): The dataset you want to parse. The available choices are: highD, inD, rounD, exiD, uniD. Defaults to "".
        """
        if dataset not in self.REGISTERED_DATASET:
            raise KeyError(
                f"{dataset} is not an available LevelX-series dataset. The available datasets are {self.REGISTERED_DATASET}."
            )

        self.dataset = dataset

    def _calibrate_location(self, x: float, y: float):
        return x, y

    def parse_trajectory(
        self, file_id: int, folder_path: str, stamp_range: Tuple[float, float] = None
    ):
        """

        Args:
            file_id (int): The id of the trajectory file. With the given file id, the parser will parse the trajectory data from the following files: {file_id}_tracks.csv, {file_id}_tracksMeta.csv.
            folder_path (str): The path to the folder containing the trajectory data.
            stamp_range (Tuple[float, float], optional): The time range of the trajectory data to parse. If the stamp range is not given, the parser will parse the whole trajectory data. Defaults to None.

        Returns:
            dict: A dictionary of participants. The keys are the ids of the participants. The values are the participants.
        """
        df_track_chunk = pd.read_csv(
            os.path.join(folder_path, "%02d_tracks.csv" % file_id), iterator=True, chunksize=10000
        )
        df_track_meta = pd.read_csv(
            os.path.join(folder_path, "%02d_tracksMeta.csv" % file_id), engine="pyarrow"
        )

        # load the vehicles that have frame in the arbitrary range
        participants = dict()

        id_key = "id" if self.dataset == "highD" else "trackId"

        if stamp_range is None:
            time_stamp = (-float("inf"), float("inf"))

        for _, participant_info in df_track_meta.iterrows():
            first_stamp = participant_info["initialFrame"] / 25.0
            last_stamp = participant_info["finalFrame"] / 25.0

            if last_stamp < stamp_range[0] or first_stamp > stamp_range[1]:
                continue

            id_ = participant_info[id_key]
            class_ = self.CLASS_MAPPING[participant_info["class"]]
            type_ = self.TYPE_MAPPING[participant_info["class"]]

            if self.dataset == "highD":
                participant = class_(
                    id_=id_,
                    type_=type_,
                    length=participant_info["width"],
                    width=participant_info["height"],
                )
            else:
                participant = class_(
                    id_=id_,
                    type_=type_,
                    length=participant_info["length"],
                    width=participant_info["width"],
                )

            participants[id_] = participant

        participant_ids = set(participants.keys())

        # parse the corresponding trajectory to each participant and bind them
        trajectories = dict()

        for chunk in df_track_chunk:
            chunk_ids = set(pd.unique(chunk[id_key]))
            if len(chunk_ids.union(participant_ids)) == 0:
                continue

            for _, state_info in chunk.iterrows():
                time_stamp = state_info["frame"] / 25.0
                frame = round(time_stamp * 1000)

                if time_stamp < stamp_range[0] or time_stamp > stamp_range[1]:
                    continue

                trajectory_id = int(state_info[id_key])
                if trajectory_id not in trajectories:
                    trajectories[trajectory_id] = Trajectory(id_=trajectory_id, fps=25.0)

                if self.dataset == "highD":
                    x, y = self._calibrate_location(state_info["x"], state_info["y"])
                    heading = round(math.atan2(state_info["xVelocity"], state_info["yVelocity"]), 5)
                    state = State(frame, x=x, y=y, heading=heading)
                else:
                    x, y = self._calibrate_location(state_info["xCenter"], state_info["yCenter"])
                    state = State(
                        frame, x=x, y=y, heading=state_info["heading"] * 2 * math.pi / 360
                    )
                state.set_velocity(state_info["xVelocity"], state_info["yVelocity"])
                state.set_accel(state_info["xAcceleration"], state_info["yAcceleration"])

                trajectories[trajectory_id].append_state(state)

        for participant_id in participants.keys():
            participants[participant_id].bind_trajectory(trajectories[participant_id])

        return participants