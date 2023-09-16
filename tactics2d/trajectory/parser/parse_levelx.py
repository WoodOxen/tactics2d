from typing import Tuple
import os
import math

import pandas as pd

from tactics2d.participant.element import Vehicle, Pedestrian, Cyclist
from tactics2d.trajectory.element import State, Trajectory


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


class LevelXParser:
    def __init__(self, dataset: str = ""):
        if dataset not in REGISTERED_DATASET:
            raise KeyError(
                f"{dataset} is not an available LevelX-series dataset. The available datasets are {REGISTERED_DATASET}."
            )

        self.dataset = dataset

    def _calibrate_location(self, x: float, y: float):
        return x, y

    def parse(
        self,
        file_id: int,
        folder_path: str,
        stamp_range: Tuple[float, float] = (-float("inf"), float("inf")),
    ):
        df_track_chunk = pd.read_csv(
            os.path.join(folder_path, "%02d_tracks.csv" % file_id),
            iterator=True,
            chunksize=10000,
        )
        df_track_meta = pd.read_csv(os.path.join(folder_path, "%02d_tracksMeta.csv" % file_id))
        df_recording_meta = pd.read_csv(
            os.path.join(folder_path, "%02d_recordingMeta.csv" % file_id)
        )

        # load the vehicles that have frame in the arbitrary range
        participants = dict()

        id_key = "id" if self.dataset == "highD" else "trackId"

        for _, participant_info in df_track_meta.iterrows():
            first_stamp = participant_info["initialFrame"] / 25.0
            last_stamp = participant_info["finalFrame"] / 25.0

            if last_stamp < stamp_range[0] or first_stamp > stamp_range[1]:
                continue

            id_ = participant_info[id_key]
            class_ = CLASS_MAPPING[participant_info["class"]]
            type_ = TYPE_MAPPING[participant_info["class"]]

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
