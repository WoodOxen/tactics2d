import os
import json
import math

import pandas as pd

from tactics2d.participant.element.vehicle import Vehicle
from tactics2d.trajectory.element.state import State
from tactics2d.trajectory.element.trajectory import Trajectory


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
    "cycle": "cycle",
    "pedestrian": "pedestrian"
}


class LevelXParser(object):
    def __init__(self, dataset: str):
        if dataset not in REGISTERED_DATASET:
            raise KeyError(f"{dataset} is not an available LevelX-series dataset.")

        self.dataset = dataset

    def type_mapping(self, type_name: str):
        return TYPE_MAPPING[type_name]

    def _calibrate_location(self, x: float, y: float):
        return x, y

    def parse(self, file_id, folder_path, stamp_range):
        df_track_chunk = pd.read_csv(
            os.path.join(folder_path, "%02d_tracks.csv" % file_id), iterator=True, chunksize=10000)
        df_track_meta = pd.read_csv(
            os.path.join(folder_path, "%02d_tracksMeta.csv" % file_id))
        df_recording_meta = pd.read_csv(
            os.path.join(folder_path, "%02d_recordingMeta.csv" % file_id))

        # load the vehicles that have frame in the arbitrary range
        vehicles = dict()

        for _, vehicle_info in df_track_meta.iterrows():
            if vehicle_info["initialFrame"] < stamp_range[0] or vehicle_info["finalFrame"] > stamp_range[-1]:
                continue
            
            vehicle_id = vehicle_info["id"] if self.dataset == "highD" else vehicle_info["trackId"]
            if self.dataset == "highD":
                vehicle = Vehicle(
                    id_=vehicle_id, type_=LevelXParser.type_mapping(vehicle_info["class"]),
                    length=vehicle_info["width"], width=vehicle_info["height"]
                )
            else:
                vehicle = Vehicle(
                    id_=vehicle_id, type_=LevelXParser.type_mapping(vehicle_info["class"]),
                    length=vehicle_info["length"], width=vehicle_info["width"]
                )

            vehicles[vehicle_id] = vehicle

        # parse the corresponding trajectory to each vehicle and bind them
        trajectories = dict()

        for chunk in df_track_chunk:

            first_vehicle_id = chunk.iloc[0]["id"] if self.dataset == "highD" else chunk.iloc[0]["trackId"]
            last_vehicle_id = chunk.iloc[-1]["id"] if self.dataset == "highD" else chunk.iloc[-1]["trackId"]
            if first_vehicle_id not in vehicles and last_vehicle_id not in vehicles:
                continue

            for _, state_info in chunk.iterrows():
                if state_info["frame"] < stamp_range[0] or state_info["frame"] > stamp_range[-1]:
                    continue

                trajectory_id = state_info["id"] if self.dataset == "highD" else state_info["trackId"]
                if trajectory_id not in trajectories:
                    trajectories[trajectory_id] = Trajectory(trajectory_id)

                if self.dataset == "highD":
                    x, y = self._calibrate_location(state_info["x"], state_info["y"])
                    heading = round(math.atan2(state_info["xVelocity"], state_info["yVelocity"]), 5)
                    state = State(frame=state_info["frame"], x=x, y=y, heading=heading)
                else:
                    x, y = self._calibrate_location(state_info["xCenter"], state_info["yCenter"])
                    state = State(frame=state_info["frame"], x=x, y=y,heading=state_info["heading"])
                state.set_velocity(state_info["xVelocity"], state_info["yVelocity"])
                state.set_accel(state_info["xAcceleration"], state_info["yAcceleration"])

                trajectories[trajectory_id].append_state(state)

        for vehicle_id in vehicles.keys():
            vehicles[vehicle_id].bind_trajectory(trajectories[vehicle_id])

        return vehicles