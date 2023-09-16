from typing import Tuple
import os

import pandas as pd

from tactics2d.participant.element import Vehicle, Pedestrian, Cyclist
from tactics2d.participant.guess_type import GuessType
from tactics2d.trajectory.element import State, Trajectory


TYPE_MAPPING = {"cyclist": Cyclist, "pedestrian": Pedestrian}


class InteractionParser:
    """This class provides pure static methods to parse trajectory data from the
    INTERACTION dataset.
    """

    @staticmethod
    def parse_vehicle(
        file_id: int,
        folder_path: str,
        stamp_range: Tuple[float, float] = (-float("inf"), float("inf")),
    ):
        df_vehicle = pd.read_csv(os.path.join(folder_path, "vehicle_tracks_%03d.csv" % file_id))

        vehicles = {}
        trajectories = {}

        for _, state_info in df_vehicle.iterrows():
            if state_info["frame_id"] < stamp_range[0] or state_info["frame_id"] > stamp_range[1]:
                continue

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
                trajectories[vehicle_id] = Trajectory(vehicle_id)

            state = State(
                frame=state_info["timestamp_ms"],
                x=state_info["x"],
                y=state_info["y"],
                heading=state_info["psi_rad"],
                vx=state_info["vx"],
                vy=state_info["vy"],
            )
            trajectories[vehicle_id].append_state(state)

        for vehicle_id, vehicle in vehicles.items():
            vehicles[vehicle_id].bind_trajectory(trajectories[vehicle_id])

        return vehicles

    @staticmethod
    def parse_pedestrians(
        participants: dict,
        file_id: int,
        folder_path: str,
        stamp_range: Tuple[float, float] = (-float("inf"), float("inf")),
    ):
        type_guesser = GuessType()

        pedestrian_path = os.path.join(folder_path, "pedestrian_tracks_%03d.csv" % file_id)
        if os.path.exists(pedestrian_path):
            df_pedestrian = pd.read_csv(pedestrian_path)
        else:
            return {}

        trajectories = {}
        pedestrian_ids = {}
        id_cnt = max(list(participants.keys())) + 1

        for _, state_info in df_pedestrian.iterrows():
            time_stamp = float(state_info["frame_id"]) / 100.0
            if time_stamp < stamp_range[0] or time_stamp > stamp_range[1]:
                continue

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
            type_ = type_guesser.guess_by_trajectory(trajectory)
            class_ = TYPE_MAPPING[type_]
            participants[trajectory_id] = class_(trajectory_id, type_, trajectory=trajectory)

        return participants

    @staticmethod
    def parse(
        file_id: int,
        folder_path: str,
        stamp_range: Tuple[float, float] = (-float("inf"), float("inf")),
    ):
        participants = InteractionParser.parse_vehicle(file_id, folder_path, stamp_range)
        participants = InteractionParser.parse_pedestrians(
            participants, file_id, folder_path, stamp_range
        )
        return participants
