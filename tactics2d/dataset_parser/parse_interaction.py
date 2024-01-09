from typing import Tuple
import os

import pandas as pd

from tactics2d.participant.element import Vehicle, Pedestrian, Cyclist
from tactics2d.participant.guess_type import GuessType
from tactics2d.trajectory.element import State, Trajectory


CLASS_MAPPING = {"cyclist": Cyclist, "pedestrian": Pedestrian}


class InteractionParser:
    """This class implements a parser for INTERACTION dataset.

    Zhan, Wei, et al. "Interaction dataset: An international, adversarial and cooperative motion dataset in interactive driving scenarios with semantic maps." arXiv preprint arXiv:1910.03088 (2019).
    """

    def parse_vehicle(
        self, file_id: int, folder_path: str, stamp_range: Tuple[float, float] = None
    ):
        df_vehicle = pd.read_csv(os.path.join(folder_path, "vehicle_tracks_%03d.csv" % file_id))

        vehicles = dict()
        trajectories = dict()

        if stamp_range is None:
            stamp_range = (-float("inf"), float("inf"))

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
                trajectories[vehicle_id] = Trajectory(vehicle_id, fps=10)

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

    def parse_pedestrians(
        self,
        participants: dict,
        file_id: int,
        folder_path: str,
        stamp_range: Tuple[float, float] = None,
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

        if stamp_range is None:
            stamp_range = (-float("inf"), float("inf"))

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
            class_ = CLASS_MAPPING[type_]
            participants[trajectory_id] = class_(trajectory_id, type_, trajectory=trajectory)

        return participants

    def parse_trajectory(
        self, file_id: int, folder_path: str, stamp_range: Tuple[float, float] = None
    ):
        """Parse the trajectory data of INTERACTION dataset. The states were collected at 10Hz.

        Args:
            file_id (int): The id of the trajectory file. With the given file id, the parser will parse the trajectory data from the following files: vehicle_tracks_{file_id}.csv, pedestrian_tracks_{file_id}.csv. When the pedestrian trajectory file is not available, the parser will ignore the pedestrian data.
            folder_path (str): The path to the folder containing the trajectory data.
            stamp_range (Tuple[float, float], optional): The time range of the trajectory data to parse. If the stamp range is not given, the parser will parse the whole trajectory data. Defaults to None.

        Returns:
            dict: A dictionary of participants. The keys are the ids of the participants. The values are the participants.
        """
        participants = self.parse_vehicle(file_id, folder_path, stamp_range)
        participants = self.parse_pedestrians(participants, file_id, folder_path, stamp_range)
        return participants
