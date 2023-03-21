import sys

sys.path.append(".")
sys.path.append("..")
import logging
logging.basicConfig(level=logging.DEBUG)
from typing import Tuple
import os
import math
import pandas as pd
import pickle

from tactics2d.participant.element import Vehicle, Pedestrian, Cyclist
from tactics2d.trajectory.element import State, Trajectory

from concurrent import futures


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


class LevelXPedestrainAndCycleParser(object):
    """
    This class implements a parser of the LevelX Dataset.
    
    It will sift through the dataset for pedestrians data and cyclists data.
    """
    def __init__(self, dataset: str = ""):
        if dataset not in REGISTERED_DATASET:
            raise KeyError(
                f"{dataset} is not an available LevelX-series dataset. The available datasets are {REGISTERED_DATASET}."
            )

        self.dataset = dataset

    def _calibrate_location(self, x: float, y: float):

        return x, y

    def parse(
        self, file_id: int, folder_path: str,
        stamp_range: Tuple[float, float] = (-float("inf"), float("inf")),
    ):
        df_track_chunk = pd.read_csv(
            os.path.join(folder_path, "%02d_tracks.csv" % file_id),
            iterator=True, chunksize=10000,
        )
        df_track_meta = pd.read_csv(
            os.path.join(folder_path, "%02d_tracksMeta.csv" % file_id)
        )
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

            if participant_info["class"] != "motorcycle" and participant_info["class"] != "bicycle" and \
            participant_info["class"] != "cycle" and participant_info["class"] != "pedestrian":
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
                if(state_info[id_key] not in participant_ids):
                    continue

                time_stamp = state_info["frame"] / 25.0
                frame = round(time_stamp * 1000)

                if time_stamp < stamp_range[0] or time_stamp > stamp_range[1]:
                    continue

                trajectory_id = int(state_info[id_key])
                if trajectory_id not in trajectories:
                    trajectories[trajectory_id] = Trajectory(
                        id_=trajectory_id, fps=25.0
                    )

                if self.dataset == "highD":
                    x, y = self._calibrate_location(state_info["x"], state_info["y"])
                    heading = round(
                        math.atan2(state_info["xVelocity"], state_info["yVelocity"]), 5
                    )
                    state = State(frame, x=x, y=y, heading=heading)
                else:
                    x, y = self._calibrate_location(
                        state_info["xCenter"], state_info["yCenter"]
                    )
                    state = State(frame, x=x, y=y, heading=state_info["heading"])
                state.set_velocity(state_info["xVelocity"], state_info["yVelocity"])
                state.set_accel(
                    state_info["xAcceleration"], state_info["yAcceleration"]
                )

                trajectories[trajectory_id].append_state(state)

        #trajectiories_cyclist_and_pedestrian = dict()
        #id = 0
        for participant_id in participants.keys():
            participants[participant_id].bind_trajectory(trajectories[participant_id])
        
        return participants

# D:/study/Tactics/TacticTest/tactics2d
"""
"dataset, file_id_start, file_id_end",
    ("inD", 0, 32),
    ("rounD", 0, 23),
    ("uniD", 0, 12)
"""

def levelX_parser(file_path:str, dataset: str, file_id: int, stamp_range: tuple):
    """
    This function parse one levelX file and sift through the dataset for pedestrians data and cyclists data

    """
    trajectory_parser = LevelXPedestrainAndCycleParser(dataset)
    participants = trajectory_parser.parse(file_id, file_path, stamp_range)
    return participants

def levelX_process(file_address:str, dataset: str, file_id_start: int, file_id_end: int):
    """
    This function use futures to parse all files in a levelX dataset.

    """
    stamp_range = (-float("inf"), float("inf"))
    file_path = file_address  + f"/levelx/{dataset}/data/"
    workers = 5

    with futures.ThreadPoolExecutor(max_workers=workers) as t:
        tasks = {}
        result = {}
        for file_id in range(file_id_start,file_id_end+1):
            print(dataset,file_id," start ")
            tasks[file_id] = t.submit(levelX_parser, file_path, dataset, file_id, stamp_range )
        
        trajectiories_cyclist_and_pedestrian = dict()
        id = 0
        for file_id in range(file_id_start,file_id_end+1):
            result[file_id] = tasks[file_id].result()
            for participant_id in result[file_id].keys():
                trajectiories_cyclist_and_pedestrian[id] = result[file_id][participant_id]
                id += 1
            print(dataset,file_id," length: ",len(result[file_id]))
        t.shutdown()   
    return trajectiories_cyclist_and_pedestrian

def levelX_saveTrajectory():
    """
    This function save the data of pedestrians and cyclists as .pkl file

    """
    file_address = "D:/study/Tactics/TacticTest/tactics2d"
    data_inD = levelX_process(file_address,"inD", 0, 20)
    data_inD = levelX_process(file_address,"inD", 21, 32)
    data_rounD = levelX_process(file_address,"rounD", 0, 23)
    data_uniD = levelX_process(file_address,"uniD", 0, 6)
    data_uniD = levelX_process(file_address,"uniD", 7, 12)


    with open("./pedestrian_and_cyclist_trajectory/inD0-20_data_pedestrian_and_cycle.pkl", "wb") as tf:
        pickle.dump(data_inD,tf)
    with open("./pedestrian_and_cyclist_trajectory/inD21-32_data_pedestrian_and_cycle.pkl", "wb") as tf:
        pickle.dump(data_inD,tf)
    with open("./pedestrian_and_cyclist_trajectory/rounD0-23_data_pedestrian_and_cycle.pkl", "wb") as tf:
        pickle.dump(data_rounD,tf)
    with open("./pedestrian_and_cyclist_trajectory/uniD0-6_data_pedestrian_and_cycle.pkl", "wb") as tf:
        pickle.dump(data_uniD,tf)
    with open("./pedestrian_and_cyclist_trajectory/uniD7-12_data_pedestrian_and_cycle.pkl", "wb") as tf:
        pickle.dump(data_uniD,tf)

if __name__ == '__main__':
    levelX_saveTrajectory()