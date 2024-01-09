from typing import Tuple
import json

from tactics2d.participant.element import Vehicle, Pedestrian, Cyclist, Other
from tactics2d.trajectory.element import State, Trajectory


class DLPParser:
    """This class implements a parser of the Dragon Lake Parking Dataset.

    Shen, Xu, et al. "Parkpredict: Motion and intent prediction of vehicles in parking lots." 2020 IEEE Intelligent Vehicles Symposium (IV). IEEE, 2020.
    """

    TYPE_MAPPING = {
        "Car": "car",
        "Medium Vehicle": "car",
        "Bus": "bus",
        "Motorcycle": "motorcycle",
        "Bicycle": "bicycle",
        "Pedestrian": "pedestrian",
        "Undefined": "other",
    }

    CLASS_MAPPING = {
        "Car": Vehicle,
        "Medium Vehicle": Vehicle,
        "Bus": Vehicle,
        "Motorcycle": Cyclist,
        "Bicycle": Cyclist,
        "Pedestrian": Pedestrian,
        "Undefined": Other,
    }

    def _generate_participant(self, instance, id_):
        type_ = self.TYPE_MAPPING[instance["type"]]
        class_ = self.CLASS_MAPPING[instance["type"]]
        participant = class_(
            id_=id_,
            type_=type_,
            length=instance["size"][0],
            width=instance["size"][1],
            trajectory=Trajectory(id_=id_, fps=25.0),
        )

        return participant

    def parse_trajectory(
        self, file_id: int, folder_path: str, stamp_range: Tuple[float, float] = None
    ):
        """Parse the trajectory data of DLP dataset. The states were collected at 25Hz.

        Args:
            file_id (int): The id of the scenario to parse. With the given file id, the parser will parse the trajectory data from the following files: DJI_{file_id}_agents.json, DJI_{file_id}_frames.json, DJI_{file_id}_instances.json, DJI_{file_id}_obstacles.json.
            folder_path (str): The path to the folder containing the trajectory data.
            stamp_range (Tuple[float, float], optional): The time range of the trajectory data to parse. If the stamp range is not given, the parser will parse the whole trajectory data. Defaults to None.
        """

        with open("%s/DJI_%04d_agents.json" % (folder_path, file_id), "r") as f_agent:
            df_agent = json.load(f_agent)
        with open("%s/DJI_%04d_frames.json" % (folder_path, file_id), "r") as f_frame:
            df_frame = json.load(f_frame)
        with open("%s/DJI_%04d_instances.json" % (folder_path, file_id), "r") as f_instance:
            df_instance = json.load(f_instance)
        with open("%s/DJI_%04d_obstacles.json" % (folder_path, file_id), "r") as f_obstacle:
            df_obstacle = json.load(f_obstacle)

        participants = {}
        id_cnt = 0

        if stamp_range is None:
            stamp_range = (-float("inf"), float("inf"))

        for frame in df_frame.values():
            if frame["timestamp"] < stamp_range[0] or frame["timestamp"] > stamp_range[1]:
                continue

            for obstacle in df_obstacle.values():
                state = State(
                    frame=round(frame["timestamp"] * 1000),
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

        return participants
