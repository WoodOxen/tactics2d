from typing import Tuple
import json

from tactics2d.participant.element import Vehicle, Pedestrian, Other
from tactics2d.trajectory.element import State, Trajectory


TYPE_MAPPING = {
    "Car": "car",
    "Medium Vehicle": "car",
    "Bus": "bus",
    "Pedestrian": "pedestrian",
    "Undefined": "other",
}

CLASS_MAPPING = {
    "Car": Vehicle,
    "Medium Vehicle": Vehicle,
    "Bus": Vehicle,
    "Pedestrian": Pedestrian,
    "Undefined": Other,
}


class DLPParser(object):
    """
    This class implements a parser of the Dragon Lake Parking Dataset.

    Shen, Xu, et al. "Parkpredict: Motion and intent prediction of vehicles in parking lots." 2020 IEEE Intelligent Vehicles Symposium (IV). IEEE, 2020.
    """

    def _generate_participant(self, instance, id_):
        type_ = TYPE_MAPPING[instance["type"]]
        class_ = CLASS_MAPPING[instance["type"]]
        participant = class_(
            id_=id_,
            type_=type_,
            length=instance["size"][0],
            width=instance["size"][1],
            trajectory=Trajectory(id_=id_, fps=25.0),
        )

        return participant

    def load(self, file_id: int, folder_path: str):
        with open("%s/DJI_%04d_agents.json" % (folder_path, file_id), "r") as f_agent:
            df_agent = json.load(f_agent)
        with open("%s/DJI_%04d_frames.json" % (folder_path, file_id), "r") as f_frame:
            df_frame = json.load(f_frame)
        with open(
            "%s/DJI_%04d_instances.json" % (folder_path, file_id), "r"
        ) as f_instance:
            df_instance = json.load(f_instance)
        with open(
            "%s/DJI_%04d_obstacles.json" % (folder_path, file_id), "r"
        ) as f_obstacle:
            df_obstacle = json.load(f_obstacle)

        return df_agent, df_frame, df_instance, df_obstacle

    def parse(
        self,
        file_id: int,
        folder_path: str,
        stamp_range: Tuple[float, float] = (-float("inf"), float("inf")),
    ):
        """_summary_

        Args:
            file_id (int): _description_
            folder_path (str): _description_
            stamp_range (Tuple[float, float], optional): _description_. Defaults to None.
        """

        df_agent, df_frame, df_instance, df_obstacle = self.load(file_id, folder_path)

        participants = {}
        id_cnt = 0

        for frame in df_frame.values():
            if frame["timestamp"] < stamp_range[0] or frame["timestamp"] > stamp_range[1]:
                continue

            for obstacle in df_obstacle.values():
                state = State(frame=round(frame["timestamp"] * 1000))

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
                    ax=instance["acceleration"],
                    ay=instance["acceleration"],
                )
                state.set_speed(instance["speed"])

                if instance["agent_token"] not in participants:
                    participants[instance["agent_token"]] = self._generate_participant(
                        df_agent[instance["agent_token"]], id_cnt
                    )
                    id_cnt += 1

                participants[instance["agent_token"]].trajectory.append_state(state)

        return participants
