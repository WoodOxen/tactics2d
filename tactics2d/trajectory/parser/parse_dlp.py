from typing import Tuple
import json
import ray
import time
import sys
from concurrent import futures


from tactics2d.participant.element import Vehicle, Pedestrian, Other, Cyclist
from tactics2d.trajectory.element import State, Trajectory

# D:/study/Tactics/TacticTest/tactics2d
TYPE_MAPPING = {
    "Car": "car",
    "Medium Vehicle": "car",
    "Bus": "bus",
    "Pedestrian": "pedestrian",
    "Bicycle": "cyclist",
    "Undefined": "other",
}

CLASS_MAPPING = {
    "Car": Vehicle,
    "Medium Vehicle": Vehicle,
    "Bus": Vehicle,
    "Pedestrian": Pedestrian,
    "Bicycle": Cyclist,
    "Undefined": Other,
}

# @ray.remote
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

    def load(
        self,
        file_id: int,
        folder_path: str,
    ):
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
            if (
                frame["timestamp"] < stamp_range[0]
                or frame["timestamp"] > stamp_range[1]
            ):
                continue

            for obstacle in df_obstacle.values():
                state = State(frame=round(frame["timestamp"] * 1000))

                if obstacle["obstacle_token"] not in participants:
                    participants[
                        obstacle["obstacle_token"]
                    ] = self._generate_participant(obstacle, id_cnt)
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

@ray.remote
def test_dlp_parser_remote(file_id: int, stamp_range: tuple):
    file_path = "./DLP"
    print("dlp started, id = ", file_id)
    trajectory_parser = DLPParser.remote()

    participants = trajectory_parser.parse.remote(
        file_id, file_path, stamp_range)
    result = ray.get(participants)
    print("dlp finished, id = ", file_id)
    return result

def test_dlp_parser(file_id: int):
    file_path = "./DLP"
    print("dlp started, id = ", file_id)
    trajectory_parser = DLPParser()

    participants = trajectory_parser.parse(
        file_id, file_path, (-float("inf"), float("inf")))
    print("dlp finished, id = ", file_id)
    print("id: ", file_id, ", lenth is ", len(participants))
    return participants


def ray_dlp_parser():
    ray.init(num_cpus=4)
    start = time.time()
    dlpResult = [0] * 31
    participant = [0] * 31
    for i in range(1,11):
        file_path = "./DLP"
        print("dlp started, id = ", i)
        trajectory_parser = DLPParser.remote()

        dlpResult[i] = trajectory_parser.parse.remote(
            i, file_path, (-float("inf"), float("inf")))
    for i in range(1,11):
        participant[i] = ray.get(dlpResult[i])
    for i in range(1,11):
        print("id: ", i, ", lenth is ", len(participant[i]))
    end = time.time()
    elapsed = end - start
    print ("time spend: ", elapsed)

def future_dlp_parser(workers):
    MAX_WORKERS = workers
    start = time.time()
    workers = min(MAX_WORKERS, 30)
    with futures.ThreadPoolExecutor(workers) as executor:  #Instantiate the thread pool
        # test_dlp_parser(i,(-float("inf"), float("inf")))
        res = executor.map(test_dlp_parser, range(1,2))
    print(list(res)[0])
    end = time.time()
    elapsed = end - start
    print("Time spent:",elapsed)
    result = list(res)
    return len(result)

#if __name__ == "__main__":
    # ray_dlp_parser()
    # # future_dlp_parser(4)