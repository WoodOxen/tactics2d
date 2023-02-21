import os
import json
import multiprocessing

import pandas as pd


class DLPPreprocess(object):
    """
    This processor converts the Dragon Lake Parking dataset to a more readable and 
    storage-efficient format.
    """

    def __init__(self):
        self.scene_id = None
        self.frames = {}
        self.agents = {}
        self.instances = {}
        self.scene = {}
        self.obstacles = {}

    def load(self, file_id, folder_path):
        self.scene_id = file_id
        with open(os.path.join(folder_path, "DJI_%04d_agents.json" % file_id), "r") as f:
            self.agents = json.load(f)
        with open(os.path.join(folder_path, "DJI_%04d_frames.json" % file_id), "r") as f:
            self.frames = json.load(f)
        with open(os.path.join(folder_path, "DJI_%04d_instances.json" % file_id), "r") as f:
            self.instances = json.load(f)
        with open(os.path.join(folder_path, "DJI_%04d_scene.json" % file_id), "r") as f:
            self.scene = json.load(f)
        with open(os.path.join(folder_path, "DJI_%04d_obstacles.json" % file_id), "r") as f:
            self.obstacles = json.load(f)

    def _valid_obstacles(self):
        obstacle_tokens = set(self.scene["obstacles"])
        obstacle_tokens_ = set(self.obstacles.keys())
        return len(obstacle_tokens.difference(obstacle_tokens_))  == 0

    def _valid_agents(self):
        agent_tokens = set(self.scene["agents"])
        agent_tokens_ = set(self.agents.keys())
        return len(agent_tokens.difference(agent_tokens_)) == 0

    def process_tracks(self) -> pd.DataFrame:
        if not self._valid_agents():
            print("Warning: The agent token in the scene and the agent files are inconsistent.")
        if not self._valid_obstacles():
            print("Warning: The obstacle token in the scene and the obstacle files are inconsistent.")

        df = pd.DataFrame(columns=[
            "sceneId", "trackId", "frame", "timeStamp", "type", "width", "length", "mode",
            "xCenter", "yCenter", "heading", "speed", "xAccel", "yAccel"
        ])

        idx = 0
        frame = 0
        id_cnt = 0
        track_ids = {}

        for frame_info in self.frames.values():
            timestamp_ms = round(frame_info["timestamp"] * 1000)

            for obstacle_info in self.obstacles.values():
                if obstacle_info["obstacle_token"] not in track_ids:
                    track_ids[obstacle_info["obstacle_token"]] = id_cnt
                    id_cnt += 1

            for instance_token in frame_info["instances"]:
                instance_info = self.instances[instance_token]
                agent_info = self.agents[instance_info["agent_token"]]
                if instance_info["agent_token"] not in track_ids:
                    track_ids[instance_info["agent_token"]] = id_cnt
                    id_cnt += 1

                df_line = [
                    self.scene_id, track_ids[instance_info["agent_token"]], frame, timestamp_ms, 
                    agent_info["type"], agent_info["size"][1], agent_info["size"][0], 
                    instance_info["mode"], instance_info["coords"][0], instance_info["coords"][1], 
                    instance_info["heading"], instance_info["speed"],
                    instance_info["acceleration"][0], instance_info["acceleration"][1]
                ]
                df.loc[idx] = df_line
                idx += 1

            for obstacle_info in self.obstacles.values():
                df_line = [
                    self.scene_id, track_ids[obstacle_info["obstacle_token"]], frame, timestamp_ms, 
                    obstacle_info["type"], obstacle_info["size"][1], obstacle_info["size"][0], 
                    "obstacle", obstacle_info["coords"][0], obstacle_info["coords"][1], 
                    obstacle_info["heading"], 0, 0, 0
                ]
                df.loc[idx] = df_line
                idx += 1

            frame += 1

        return df

    def clean(self):
        self.frames.clear()
        self.agents.clear()
        self.instances.clear()
        self.scene.clear()
        self.scene.clear()


if __name__ == "__main__":
    source_path = "../data/trajectory_raw/DLP/"
    target_path = "../data/trajectory_processed/DLP/"
    processor = DLPPreprocess()

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    def sub_process(i):
        processor.load(i+1, source_path)
        print("Start processing scene %02d" % int(i+1))
        df_track = processor.process_instances()
        df_track.to_csv(target_path+"%04d_tracks.csv" % int(i+1), index=False)
        print("Finished processing scene %02d" % int(i+1))

    n_process = min(16, multiprocessing.cpu_count())
    pool = multiprocessing.Pool(processes=n_process)
    tasks = list(range(30))

    pool.map(sub_process, tasks)