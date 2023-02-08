import os
import json
import multiprocessing

import pandas as pd


"""
This script provides a pre-processor for the Dragon Lake Parking dataset to convert it to
a more readable and storage-efficient format.
"""


class DLPPreprocess(object):
    def __init__(self):
        self.scene_id = None
        self.frames = {}
        self.agents = {}
        self.instances = {}
        self.scene = {}
        self.obstacles = {}

    def load(self, file_id, file_path):
        self.scene_id = file_id
        with open(file_path+"DJI_%04d_agents.json" % file_id, "r") as f:
            self.agents = json.load(f)
        with open(file_path+"DJI_%04d_frames.json" % file_id, "r") as f:
            self.frames = json.load(f)
        with open(file_path+"DJI_%04d_instances.json" % file_id, "r") as f:
            self.instances = json.load(f)
        with open(file_path+"DJI_%04d_scene.json" % file_id, "r") as f:
            self.scene = json.load(f)
        with open(file_path+"DJI_%04d_obstacles.json" % file_id, "r") as f:
            self.obstacles = json.load(f)

    @property
    def _valid_obstacles(self):
        obstacle_tokens = set(self.scene["obstacles"])
        obstacle_tokens_ = set(self.obstacles.keys())
        return len(obstacle_tokens.difference(obstacle_tokens_))  == 0

    def process_obstacles(self):
        if not self._valid_obstacles:
            print("Warning: The obstacle token in the scene and the obstacle files are inconsistent.")

        df = pd.DataFrame(
            columns=["sceneId", "type", "width", "length", "xCenter", "yCenter", "heading"])
        idx = 0
        for info in self.obstacles.values():
            df.loc[idx] = [
                self.scene_id, info["type"], info["size"][1], info["size"][0], 
                info["coords"][0], info["coords"][1], info["heading"]
            ]
            idx += 1
        return df

    @property
    def _valid_agents(self):
        agent_tokens = set(self.scene["agents"])
        agent_tokens_ = set(self.agents.keys())
        return len(agent_tokens.difference(agent_tokens_)) == 0

    def process_instances(self):
        if not self._valid_agents:
            print("Warning: The agent token in the scene and the agent files are inconsistent.")

        df = pd.DataFrame(columns=[
            "sceneId", "frame", "timeStamp", "type", "width", "length", "mode",
            "xCenter", "yCenter", "heading", "speed", "xAccel", "yAccel"
        ])

        idx = 0
        frame = 0

        for frame_info in self.frames.values():
            timestamp_ms = int(frame_info["timestamp"] * 1000)
            for instance_token in frame_info["instances"]:
                instance_info = self.instances[instance_token]
                agent_info = self.agents[instance_info["agent_token"]]
                df_line = [
                    self.scene_id, frame, timestamp_ms, 
                    agent_info["type"], agent_info["size"][1], agent_info["size"][0], 
                    instance_info["mode"], instance_info["coords"][0], instance_info["coords"][1], 
                    instance_info["heading"], instance_info["speed"],
                    instance_info["acceleration"][0], instance_info["acceleration"][1]
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
        # print("Processing the obstacles in scene %02d" % int(i+1))
        df_obstacle = processor.process_obstacles()
        # print("Processing the agents in scene %02d" % int(i+1))
        df_track = processor.process_instances()
        df_obstacle.to_csv(target_path+"%04d_obstacles.csv" % int(i+1), index=False)
        df_track.to_csv(target_path+"%04d_tracks.csv" % int(i+1), index=False)
        print("Finished processing scene %02d" % int(i+1))

    n_process = min(15, multiprocessing.cpu_count())
    pool = multiprocessing.Pool(processes=n_process)
    tasks = list(range(30))

    pool.map(sub_process, tasks)