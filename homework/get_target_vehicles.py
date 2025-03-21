import os
from copy import deepcopy

import numpy as np
import pandas as pd

from tactics2d.utils.common import get_absolute_path


def get_target_vehicles(file: str, folder: str):
    with open(os.path.join(get_absolute_path(folder), file)) as f:
        df_track_chunk = pd.read_csv(f, iterator=True, chunksize=10000)

        meta_data_dict = dict()

        for chunk in df_track_chunk:
            for _, info in chunk.iterrows():
                id_ = info["Vehicle_ID"]

                if id_ not in meta_data_dict:
                    meta_data_dict[id_] = dict()
                    meta_data_dict[id_]["File"] = file
                    meta_data_dict[id_]["Folder"] = folder
                    meta_data_dict[id_]["v_Length"] = info["v_Length"]
                    meta_data_dict[id_]["v_Width"] = info["v_Width"]
                    meta_data_dict[id_]["v_Class"] = info["v_Class"]
                    meta_data_dict[id_]["Start_Frame"] = info["Frame_ID"]
                    meta_data_dict[id_]["End_Frame"] = info["Frame_ID"]
                    meta_data_dict[id_]["num_Lane_Change"] = 0
                    last_lane = info["Lane_ID"]

                    meta_data_dict[id_]["Preceeding"] = {info["Preceeding"]}
                    meta_data_dict[id_]["Following"] = {info["Following"]}
                else:
                    meta_data_dict[id_]["Start_Frame"] = np.min(
                        [meta_data_dict[id_]["Start_Frame"], info["Frame_ID"]]
                    )
                    meta_data_dict[id_]["End_Frame"] = np.max(
                        [meta_data_dict[id_]["End_Frame"], info["Frame_ID"]]
                    )

                    if last_lane != info["Lane_ID"]:
                        meta_data_dict[id_]["num_Lane_Change"] += 1
                        last_lane = info["Lane_ID"]

                    meta_data_dict[id_]["Preceeding"].add(info["Preceeding"])
                    meta_data_dict[id_]["Following"].add(info["Following"])

        print(len(meta_data_dict))
        # Filter the followable vehicles
        # 0. keep a list of all the existing vehicles
        existing_ids = set(meta_data_dict.keys())
        meta_data_dict_all = deepcopy(meta_data_dict)

        # 1. remove all the vehicles that have changed lane
        for id_ in list(meta_data_dict.keys()):
            if meta_data_dict[id_]["num_Lane_Change"] > 0:
                meta_data_dict.pop(id_)

        print(len(meta_data_dict))

        # 2. check if the vehicle has a valid follower
        f.seek(0)
        df_track_chunk = pd.read_csv(f, iterator=True, chunksize=10000)

        for chunk in df_track_chunk:
            for _, info in chunk.iterrows():
                id_ = info["Vehicle_ID"]

                if id_ not in meta_data_dict:
                    continue

                for following_id in list(meta_data_dict[id_]["Following"]):
                    if following_id not in existing_ids:
                        meta_data_dict[id_]["Following"].remove(following_id)
                        continue

                    if int(id_) not in meta_data_dict_all[int(following_id)]["Preceeding"]:
                        print(id_, meta_data_dict[following_id])
                        meta_data_dict[id_]["Following"].remove(following_id)

                if len(meta_data_dict[id_]["Following"]) != 1:
                    meta_data_dict.pop(id_)

        print(len(meta_data_dict))

        # 3. only keep those targets with over 200 frame by the same follower
        f.seek(0)
        df_track_chunk = pd.read_csv(f, iterator=True, chunksize=10000)

        for chunk in df_track_chunk:
            for _, info in chunk.iterrows():
                id_ = info["Vehicle_ID"]

                if id_ not in meta_data_dict:
                    continue

                if int(info["Following"]) == int(list(meta_data_dict[id_]["Following"])[0]):
                    if "Start_Following" not in meta_data_dict[id_]:
                        meta_data_dict[id_]["Start_Following"] = info["Frame_ID"]

        for id_ in list(meta_data_dict.keys()):
            if meta_data_dict[id_]["End_Frame"] - meta_data_dict[id_]["Start_Following"] < 200:
                meta_data_dict.pop(id_)

    print(len(meta_data_dict))
    meta_data_df = pd.DataFrame.from_dict(meta_data_dict, orient="index")
    meta_data_df.index.name = "Vehicle_ID"
    meta_data_df.index = meta_data_df.index.astype(int)
    meta_data_df = meta_data_df.drop("Start_Frame", axis=1).drop("Preceeding")
    meta_data_df["Following"] = meta_data_df["Following"].apply(lambda x: int(list(x)[0]))

    return meta_data_df


if __name__ == "__main__":
    folder = "./data/NGSIM/US-101-LosAngeles-CA"
    file = "trajectories-0750am-0805am.csv"

    get_target_vehicles(file, folder)
