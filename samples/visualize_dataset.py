##! python3
# -*- coding: utf-8 -*-
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: visualize_dataset.py
# @Description: This script provides the code to reproduce the demo of scenario=s from datasets in the README.
# @Author: Yueyuan Li
# @Version: 1.0.0

import sys

sys.path.append("..")

import os
import argparse
import json
import xml.etree.ElementTree as ET
import time

import matplotlib.pyplot as plt

from tactics2d.dataset_parser import (
    ArgoverseParser,
    DLPParser,
    InteractionParser,
    LevelXParser,
    NuPlanParser,
    WOMDParser,
)
from tactics2d.map.parser import Lanelet2Parser


DATASET_MAPPING = {
    "highd": "highD",
    "ind": "inD",
    "round": "rounD",
    "exid": "exiD",
    "unid": "uniD",
    "argoverse": "Argoverse",
    "dlp": "DLP",
    "interaction": "INTERACTION",
    "nuplan": "NuPlan",
    "womd": "WOMD",
}


def parse_data(args):
    map_path = "./tactics2d/data/map"
    config_path = "./tactics2d/data/map/map.config"
    with open(config_path, "r") as f:
        configs = json.load(f)

    t1 = time.time()

    if args.dataset in ["highd", "ind", "round", "exid"]:
        dataset_parser = LevelXParser(DATASET_MAPPING[args.dataset])
        trajectories = dataset_parser.parse_trajectory(args.file, args.folder)

        map_id = dataset_parser.get_location(args.file, args.folder)
        map_name = (
            "%s_%s" % (DATASET_MAPPING[args.dataset], map_id)
            if args.map_name is None
            else args.map_name
        )
        map_config = configs[map_name]
        map_path = os.path.join(map_path, map_config["osm_path"])
        map_root = ET.parse(map_path).getroot()
        map_ = Lanelet2Parser.parse(map_root, configs[map_name])

    elif args.dataset == "argoverse":
        dataset_parser = ArgoverseParser()
        trajectories = dataset_parser.parse_trajectory(args.file, args.folder)

        map_ = dataset_parser.parse_map(args.map_file, args.folder)

    elif args.dataset == "dlp":
        dataset_parser = DLPParser()
        trajectories = dataset_parser.parse_trajectory(args.file, args.folder)

        map_config = configs["DLP"]
        map_path = os.path.join(map_path, map_config["osm_path"])
        map_root = ET.parse(map_path).getroot()
        map_ = Lanelet2Parser.parse(map_root, map_config)

    elif args.dataset == "interaction":
        dataset_parser = InteractionParser()
        trajectories = dataset_parser.parse_trajectory(args.file, args.folder)

        map_config = configs[args.map_name]
        map_path = os.path.join(map_path, map_config["osm_path"])
        map_root = ET.parse(map_path).getroot()
        map_ = Lanelet2Parser.parse(map_root, configs[args.map_name])

    elif args.dataset == "nuplan":
        dataset_parser = NuPlanParser()
        trajectories = dataset_parser.parse_trajectory(args.file, args.folder)

        location = dataset_parser.get_location(args.file, args.folder)
        map_ = dataset_parser.parse_map(configs[location]["gpkg_path"], map_path)

    elif args.dataset == "womd":
        dataset_parser = WOMDParser()
        trajectories = dataset_parser.parse_trajectory(file=args.file, folder=args.folder)

        # map_ = dataset_parser.parse_map(file=args.map_file, folder=args.folder)
        map_ = dataset_parser.parse_map(
            file="motion_data_one_scenario.tfrecord", folder=args.folder
        )
    else:
        raise ValueError(f"Dataset {args.dataset} is not supported.")

    t2 = time.time()
    print("Time to parse a %s scenario: %.03f" % (DATASET_MAPPING[args.dataset], t2 - t1))

    return trajectories, map_


def visualize_data(trajectories, map_, args):
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "highd",
            "ind",
            "round",
            "exid",
            "argoverse",
            "dlp",
            "interaction",
            "nuplan",
            "womd",
        ],
        required=True,
        help="Choose the dataset to visualize. The options are: highd, ind, round, exid, argoverse, dlp, interaction, nuplan, womd.",
    )

    parser.add_argument(
        "--folder",
        type=str,
        default="../tactics2d/data/trajectory_sample",
        help="The folder path of the dataset.",
    )

    parser.add_argument("--file", type=str, help="The name of the data file.")

    parser.add_argument("--map-file", type=str, help="The name of the map file.")

    parser.add_argument("--map-name", type=str, help="The name of the map.")

    parser.add_argument("--export", type=str)

    args = parser.parse_args()

    parse_data(args)
