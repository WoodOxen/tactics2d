##! python3
# -*- coding: utf-8 -*-
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# @File: test_dataset_parser.py
# @Description: This file implements the test cases for the dataset parser.
# @Author: Yueyuan Li
# @Version: 1.0.0

import sys

sys.path.append(".")
sys.path.append("..")

import os
import time
import logging

from zipfile import ZipFile

import pytest

from tactics2d.dataset_parser import (
    ArgoverseParser,
    DLPParser,
    InteractionParser,
    LevelXParser,
    NuPlanParser,
    WOMDParser,
)


@pytest.mark.dataset_parser
@pytest.mark.parametrize(
    "sub_folder, expected",
    [
        ("train/0a0a2bb7-c4f4-44cd-958a-9ee15cb34aca", 40),
        ("test/0a0af725-fbc3-41de-b969-3be718f694e2", 19),
        ("val/00a0ec58-1fb9-4a2b-bfd7-f4e5da7a9eff", 73),
    ],
)
def test_argoverse_parser(sub_folder, expected):
    dataset_folder = "./tactics2d/data/trajectory_sample/Argoverse"
    folder = os.path.join(dataset_folder, sub_folder)
    dataset_parser = ArgoverseParser()
    files = os.listdir(folder)
    map_file = [file for file in files if ".json" in file][0]
    trajectory_files = [file for file in files if ".parquet" in file][0]

    t1 = time.time()
    participants, _ = dataset_parser.parse_trajectory(trajectory_files, folder)
    t2 = time.time()
    _ = dataset_parser.parse_map(map_file, folder)
    t3 = time.time()
    assert len(participants) == expected
    logging.info(f"The time needed to parse an Argoverse trajectory file: {t2 - t1}s")
    logging.info(f"The time needed to parse an Argoverse map file: {t3 - t2}s")


@pytest.mark.dataset_parser
@pytest.mark.parametrize(
    "dataset, file_id, stamp_range, expected",
    [
        ("highD", 1, (-float("inf"), float("inf")), 1047),
        ("inD", "0", (-float("inf"), float("inf")), 384),
        ("rounD", "00", (-float("inf"), float("inf")), 348),
        ("exiD", "01_tracks.csv", None, 871),
        ("uniD", "00_tracksMeta.csv", None, 362),
        ("highD", "01_recordkingMeta.csv", (0, 10000), 20),
        ("inD", 0, (0, 10000), 10),
        ("rounD", 0, (0, 10000), 12),
        ("exiD", 1, (0, 10000), 38),
        ("uniD", 0, (0, 10000), 47),
    ],
)
def test_levelx_parser(dataset: str, file_id: int, stamp_range: tuple, expected: int):
    file_path = f"./tactics2d/data/trajectory_sample/{dataset}/data/"

    dataset_parser = LevelXParser(dataset)

    t1 = time.time()
    participants, _ = dataset_parser.parse_trajectory(file_id, file_path, stamp_range)
    t2 = time.time()

    assert len(participants) == expected
    logging.info(f"The time needed to parse a {dataset} scenario: {t2 - t1}s")


@pytest.mark.dataset_parser
@pytest.mark.parametrize(
    "file_id, stamp_range, expected",
    [(0, None, 97), (0, (-float("inf"), float("inf")), 97), (0, (0, 10000), 5)],
)
def test_interaction_parser(file_id: int, stamp_range: tuple, expected: int):
    folder_path = (
        "./tactics2d/data/trajectory_sample/INTERACTION/recorded_trackfiles/DR_USA_Intersection_EP0"
    )
    dataset_parser = InteractionParser()

    t1 = time.time()
    participants, _ = dataset_parser.parse_trajectory(file_id, folder_path, stamp_range)
    t2 = time.time()

    assert len(participants) == expected
    logging.info(f"The time needed to parse an INTERACTION scenario: {t2 - t1}s")


@pytest.mark.dataset_parser
@pytest.mark.parametrize(
    "file_id, stamp_range, expected",
    [
        (12, None, 342),
        ("DJI_0012_frames.json", (-float("inf"), float("inf")), 342),
        ("12", (0, 10000), 252),
    ],
)
def test_dlp_parser(file_id: int, stamp_range: tuple, expected: int):
    zip_path = "./tactics2d/data/trajectory_sample/DLP.zip"
    unzip_path = "./tactics2d/data/trajectory_sample"
    file_path = "./tactics2d/data/trajectory_sample/DLP"

    with ZipFile(zip_path, "r") as zipObj:
        zipObj.extractall(unzip_path)

    dataset_parser = DLPParser()

    t1 = time.time()
    participants, _ = dataset_parser.parse_trajectory(file_id, file_path, stamp_range)
    t2 = time.time()

    assert (len(participants)) == expected
    logging.info(f"The time needed to parse a DLP scenario: {t2 - t1}s")


@pytest.mark.dataset_parser
@pytest.mark.parametrize(
    "file_name, stamp_range, expected",
    [
        ("train_boston/2021.08.26.18.24.36_veh-28_00578_00663.db", None, 343),
        ("train_pittsburgh/2021.09.13.19.54.06_veh-45_00781_00843.db", None, 57),
        (
            "train_singapore/2021.09.29.01.04.10_veh-49_00808_00872.db",
            (-float("inf"), float("inf")),
            35,
        ),
        (
            "train_vegas_1/2021.05.18.21.31.22_veh-30_00062_00160.db",
            (-float("inf"), float("inf")),
            285,
        ),
        ("val/2021.08.24.12.39.05_veh-42_01860_01929.db", None, 45),
        ("test/2021.09.16.14.14.03_veh-45_00441_00502.db", None, 48),
    ],
)
def test_nuplan_parser(file_name: str, stamp_range: tuple, expected: int):
    folder_path = "./tactics2d/data/trajectory_sample/NuPlan/data/cache"

    dataset_parser = NuPlanParser()

    t1 = time.time()
    participants, _ = dataset_parser.parse_trajectory(file_name, folder_path, stamp_range)
    t2 = time.time()

    assert (len(participants)) == expected
    logging.info(f"The time needed to parse a NuPlan scenario: {t2 - t1}s")


@pytest.mark.dataset_parser
@pytest.mark.parametrize("scenario_id", [(None), (0), (10), ("637f20cafde22ff8"), ("not_exist")])
def test_womd_parser(scenario_id):
    folder_path = "./tactics2d/data/trajectory_sample/WOMD"
    file_name = "motion_data_one_scenario.tfrecord"

    dataset_parser = WOMDParser()

    t1 = time.time()
    participants, _ = dataset_parser.parse_trajectory(
        scenario_id, file=file_name, folder=folder_path
    )
    t2 = time.time()
    _ = dataset_parser.parse_map(scenario_id, file=file_name, folder=folder_path)
    t3 = time.time()
    assert len(participants) == 83
    logging.info(f"The time needed to parse trajectories in a WOMD scenario: {t2 - t1}s")
    logging.info(f"The time needed to parse the map for a WOMD scenario: {t3 - t2}s")
