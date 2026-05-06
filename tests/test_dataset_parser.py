#! python3
# Copyright (C) 2024, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for dataset parser."""


import sys

sys.path.append(".")
sys.path.append("..")

import logging
import os
import time
from zipfile import ZipFile

import pytest

from tactics2d.dataset_parser import (
    Argoverse2Parser,
    CitySimParser,
    DLPParser,
    InteractionParser,
    LevelXParser,
    NGSIMParser,
    NuPlanParser,
    WOMDParser,
)
from tactics2d.dataset_parser.parse_driveinsightd import DriveInsightDParser
from tactics2d.map.map_config import NUPLAN_MAP_CONFIG


@pytest.mark.dataset_parser
@pytest.mark.parametrize(
    "sub_folder, expected",
    [
        ("train/0a0a2bb7-c4f4-44cd-958a-9ee15cb34aca", 40),
        ("test/0a0af725-fbc3-41de-b969-3be718f694e2", 19),
        ("val/00a0ec58-1fb9-4a2b-bfd7-f4e5da7a9eff", 73),
    ],
)
def test_argoverse2_parser(sub_folder, expected):
    dataset_folder = "./tactics2d/data/trajectory_sample/Argoverse"
    folder = os.path.join(dataset_folder, sub_folder)
    dataset_parser = Argoverse2Parser()
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
@pytest.mark.parametrize("ids, expected", [(None, 136), ([1, 2, 3], 3)])
def test_citysim_parser(ids, expected):
    folder = "./tactics2d/data/trajectory_sample/CitySim/Intersection B (Non-Signalized Intersection)/Trajectories"
    file = "IntersectionB-01.csv"
    dataset_parser = CitySimParser()

    t1 = time.time()
    participants, _ = dataset_parser.parse_trajectory(file, folder, ids=ids)
    t2 = time.time()

    assert len(participants) == expected
    logging.info(f"The time needed to parse a CitySim trajectory file: {t2 - t1}s")


@pytest.mark.dataset_parser
@pytest.mark.parametrize(
    "dataset, file_id, stamp_range, expected",
    [
        ("highD", 1, (-float("inf"), float("inf")), 1047),
        ("inD", "0", (-float("inf"), float("inf")), 384),
        ("rounD", "00", (-float("inf"), float("inf")), 348),
        ("exiD", "01_tracks.csv", None, 871),
        ("uniD", "00_tracksMeta.csv", None, 362),
        ("highD", "01_recordingMeta.csv", (0, 10000), 20),
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
    "file, stamp_range, ids",
    [
        ("trajectories-0750am-0805am.csv", None, None),
        ("trajectories-0750am-0805am.csv", None, [2]),
        ("trajectories-0750am-0805am.csv", None, [2, 4]),
    ],
)
def test_ngsim_parser(file, stamp_range, ids):
    folder = "./tactics2d/data/trajectory_sample/NGSIM"
    parser = NGSIMParser()

    t1 = time.time()
    participants, _ = parser.parse_trajectory(file, folder, stamp_range, ids)
    t2 = time.time()

    if ids is not None:
        assert len(ids) == len(participants)
    logging.info(f"The time needed to parse a NGSIM file: {t2 - t1}s.")

    parser.extract_meta_data(file, folder)


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
    map_folder_path = "./tactics2d/data/map/NuPlan"

    dataset_parser = NuPlanParser()

    t1 = time.time()
    participants, _ = dataset_parser.parse_trajectory(file_name, folder_path, stamp_range)
    t2 = time.time()
    location = dataset_parser.get_location(file_name, folder_path)
    map_path = NUPLAN_MAP_CONFIG[location]["gpkg_file"]

    try:
        _ = dataset_parser.parse_map(map_path, map_folder_path)
    except:
        logging.info(f"{map_path}")

    t3 = time.time()

    assert (len(participants)) == expected
    logging.info(f"The time needed to parse a NuPlan scenario: {t2 - t1}s")
    logging.info(f"The time needed to parse the map for a NuPlan scenario: {t3 - t2}s")


@pytest.mark.dataset_parser
@pytest.mark.parametrize(
    "file_name, scenario_id, expected",
    [
        (
            "uncompressed_scenario_training_training.tfrecord-00001-of-01000",
            "610c7fcfe8e7eb4d",
            {
                "time_range": (0, 9000),
                "classes": {"Vehicle": 44, "Pedestrian": 42, "Cyclist": 21},
                "areas": {"crosswalk": 6, "drivable_area": 10},
                "regs": {"traffic_light": 11},
                "lane_count": 79,
                "roadline_count": 63,
            },
        ),
        (
            "uncompressed_scenario_validation_interactive_validation_interactive.tfrecord-00000-of-00150",
            "234dfbe99b740c80",
            {
                "time_range": (0, 8997),
                "classes": {"Vehicle": 52, "Pedestrian": 2, "Cyclist": 1},
                "areas": {"crosswalk": 32, "speed_bump": 18, "drivable_area": 36},
                "regs": {"stop_sign": 10, "traffic_light": 6},
                "lane_count": 664,
                "roadline_count": 143,
            },
        ),
        (
            "uncompressed_scenario_validation_validation.tfrecord-00001-of-00150",
            None,
            {
                "time_range": (0, 9002),
                "classes": {"Vehicle": 202, "Pedestrian": 6},
                "areas": {"crosswalk": 14, "speed_bump": 7, "drivable_area": 20},
                "regs": {"stop_sign": 2, "traffic_light": 13},
                "lane_count": 327,
                "roadline_count": 111,
            },
        ),
    ],
)
def test_womd_parser_official_shards(file_name: str, scenario_id: str, expected: dict):
    folder_path = "./tactics2d/data/trajectory_sample/WOMD"
    full_path = os.path.join(folder_path, file_name)
    if not os.path.isfile(full_path):
        pytest.skip(f"Official WOMD shard not found: {full_path}")

    dataset_parser = WOMDParser()

    t1 = time.time()
    participants, actual_time_range = dataset_parser.parse_trajectory(
        scenario_id, file=file_name, folder=folder_path
    )
    t2 = time.time()
    map_ = dataset_parser.parse_map(scenario_id, file=file_name, folder=folder_path)
    t3 = time.time()

    class_counts = {}
    for participant in participants.values():
        class_name = type(participant).__name__
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    area_counts = {}
    for area in map_.areas.values():
        area_counts[area.subtype] = area_counts.get(area.subtype, 0) + 1

    reg_counts = {}
    for regulation in map_.regulations.values():
        reg_counts[regulation.subtype] = reg_counts.get(regulation.subtype, 0) + 1

    actual = {
        "time_range": actual_time_range,
        "classes": class_counts,
        "areas": area_counts,
        "regs": reg_counts,
        "lane_count": len(map_.lanes),
        "roadline_count": len(map_.roadlines),
    }

    assert actual == expected

    logging.info(f"The time needed to parse official WOMD shard trajectories: {t2 - t1}s")
    logging.info(f"The time needed to parse official WOMD shard map: {t3 - t2}s")


@pytest.mark.dataset_parser
@pytest.mark.parametrize(
    "file_name, scenario_id, expected_light_count",
    [
        (
            "uncompressed_scenario_training_training.tfrecord-00001-of-01000",
            "610c7fcfe8e7eb4d",
            11,
        ),
        (
            "uncompressed_scenario_validation_interactive_validation_interactive.tfrecord-00000-of-00150",
            "234dfbe99b740c80",
            6,
        ),
        (
            "uncompressed_scenario_validation_validation.tfrecord-00001-of-00150",
            None,
            13,
        ),
    ],
)
def test_womd_dynamic_traffic_lights(file_name: str, scenario_id: str, expected_light_count: int):
    folder_path = "./tactics2d/data/trajectory_sample/WOMD"
    full_path = os.path.join(folder_path, file_name)
    if not os.path.isfile(full_path):
        pytest.skip(f"Official WOMD shard not found: {full_path}")

    dataset_parser = WOMDParser()
    map_ = dataset_parser.parse_map(scenario_id, file=file_name, folder=folder_path)

    dynamic_lights = [reg for reg in map_.regulations.values() if reg.subtype == "traffic_light"]
    lane_centerlines_complete = all("centerline" in lane.custom_tags for lane in map_.lanes.values())
    lane_boundaries_complete = all(
        lane.left_side is not None
        and len(lane.left_side.coords) >= 2
        and lane.right_side is not None
        and len(lane.right_side.coords) >= 2
        for lane in map_.lanes.values()
    )
    signal_metadata_complete = all(
        reg.dynamic
        and reg.custom_tags
        and "states" in reg.custom_tags
        and len(reg.custom_tags["states"]) > 0
        for reg in dynamic_lights
    )

    assert len(dynamic_lights) == expected_light_count
    assert lane_centerlines_complete
    assert lane_boundaries_complete
    assert signal_metadata_complete


@pytest.mark.dataset_parser
@pytest.mark.parametrize(
    "scenario_id, folder, map_name, expected",
    [
        ("6",    "./tactics2d/data/trajectory_sample/DriveInsightD/jp_taito",     "jp_taito.xodr",      13),
        ("11",   "./tactics2d/data/trajectory_sample/DriveInsightD/cz_zlin",      "cz_zlin.xodr",       None),
        ("4464", "./tactics2d/data/trajectory_sample/DriveInsightD/us_coldwater", "usa_coldwater.xodr",  None),
    ],
)
def test_driveinsightd_parser(scenario_id: str, folder: str, map_name: str, expected):
    if not os.path.isdir(folder):
        pytest.skip(f"Dataset folder not found: {folder}")

    dataset_parser = DriveInsightDParser()

    t1 = time.time()
    participants, _ = dataset_parser.parse_trajectory(file=scenario_id, folder=folder)
    t2 = time.time()
    _ = dataset_parser.parse(scenario_id=scenario_id, folder=folder, map_name=map_name)
    t3 = time.time()

    if expected is not None:
        assert len(participants) == expected
    else:
        assert len(participants) > 0
    logging.info(f"The time needed to parse a DriveInsightD trajectory file: {t2 - t1}s")
    logging.info(f"The time needed to parse a DriveInsightD map file: {t3 - t2}s")
