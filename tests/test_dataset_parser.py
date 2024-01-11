import sys

sys.path.append(".")
sys.path.append("..")

import time
import logging

from zipfile import ZipFile

import pytest

from tactics2d.dataset_parser import (
    DLPParser,
    InteractionParser,
    LevelXParser,
    NuPlanParser,
    WOMDParser,
)


@pytest.mark.dataset_parser
@pytest.mark.parametrize(
    "dataset, file_id, stamp_range, expected",
    [
        ("highD", 1, (-float("inf"), float("inf")), 1047),
        ("inD", 0, (-float("inf"), float("inf")), 384),
        ("rounD", 0, (-float("inf"), float("inf")), 348),
        ("exiD", 1, (-float("inf"), float("inf")), 871),
        ("uniD", 0, (-float("inf"), float("inf")), 362),
        ("highD", 1, (100.0, 400.0), 386),
        ("inD", 0, (100.0, 400.0), 134),
        ("rounD", 0, (100.0, 400.0), 113),
        ("exiD", 1, (100.0, 300.0), 312),
        ("uniD", 0, (100.0, 300.0), 229),
    ],
)
def test_levelx_parser(dataset: str, file_id: int, stamp_range: tuple, expected: int):
    file_path = f"./tactics2d/data/trajectory_sample/{dataset}/data/"

    dataset_parser = LevelXParser(dataset)

    t1 = time.time()
    participants = dataset_parser.parse_trajectory(file_id, file_path, stamp_range)
    t2 = time.time()

    assert len(participants) == expected
    logging.info(f"The time needed to parse a {dataset} scenario: {t2 - t1}s")


@pytest.mark.dataset_parser
@pytest.mark.parametrize(
    "file_id, stamp_range, expected",
    [(0, (-float("inf"), float("inf")), 97), (0, (100.0, 400.0), 12)],
)
def test_interaction_parser(file_id: int, stamp_range: tuple, expected: int):
    folder_path = (
        "./tactics2d/data/trajectory_sample/INTERACTION/recorded_trackfiles/DR_USA_Intersection_EP0"
    )
    dataset_parser = InteractionParser()

    t1 = time.time()
    participants = dataset_parser.parse_trajectory(file_id, folder_path, stamp_range)
    t2 = time.time()

    assert len(participants) == expected
    logging.info(f"The time needed to parse an INTERACTION scenario: {t2 - t1}s")


@pytest.mark.dataset_parser
@pytest.mark.parametrize(
    "file_id, stamp_range, expected",
    [(12, (-float("inf"), float("inf")), 342), (12, (100.0, 400.0), 317)],
)
def test_dlp_parser(file_id: int, stamp_range: tuple, expected: int):
    zip_path = "./tactics2d/data/trajectory_sample/DLP.zip"
    unzip_path = "./tactics2d/data/trajectory_sample"
    file_path = "./tactics2d/data/trajectory_sample/DLP"

    with ZipFile(zip_path, "r") as zipObj:
        zipObj.extractall(unzip_path)

    dataset_parser = DLPParser()

    t1 = time.time()
    participants = dataset_parser.parse_trajectory(file_id, file_path, stamp_range)
    t2 = time.time()

    assert (len(participants)) == expected
    logging.info(f"The time needed to parse a DLP scenario: {t2 - t1}s")


@pytest.mark.dataset_parser
@pytest.mark.parametrize(
    "file_name, stamp_range, expected",
    [
        ("train_boston/2021.08.26.18.24.36_veh-28_00578_00663.db", None, 343),
        ("train_pittsburgh/2021.09.13.19.54.06_veh-45_00781_00843.db", None, 57),
        ("train_singapore/2021.09.29.01.04.10_veh-49_00808_00872.db", None, 35),
        ("train_vegas_1/2021.05.18.21.31.22_veh-30_00062_00160.db", None, 285),
        ("val/2021.08.24.12.39.05_veh-42_01860_01929.db", None, 45),
        ("test/2021.09.16.14.14.03_veh-45_00441_00502.db", None, 48),
    ],
)
def test_nuplan_parser(file_name: str, stamp_range: tuple, expected: int):
    folder_path = "./tactics2d/data/trajectory_sample/NuPlan/data/cache"

    dataset_parser = NuPlanParser()

    t1 = time.time()
    participants = dataset_parser.parse_trajectory(file_name, folder_path, stamp_range)
    t2 = time.time()

    assert (len(participants)) == expected
    logging.info(f"The time needed to parse a NuPlan scenario: {t2 - t1}s")
    # print(len(participants))


@pytest.mark.dataset_parser
def test_womd_parser():
    folder_path = "./tactics2d/data/trajectory_sample/WOMD"
    file_name = "motion_data_one_scenario.tfrecord"

    dataset_parser = WOMDParser()

    t1 = time.time()
    participants = dataset_parser.parse_trajectory(
        None, file_name=file_name, folder_path=folder_path
    )
    t2 = time.time()
    logging.info(f"The time needed to parse a WOMD scenario: {t2 - t1}s")
