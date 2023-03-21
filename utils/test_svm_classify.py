import sys

sys.path.append(".")
sys.path.append("..")
import logging

logging.basicConfig(level=logging.DEBUG)

from tactics2d.trajectory.parser import InteractionParser
from tactics2d.participant.guess_type import GuessType

# from tactics2d.trajectory.parser import DLPParser, InteractionParser, LevelXParser


def test_interaction_parser(dataset: str, file_id: int, stamp_range: tuple):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    folder_path = f"D:/study/Tactics/TacticTest/tactics2d/INTERACTION-Dataset-DR-v1_1/recorded_trackfiles/{dataset}/"
    clf = GuessType.get_svm_model("svm_model2.m")
    trajectory_parser = InteractionParser()

    participants = trajectory_parser.parse(file_id, folder_path, clf, stamp_range)
    print("length = ", len(participants))


"""
("DR_DEU_Roundabout_OF", 11, (-float("inf"), float("inf")))
("DR_USA_Intersection_EP0", 7, (-float("inf"), float("inf")))
("DR_USA_Intersection_EP1", 5, (-float("inf"), float("inf")))
("DR_USA_Intersection_GL", 59, (-float("inf"), float("inf")))
("DR_USA_Intersection_MA", 21, (-float("inf"), float("inf")))
("DR_USA_Roundabout_EP", 7, (-float("inf"), float("inf")))
("DR_USA_Roundabout_FT", 45, (-float("inf"), float("inf")))
("DR_USA_Roundabout_SR", 9, (-float("inf"), float("inf")))

"""
if __name__ == "__main__":
    test_interaction_parser("DR_USA_Roundabout_SR", 9, (-float("inf"), float("inf")))
