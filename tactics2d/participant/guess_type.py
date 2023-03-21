from tactics2d.trajectory.element.trajectory import Trajectory
from sklearn import svm
import joblib

import sys
sys.path.append(".")
sys.path.append("..")
import logging
import numpy as np
logging.basicConfig(level=logging.DEBUG)



DEFAULT_GUESS_TYPE = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "cyclist": "cyclist",
    "pedestrian": "pedestrian",
}


class GuessType:
    @staticmethod
    def guess_by_size(size_info: tuple, hint_type: str):
        return DEFAULT_GUESS_TYPE[hint_type]

    @staticmethod
    def guess_by_trajectory(trajectory: Trajectory, hint_type: str):
        return DEFAULT_GUESS_TYPE[hint_type]
    
    @staticmethod
    def get_svm_model(model_name:str = "svm_model2.m"):
        clf = joblib.load("./" + model_name)
        return clf
    
    @staticmethod
    def guess_by_svm(clf:svm._classes.SVC, trajectory: Trajectory, hint_type: str):
        states_num = len(trajectory.history_states)
        speed_average = trajectory.average_speed
        speeddif_sum = 0
        # accel_sum = 0
        for state in trajectory.history_states.values():
            speeddif_sum += (state.speed - speed_average)**2
            # accel_sum += state.accel
        speed_variance = speeddif_sum / states_num
        # accel_average = accel_sum / states_num

        speed_max = max(state.speed for state in trajectory.history_states.values())
        X = np.ndarray(shape=(1,3))
        X[0][0] = speed_average
        X[0][1] = speed_variance
        X[0][2] = speed_max
        svm_result = clf.predict(X)
        logging.debug(str(svm_result[0]) + "[" + str(X[0][0]) + "," + str(X[0][1]) + "," + str(X[0][2]) + "]")
        return "pedestrian" if svm_result == 0 else "cyclist"
