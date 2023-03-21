import numpy as np
from sklearn import svm
from sklearn.externals import joblib

from tactics2d.trajectory.element.trajectory import Trajectory


PARTICIPANT_TYPE = ["pedestrian", "cyclist", "vehicle"]


class GuessType:
    trajectory_clf = joblib.load("./trajectory_classifier.m")

    def guess_by_size(size_info: tuple, hint_type: str):
        """Guess the type of the participant by the size information with SVM model.

        Args:
            size_info (tuple): _description_
            hint_type (str): _description_
        """
        return

    def guess_by_trajectory(self, trajectory: Trajectory, hint_type: str):
        """Guess the type of the participant by the trajectory with SVM model.

        Args:
            trajectory (Trajectory): _description_
            hint_type (str): _description_

        Returns:
            _type_: _description_
        """
        history_speed = [state.speed for state in trajectory.history_states.values()]
        speed_mean = np.mean(history_speed)
        speed_std = np.std(history_speed)
        speed_max = np.max(history_speed)

        X = np.ndarray([[speed_mean, speed_std, speed_max]])
        svm_result = self.trajectory_clf.predict(X)

        return PARTICIPANT_TYPE[svm_result]
