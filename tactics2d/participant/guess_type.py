import numpy as np
import joblib

from tactics2d.trajectory.element.trajectory import Trajectory


class GuessType:
    """This class provides a set of SVM classifiers to roughly guess the type of a traffic participant based on different features.

    The training process of the SVM classifiers are in the ./utils folder.
    """

    def __init__(self):
        self.trajectory_clf = joblib.load("./tactics2d/participant/trajectory_classifier.m")

    def guess_by_size(size_info: tuple, hint_type: str):
        """Guess the type of the participant by the size information with SVM model.

        This method is usually used to distinguish different type of vehicles.

        Args:
            size_info (tuple): _description_
            hint_type (str): _description_
        """
        return

    def guess_by_trajectory(self, trajectory: Trajectory) -> str:
        """Guess the type of the participant by the trajectory with SVM model.

        This method is recommend for distinguishing the pedestrians from the cyclists.

        Args:
            trajectory (Trajectory): _description_
            hint_type (str): _description_

        Returns:
            _type_: _description_
        """
        history_speed = np.array([state.speed for state in trajectory.history_states.values()])
        history_heading = np.array([state.heading for state in trajectory.history_states.values()])
        speed_max = np.max(history_speed)
        speed_min = np.min(history_speed)
        speed_mean = np.mean(history_speed)
        speed_std = np.std(history_speed)
        heading_changing_std = np.std(history_heading[1:] - history_heading[:-1])

        X = np.array([[speed_min, speed_max, speed_mean, speed_std, heading_changing_std]])
        y_predict = self.trajectory_clf.predict(X)

        return y_predict[0]
