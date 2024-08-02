import math
import pickle

import numpy as np


class PredictionLoader:
    r"""

    Examples::

        >>> from prediction.predictor import PredictionLoader
        >>> predictor = PredictionLoader()
        >>> from DataLoader import WaymoDL
        >>> loader = WaymoDL('./waymo_tf_example/training')
        >>> data, _ = loader.get_next()
        >>> output = predictor(data)
        >>> print(output)
    """

    def __init__(self, all_agents=False):
        self.all_agents = all_agents

    def __call__(self, data, prediction_loaded, file_mode=0):
        """
        Args:
            data: the infos for the loading scenario
            prediction_loaded: loaded dictionary of the detection result
            file_mode: mode 0 - [scenario_id]: 'rst': np.array (n, 81, 2)

        Returns:
            agent_to_pred:
                a mapping which maps from agent_id to a dictionary {
                    'pred_trajectory': pred_trajectory,
                    'pred_yaw': pred_yaw
                }
                where pred_trajectory is a numpy array of shape (80, 2),
                pred_yaw is a numpy array of shape (80,).
        """

        scenario = data["scenario"]
        vectors = []
        agent_to_pred = {}

        # with open(result_path, 'rb') as f:
        #     prediction_loaded = pickle.load(f)

        for agent_id in data["agent"]:
            agent = data["agent"][agent_id]
            pose = agent["pose"]
            speed = agent["speed"]
            shape = agent["shape"]
            type = int(agent["type"])
            to_predict = int(agent["to_predict"])

            if self.all_agents or to_predict:
                # import random
                # if random.random() < 0.7:
                #     continue
                pred_trajectory = np.zeros([6, 80, 2])
                assert len(pose) >= 11, len(pose)
                x, y = pose[10, 0], pose[10, 1]
                pred_yaw = np.zeros([6, 80])
                pred_scores = np.zeros(6)

                def get_angle(x, y):
                    return math.atan2(y, x)

                delta_list = []
                for i in range(6, 10):
                    delta = pose[i, 0] - pose[i - 1, 0], pose[i, 1] - pose[i - 1, 1]
                    delta_list.append(delta)
                delta_x, delta_y = np.array(delta_list).mean(axis=0)

                # for i in range(80):
                #     x, y = x + delta_x, y + delta_y
                #     pred_trajectory[:, i, 0], pred_trajectory[:, i, 1] = x, y

                if scenario in prediction_loaded:
                    if agent_id in prediction_loaded[scenario]:
                        if "rst" in prediction_loaded[scenario][agent_id]:
                            # load without offset
                            pred_scores = np.exp(prediction_loaded[scenario][agent_id]["score"])
                            loaded_pred = prediction_loaded[scenario][agent_id]["rst"]  # result

                            for each_prediction in range(6):
                                # agent_index = prediction_loaded[scenario]['ids'].index(agent_id)
                                # pred_trajectory[each_prediction, :, :] = prediction_loaded[scenario]['rst'][each_prediction, agent_index, :, :]
                                pred_trajectory[each_prediction, :, :] = loaded_pred[
                                    each_prediction, :, :
                                ]
                                for i in range(80):
                                    if i > 0:
                                        x, y = (
                                            pred_trajectory[each_prediction, i - 1, 0],
                                            pred_trajectory[each_prediction, i - 1, 1],
                                        )
                                    else:
                                        x, y = pose[10, 0], pose[10, 1]
                                    pred_yaw[each_prediction, i] = get_angle(
                                        pred_trajectory[each_prediction, i, 0] - x,
                                        pred_trajectory[each_prediction, i, 1] - y,
                                    )
                                    if pred_yaw[each_prediction, i] < 0:
                                        pred_yaw[each_prediction, i] += 2.0 * math.pi
                                    # TODO: delta_x not defined
                                    if abs(delta_x) + abs(delta_y) < 0.01:
                                        pred_yaw[each_prediction, i] = pose[10, -1]
                        else:
                            print(list(prediction_loaded[scenario][agent_id].keys()))
                            assert (
                                False
                            ), f"rst not in prediction result, is it with time offset? if so, use the predictor with time offset"

                # if scenario in prediction_loaded:
                #     loaded_scores = np.exp(np.array(prediction_loaded[scenario]['score']))
                #     # print("test0:", loaded_scores, prediction_loaded[scenario]['ids'], agent_id)
                #     for each_prediction in range(6):
                #         for agent_index, each_agent_id in enumerate(prediction_loaded[scenario]['ids']):
                #             if int(each_agent_id) == int(agent_id):
                #                 pred_scores = loaded_scores[agent_index]
                #                 print("test:", float(pred_scores), int(pred_scores))
                #                 # agent_index = prediction_loaded[scenario]['ids'].index(agent_id)
                #                 # pred_trajectory[each_prediction, :, :] = prediction_loaded[scenario]['rst'][each_prediction, agent_index, :, :]
                #                 pred_trajectory[each_prediction, :, :] = prediction_loaded[scenario]['rst'][agent_index][each_prediction, :, :]
                #                 for i in range(80):
                #                     if i > 0:
                #                         x, y = pred_trajectory[each_prediction, i - 1, 0], pred_trajectory[each_prediction, i - 1, 1]
                #                     else:
                #                         x, y = pose[10, 0], pose[10, 1]
                #                     pred_yaw[each_prediction, i] = get_angle(pred_trajectory[each_prediction, i, 0] - x,
                #                                                              pred_trajectory[each_prediction, i, 1] - y)
                #                     if pred_yaw[each_prediction, i] < 0:
                #                         pred_yaw[each_prediction, i] += 2.0 * math.pi
                #                     # TODO: delta_x not defined
                #                     if abs(delta_x) + abs(delta_y) < 0.05:
                #                         pred_yaw[each_prediction, i] = pose[10, -1]
                # else:
                #     loaded_ids = prediction_loaded[scenario]['ids']
                #     print(f'agent {agent_id} in {loaded_ids}/{scenario} not found in prediction result')
                else:
                    # skip scenarios not in prediction result file
                    print(
                        f"scenario {scenario} not found in prediction result {prediction_loaded.keys()}"
                    )
                    return None

                agent_to_pred[agent_id] = {}
                agent_to_pred[agent_id]["pred_trajectory"] = pred_trajectory
                agent_to_pred[agent_id]["pred_yaw"] = pred_yaw
                if np.sum(pred_scores) > 0.01:
                    agent_to_pred[agent_id]["pred_scores"] = pred_scores / np.sum(pred_scores)
                else:
                    agent_to_pred[agent_id]["pred_scores"] = pred_scores
                # print(pose[10, -1], pred_yaw[10], pose[11:][:10, :2], pred_trajectory[:10])

        #                 info_dic = {'pred_trajectory': pred_trajectories,
        #                             'pred_yaw': pred_yaws,
        #                             'pred_scores': pred_scores}

        return agent_to_pred
