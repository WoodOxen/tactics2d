import copy
import math
import os
import time
from enum import IntEnum
from typing import List

import numpy as np
import prediction.utils as utils
import prediction.utils_cython as utils_cython
import torch
import torch.distributed as dist
from prediction.vectornet import VectorNet
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm


class TrajectoryType(IntEnum):
    STATIONARY = 0
    STRAIGHT = 1
    STRAIGHT_LEFT = 2
    STRAIGHT_RIGHT = 3
    LEFT_U_TURN = 4
    LEFT_TURN = 5
    RIGHT_U_TURN = 6
    RIGHT_TURN = 7


class AgentType(IntEnum):
    unset = 0
    vehicle = 1
    pedestrian = 2
    cyclist = 3
    other = 4

    @staticmethod
    def to_string(a: int):
        return str(AgentType(a)).split(".")[1]


def get_normalized(polygons, x, y, angle):
    cos_ = math.cos(angle)
    sin_ = math.sin(angle)
    n = polygons.shape[1]
    new_polygons = np.zeros_like(polygons, dtype=np.float32)
    assert new_polygons.shape[2] == 2, new_polygons.shape
    for polygon_idx in range(polygons.shape[0]):
        for i in range(n):
            polygons[polygon_idx, i, 0] -= x
            polygons[polygon_idx, i, 1] -= y
            new_polygons[polygon_idx, i, 0] = (
                polygons[polygon_idx, i, 0] * cos_ - polygons[polygon_idx, i, 1] * sin_
            )
            new_polygons[polygon_idx, i, 1] = (
                polygons[polygon_idx, i, 0] * sin_ + polygons[polygon_idx, i, 1] * cos_
            )
    return new_polygons


def predict_reactor_for_onepair(
    target_reactor_id,
    all_agent_trajectories,
    track_type_int,
    time_offset,
    raw_data,
    history_frame_num,
    future_frame_num,
    objects_id,
    args,
    model,
    device,
    threshold=0.5,
):
    start_time = time.time()
    mapping_to_return = []
    for i in range(0, objects_id.shape[0]):
        if objects_id[i] != target_reactor_id:
            continue

        all_agent_trajectories_this_batch = all_agent_trajectories.copy()
        objects_id_this_batch = objects_id.copy()
        # objects_id包含了所有agent的id
        # default without reactor's intentions
        gt_reactor = np.zeros_like(all_agent_trajectories_this_batch[i])  # 生成一个长度等于当前agent的轨迹长度的零向量
        last_valid_index = history_frame_num - 1  # 10
        # speed = utils.get_dis_point2point((gt_trajectory[0, history_frame_num - 1, 5], gt_trajectory[0, history_frame_num - 1, 6]))
        waymo_yaw = all_agent_trajectories_this_batch[i, last_valid_index, 3]
        # headings = gt_trajectory[0, history_frame_num:, 4].copy()
        angle = -waymo_yaw + math.radians(90)

        normalizer = utils.Normalizer(
            all_agent_trajectories_this_batch[i, last_valid_index, 0],
            all_agent_trajectories_this_batch[i, last_valid_index, 1],
            angle,
        )
        # normalize without Cython

        # print("Tag relation before normalize")
        # all_agent_trajectories_this_batch[:, :, :] = utils_cython.get_normalized(all_agent_trajectories_this_batch[:, :, :], normalizer)
        all_agent_trajectories_this_batch[:, :, :2] = get_normalized(
            all_agent_trajectories_this_batch[:, :, :2], normalizer.x, normalizer.y, normalizer.yaw
        )
        # 将轨迹中的所有点关于当前agent的坐标点旋转。旋转角度的大小为当前agent的航向角yaw

        # print("Tag relation after normalize1")
        # gt_reactor[:, :] = utils_cython.get_normalized(gt_reactor[:, :][np.newaxis, :], normalizer)[0]
        # print("Tag relation after normalize2")
        labels = all_agent_trajectories_this_batch[
            0, history_frame_num : history_frame_num + future_frame_num, :2
        ].copy()
        # labels是所有轨迹的xy坐标点

        image = np.zeros([224, 224, 60 + 90], dtype=np.int8)
        args.image = image

        # create some dummies just to check if it works
        gt_future_is_valid = np.ones_like(all_agent_trajectories_this_batch)[:, :, 0]
        # 第一个维度是agent的数量，第二个维度是轨迹的长度
        # print("Tag relation before classify")
        trajectory_type = utils_cython.classify_track(
            gt_future_is_valid[0], all_agent_trajectories_this_batch[0]
        )
        # 判断是静止、直行、左转或右转
        # print("Tag relation after classify")
        # if trajectory_type == 'STATIONARY':
        #     continue
        tracks_type = np.ones_like(all_agent_trajectories_this_batch)[:, 0, 0]
        # print("Tag relation before get agents")
        vectors, polyline_spans, trajs = utils_cython.get_agents(
            all_agent_trajectories_this_batch,
            gt_future_is_valid,
            tracks_type,
            False,
            args,
            gt_reactor,
        )
        # print("Tag relation after get agents")
        map_start_polyline_idx = len(polyline_spans)
        # print("Tag relation before get roads")
        vectors_, polyline_spans_, goals_2D, lanes = utils_cython.get_roads(
            raw_data, normalizer, args
        )
        # print("Tag relation after get roads")
        polyline_spans_ = polyline_spans_ + len(vectors)
        vectors = np.concatenate([vectors, vectors_])
        polyline_spans = np.concatenate([polyline_spans, polyline_spans_])
        polyline_spans = [slice(each[0], each[1]) for each in polyline_spans]

        stage_one_label = (
            np.argmin(
                [
                    utils.get_dis(lane, all_agent_trajectories_this_batch[i, -1, :2]).min()
                    for lane in lanes
                ]
            )
            if len(lanes) > 0
            else 0
        )

        mapping = {
            "matrix": vectors,
            "polyline_spans": polyline_spans,
            "map_start_polyline_idx": map_start_polyline_idx,
            "labels": labels,
            "normalizer": normalizer,
            "goals_2D": goals_2D,
            "polygons": lanes,
            "stage_one_label": stage_one_label,
            "waymo_yaw": waymo_yaw,
            "track_type_int": track_type_int,
            # 'track_type_string': AgentType.to_string(track_type_int),
            "trajectory_type": trajectory_type,
            "tracks_type": tracks_type,
            "eval_time": 80,
            "scenario_id": "001",
            # 'object_id': tf.convert_to_tensor(objects_id)[0],
            "inf_id": objects_id_this_batch[1],
            "all_agent_ids": objects_id_this_batch.copy(),
            # 'inf_label': inf_label,
            "image": args.image,
        }
        # if eval_time < 80:
        #     mapping['final_idx'] = eval_time - 1

        final_idx = mapping.get("final_idx", -1)

        mapping["goals_2D_labels"] = np.argmin(utils.get_dis(goals_2D, labels[final_idx]))
        mapping_to_return.append(mapping)

    # print(f"***** model loaded, predicting {len(mapping_to_return)} samples *****")
    if len(mapping_to_return) == 0:
        return None
    scores, all_agent_ids, scenario_ids = model(mapping_to_return, device)
    print(f"scores:{scores}")
    print(f"all_agent_ids:{all_agent_ids}")
    result_to_return = []
    for i, each_mapping in enumerate(mapping_to_return):
        reactor_id = int(all_agent_ids[i][0])
        inf_id = int(all_agent_ids[i][1])
        r_pred = np.argmax(scores[i])
        # print("test 000000: ", scores[i], reactor_id, inf_id)
        if r_pred == 1 and scores[i][r_pred] > threshold:  # add more filter logics here
            if [inf_id, reactor_id] not in result_to_return:
                print(inf_id, reactor_id)
                result_to_return.append([inf_id, reactor_id])
    # print(f"***** result unpacked for {time_offset} *****")
    end_time = time.time()
    print(f"predict_reactor_for_onepair_in{end_time-start_time}")
    return result_to_return


class RelationPredictor:
    def __init__(self, **kwargs):
        self.data = None
        self.threshold = 0.5
        self.model = None
        self.device = None
        self.args = None
        self.model_path = None
        self.predict_device = "cpu"
        self.max_prediction_num = 128
        self.rank = None
        self.prediction_data = None
        self.dataset = "Waymo"
        self.predicting_horizon = kwargs["time_horizon"] if "time_horizon" in kwargs else 80

        self.predicting_lock = False

    def __call__(self, **kwargs):
        # init predictor and load model
        if self.device is None:
            predict_device = kwargs["predict_device"]
            if predict_device in ["cpu", "mps"]:
                self.device = predict_device
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu", 0)
                self.rank = 0
                print(f"predicting relation with {self.device}")
        if self.args is None:
            utils.args = utils.Args
            self.args = utils.args
            assert (
                "laneGCN-4" in self.args.other_params and "raster" in self.args.other_params
            ), self.args.other_params

        if self.model is None:
            if "model" in kwargs and kwargs["model"] is not None:
                self.model = kwargs["model"]
            else:
                self.model_path = kwargs["model_path"]["relation_pred"]
                self.predict_device = kwargs["predict_device"]
                self.load_model()

        self.data = kwargs["new_data"]
        predictor_list = kwargs["predictor_list"]
        use_prediction = kwargs["use_prediction"]
        if True:  # 'predicting' not in self.data:
            self.data["predicting"] = {
                "ego_id": None,  # [select, selected_agent_id]
                "goal_pts": {
                    # 'sample_agent_id': [[0.0, 0.0], 3.14]
                },
                "follow_goal": {
                    # 'sample_agent_id': True
                },
                "relevant_agents": [],
                "relation": np.array([]),
                "colliding_pairs": [],
                "marginal_trajectory": {
                    # 'sample_agent_id': {
                    #     'rst': np.zeros([self.predicting_horizon, 6, 2]),
                    #     'score': np.zeros([self.predicting_horizon, 6])
                    # }
                },
                "conditional_trajectory": {
                    # 'sample_agent_id': {
                    #     'rst': np.zeros([self.predicting_horizon, 6, 2]),
                    #     'score': np.zeros([self.predicting_horizon, 6])
                    # }
                },
                "guilded_trajectory": {
                    # 'sample_agent_id': {
                    #     'rst': np.zeros([self.predicting_horizon, 6, 2]),
                    #     'score': np.zeros([self.predicting_horizon, 6])
                    # }
                },
                "XPt": {
                    # (reactor_id, inf_id): {
                    # 'pred_collision_pt': [0.0, 0.0],
                    # 'pred_cp_scores': 0.0,
                    # 'scenarios_id': scenario_ids[i],
                    # 'all_agents': all_agent_ids[i]}
                },
                "original_trajectory": copy.deepcopy(self.data["agent"]),
                "points_to_mark": [],
                "trajectory_to_mark": [],
                "emergency_stopping": False,
                "route": {},
                "all_relations_last_step": [],
                "goal_fit": None,  # True if goal fit the assigned route, calculate goal past
                "terminate_log": [],  # [terminate_frame_id, terminate_reason] terminate_reason: 'collision', 'offroad'
            }
        self.predicting_horizon = kwargs["time_horizon"]
        # self.goal_setter()
        if predictor_list is not None:
            _, relation_predictor, _ = predictor_list
            self.model_path = kwargs["model_path"]
            self.predict_device = kwargs["predict_device"]
            self.model = relation_predictor.model

        agent_dic = self.data["agent"]
        select = -1
        selected_agent_id = -1

        if self.dataset == "Waymo":
            interact_only = False

            for i, agent_id in enumerate(agent_dic):
                # select the ego agent from the two interactive agents once per scenario
                # TODO: extend ego selection for the validation set
                if agent_dic[agent_id]["type"] != 1:
                    continue
                if interact_only:
                    if agent_dic[agent_id]["to_interact"]:
                        if np.max(agent_dic[agent_id]["pose"][:11, :2]) == -1:
                            print("skip invalid")
                        else:
                            select = i
                            selected_agent_id = agent_id
                else:
                    if agent_dic[agent_id]["to_predict"]:
                        if np.max(agent_dic[agent_id]["pose"][:11, :2]) == -1:
                            print("skip invalid")
                        else:
                            select = i
                            selected_agent_id = agent_id

            # selected_agent_id = 857
            # selecting ego agent for planning
            self.data["predicting"]["ego_id"] = [select, selected_agent_id]

            # if len(list(agent_dic.keys())) > 30:
            #     self.data['skip'] = True
            #     print("skipping large agent number scenarios")

            if selected_agent_id == -1:
                print("Predictor Skip without ego agent")
                self.data["skip"] = True

        elif self.dataset == "NuPlan":
            if "ego" in agent_dic:
                selected_agent_id = "ego"
                select = list(agent_dic.keys()).index("ego")
            else:
                print("Predictor Skip without ego agent")
                self.data["skip"] = True

            self.data["predicting"]["ego_id"] = [select, selected_agent_id]

    def load_model(self):
        print("Start loading relation prediction model")
        if self.rank is not None:
            # setup(self.rank, 512)
            torch.cuda.set_device(self.rank)
        model = VectorNet(self.args).to(self.device)
        model_recover_path = self.model_path
        model.eval()
        torch.no_grad()

        # print("***** Recover model: %s *****", model_recover_path)
        if not torch.cuda.is_available() or self.predict_device == "cpu":
            model_recover = torch.load(model_recover_path, map_location="cpu")
        elif self.predict_device == "mps":
            model_recover = torch.load(model_recover_path, map_location="mps")
        else:
            model_recover = torch.load(model_recover_path)
        model.load_state_dict(model_recover)
        # model.module.load_state_dict(model_recover)
        self.model = model
        print("Model loaded for relation prediction")

    def predict_one_time(
        self, current_data, each_pair, current_frame=0, predict_with_rules=True, clear_history=True
    ):
        end_time = time.time()
        # 对给定的一对代理进行单次预测
        start_time = time.time()
        self.predicting_lock = True
        data = copy.deepcopy(current_data)
        future_frame_num = 80
        args = self.args
        device = self.device
        model = self.model

        skip_flag = data["skip"]
        agent_dic = data["agent"]
        assert not skip_flag

        history_frame_num = 11

        all_agent_trajectories = []
        track_type_int = []
        objects_id = np.array(list(agent_dic.keys()))  # 获得了所有的id

        # relevant_agent_ids = self.data['predicting']['relevant_agents']
        # assert len(relevant_agent_ids) > 0
        ego_agent, target_agent = each_pair

        # select, selected_id = self.data['predicting']['ego_id']
        for i, agent_id in enumerate(agent_dic):
            if len(all_agent_trajectories) >= self.max_prediction_num:
                break
            # add all agent pose and type to a numpy array
            pose = agent_dic[agent_id]["pose"]
            if isinstance(pose, list):
                short = 91 - len(pose)
                print(f"short in prediction - {short} ; {agent_id}")
                if short > 0:
                    pose += (np.ones([short, 4]) * -1).tolist()
            all_agent_trajectories.append(pose[:91, :])
            track_type_int.append(agent_dic[agent_id]["type"])
        # assert select != -1, 'no interact agent found, failed to select ego in the predictor'
        all_agent_trajectories = np.array(all_agent_trajectories, dtype=np.float32)
        track_type_int = np.array(track_type_int)

        relationship_predicted = []
        # for current_frame in range(0, future_frame_num, time_interval):
        if current_frame > 0:
            all_agent_trajectories = np.concatenate(
                [
                    all_agent_trajectories[:, current_frame:, :],
                    np.zeros_like(all_agent_trajectories[:, :current_frame, :]),
                ],
                axis=1,
            )

        # WARNING: Adding a non-learning rear collision solver
        def normalize_angle(angle):
            """
            Normalize an angle to [-pi, pi].
            :param angle: (float)
            :return: (float) Angle in radian in [-pi, pi]
            """
            while angle > np.pi:
                angle -= 2.0 * np.pi

            while angle < -np.pi:
                angle += 2.0 * np.pi

            return angle

        same_direction_threshold = 30

        ego_idx = np.where(objects_id == ego_agent)[0]
        target_idx = np.where(objects_id == target_agent)[0]
        ego_yaw = all_agent_trajectories[ego_idx, 0, 3]
        target_yaw = all_agent_trajectories[target_idx, 0, 3]
        yaw_diff = normalize_angle(ego_yaw - target_yaw)

        def get_angle_of_a_line(pt1, pt2):
            # angle from horizon to the right, counter-clockwise,
            x1, y1 = pt1
            x2, y2 = pt2
            angle = math.atan2(y2 - y1, x2 - x1)
            return angle

        if predict_with_rules:
            # 如果两者朝向的夹角小于30°，则前车为influencer，后车为reactor
            if (
                -math.pi / 180 * same_direction_threshold
                < yaw_diff
                < math.pi / 180 * same_direction_threshold
            ):
                # check rear collision relationship instead of making a prediction
                ego_to_target = get_angle_of_a_line(
                    all_agent_trajectories[ego_idx, 0, :2][0],
                    all_agent_trajectories[target_idx, 0, :2][0],
                )
                ego_yaw_diff = normalize_angle(ego_yaw - ego_to_target)
                if (
                    -math.pi / 180 * same_direction_threshold
                    < ego_yaw_diff
                    < math.pi / 180 * same_direction_threshold
                ):
                    return [[target_agent, ego_agent]]
                target_to_ego = get_angle_of_a_line(
                    all_agent_trajectories[target_idx, 0, :2][0],
                    all_agent_trajectories[ego_idx, 0, :2][0],
                )
                target_yaw_diff = normalize_angle(target_yaw - target_to_ego)
                if (
                    -math.pi / 180 * same_direction_threshold
                    < target_yaw_diff
                    < math.pi / 180 * same_direction_threshold
                ):
                    return [[ego_agent, target_agent]]

            # always yield to non-vehicle agents
            if agent_dic[target_agent]["type"] != 1:
                return [[target_agent, ego_agent]]

        objects_id = np.array(objects_id)
        # check forward relation
        select = np.where(objects_id == ego_agent)[0]
        if not isinstance(select, int):
            select = select[0]
        print(f"predicting {ego_agent} at {select} with {each_pair}")

        # swap the ego to index 1
        def swap(tensor):
            if isinstance(tensor[0], int) or isinstance(
                tensor[0], str
            ):  # id for Waymo is int and id for NuPlan is a string
                tensor[select], tensor[1] = tensor[1], tensor[select]
            else:
                tensor[select], tensor[1] = tensor[1].copy(), tensor[select].copy()

        for each in [all_agent_trajectories, objects_id]:
            swap(each)
        assert objects_id[1] == ego_agent, objects_id

        # not predict
        # return []

        if data["dataset"] == "NuPlan":
            predicted_relationships = predict_reactor_for_onepair_NuPlan(
                target_reactor_id=target_agent,
                all_agent_trajectories=all_agent_trajectories,
                track_type_int=track_type_int,
                time_offset=current_frame,
                history_frame_num=history_frame_num,
                future_frame_num=future_frame_num,
                objects_id=objects_id,
                road_dic=data["road"],
                args=args,
                model=model,
                device=device,
                threshold=self.threshold,
            )
        elif data["dataset"] == "Waymo":
            predicted_relationships = predict_reactor_for_onepair(
                target_reactor_id=target_agent,
                all_agent_trajectories=all_agent_trajectories,
                track_type_int=track_type_int,
                time_offset=current_frame,
                history_frame_num=history_frame_num,
                future_frame_num=future_frame_num,
                objects_id=objects_id,
                raw_data=data["raw"],
                args=args,
                model=model,
                device=device,
                threshold=self.threshold,
            )
        else:
            assert False, f"Unknown dataset: " + str(data["dataset"])

        if predicted_relationships is not None:
            relationship_predicted += predicted_relationships
            # print(f'predicted forward adding {predicted_relationships} to {relationship_predicted}')

        # check backward relation
        select = np.where(objects_id == target_agent)[0]
        if not isinstance(select, int):
            select = select[0]
        # print(f'predicting {target_agent} at {select} with {each_pair}')

        # swap the ego to index 1
        def swap2(tensor):
            if isinstance(tensor[0], int) or isinstance(
                tensor[0], str
            ):  # id for Waymo is int and id for NuPlan is a string
                tensor[select], tensor[1] = tensor[1], tensor[select]
            else:
                tensor[select], tensor[1] = tensor[1].copy(), tensor[select].copy()

        for each in [all_agent_trajectories, objects_id]:
            swap2(each)
        assert objects_id[1] == target_agent, objects_id

        if data["dataset"] == "NuPlan":
            predicted_relationships = predict_reactor_for_onepair_NuPlan(
                target_reactor_id=ego_agent,
                all_agent_trajectories=all_agent_trajectories,
                track_type_int=track_type_int,
                time_offset=current_frame,
                history_frame_num=history_frame_num,
                future_frame_num=future_frame_num,
                objects_id=objects_id,
                road_dic=data["road"],
                args=args,
                model=model,
                device=device,
                threshold=self.threshold,
            )
        elif data["dataset"] == "Waymo":
            predicted_relationships = predict_reactor_for_onepair(
                target_reactor_id=ego_agent,
                all_agent_trajectories=all_agent_trajectories,
                track_type_int=track_type_int,
                time_offset=current_frame,
                history_frame_num=history_frame_num,
                future_frame_num=future_frame_num,
                objects_id=objects_id,
                raw_data=data["raw"],
                args=args,
                model=model,
                device=device,
                threshold=self.threshold,
            )

        if predicted_relationships is not None:
            relationship_predicted += predicted_relationships
        # print(f'predicted backwards adding {predicted_relationships} to {relationship_predicted}')
        torch.cuda.empty_cache()
        self.args.image = None

        self.predicting_lock = False
        if clear_history:
            self.data["predicting"]["relation"] = relationship_predicted
        else:
            self.data["predicting"]["relation"] += relationship_predicted
        end_time = time.time()
        print(f"predict_one_time_in{end_time-start_time}")

    def setting_goal_points(self, current_data):
        start_time = time.time()
        # select one ego vehicle
        agent_dic = current_data["agent"]
        for i, agent_id in enumerate(agent_dic):
            # set the goal point for each agent
            self.data["predicting"]["goal_pts"][agent_id] = self.get_goal(
                current_data=current_data, agent_id=agent_id, dataset=self.dataset
            )
            if self.data["predicting"]["goal_pts"][agent_id][0] is not None:
                self.data["predicting"]["follow_goal"][agent_id] = True
            else:
                self.data["predicting"]["follow_goal"][agent_id] = False
        print("Goal points settled")
        end_time = time.time()
        print(f"setting_goal_points_in{end_time-start_time}")

    def get_goal(self, current_data, agent_id, dataset="Waymo") -> List:
        # get last valid point as the goal point
        # agent_dic = current_data['agent'][agent_id]
        agent_dic = current_data["predicting"]["original_trajectory"][agent_id]
        yaw = None
        point = None
        if dataset == "Waymo":
            # Waymo
            for frame_idx in range(1, 80):
                if yaw is not None:
                    break
                if (
                    agent_dic["pose"][-frame_idx][0] != -1
                    and agent_dic["pose"][-frame_idx][1] != -1
                ):
                    point = [agent_dic["pose"][-frame_idx][0], agent_dic["pose"][-frame_idx][1]]
                    yaw = agent_dic["pose"][-frame_idx][3]
                    break
        elif dataset == "NuPlan":
            # NuPlan
            assert "ego_goal" in current_data, "Goal Setter: Not found goal in data dic"
            goal = current_data["ego_goal"]
            if agent_id == "ego" and goal is not None:
                point = [goal[0], goal[1]]
                yaw = goal[3]
            else:
                for frame_idx in range(1, 180):
                    if yaw is not None:
                        break
                    if (
                        agent_dic["pose"][-frame_idx][0] != -1
                        and agent_dic["pose"][-frame_idx][1] != -1
                    ):
                        point = [agent_dic["pose"][-frame_idx][0], agent_dic["pose"][-frame_idx][1]]
                        yaw = agent_dic["pose"][-frame_idx][3]
                        break
                if point is None:
                    if agent_id == "ego":
                        # print('ERROR: goal point is none ', agent_dic['pose'], agent_id)
                        print("[Static goal] ERROR: goal point is none ", agent_id)
                    point = [0, 0]
                    yaw = 0
        return [point, yaw]
