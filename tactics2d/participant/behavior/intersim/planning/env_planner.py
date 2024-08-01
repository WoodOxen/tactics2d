import numpy as np
import math
import logging
import copy
import random
import time

import envs.util as utils
import plan.utils as plan_helper
from agents.car import Agent

S0 = 2
T = 0.25 #1.5  # reaction time when following
DELTA = 4  # the power term in IDM
PLANNING_HORIZON = 5  # in frames
PREDICTION_HTZ = 10  # prediction_htz
T_HEADWAY = 0.2
A_SPEEDUP_DESIRE = 0.3  # A
A_SLOWDOWN_DESIRE = 1.5  # B
XPT_SHRESHOLD = 0.7
MINIMAL_DISTANCE_PER_STEP = 0.05
MINIMAL_DISTANCE_TO_TRAVEL = 4
# MINIMAL_DISTANCE_TO_RESCALE = -999 #0.1
REACTION_AFTER = 200  # in frames
MINIMAL_SCALE = 0.3
MAX_DEVIATION_FOR_PREDICTION = 4
TRAFFIC_LIGHT_COLLISION_SIZE = 2

MINIMAL_SPEED_TO_TRACK_ORG_GOAL = 5
MINIMAL_DISTANCE_TO_GOAL = 15

OFF_ROAD_DIST = 30

PRINT_TIMER = False
DRAW_CBC_PTS = False

class EnvPlanner:
    """
    EnvPlanner is capable of using as much information as it can to satisfy its loss like avoiding collisions.
    EnvPlanner can assume it's controlling all agents around if it does not exacerbate the sim-2-real gap.
    While the baseline planner or any planner controlling the ego vehicle can only use the prediction or past data
    """

    def __init__(self, env_config, predictor, dataset='Waymo', map_api=None):
        """
        This function initializes the planner with the required configurations and components.

        Args:
            env_config: A configuration object that contains environment-specific settings.
            predictor: The predictor component used for forecasting the behavior of agents.
            dataset (str, optional): The name of the dataset being used ('Waymo' by default).
            map_api (optional): An API for map-related functionalities.

        Attributes:
            (Various self attributes are initialized based on env_config and other parameters.)
        """
        self.planning_from = env_config.env.planning_from
        self.planning_interval = env_config.env.planning_interval
        self.planning_horizon = env_config.env.planning_horizon
        self.planning_to = env_config.env.planning_to
        self.scenario_frame_number = 0
        self.online_predictor = predictor
        self.method_testing = env_config.env.testing_method  # 0=densetnt with dropout, 1=0+post-processing, 2=1+relation
        self.test_task = env_config.env.test_task
        self.all_relevant = env_config.env.all_relevant
        self.follow_loaded_relation = env_config.env.follow_loaded_relation
        self.follow_prediction_traj = env_config.env.follow_prediction
        self.target_lanes = [0, 0]  # lane_index, point_index
        self.routed_traj = {}
        self.follow_gt_first = env_config.env.follow_gt_first

        self.predict_env_for_ego_collisions = env_config.env.predict_env_for_ego_collisions
        self.predict_relations_for_ego = env_config.env.predict_relations_for_ego
        self.predict_with_rules = env_config.env.predict_with_rules
        self.frame_rate = env_config.env.frame_rate

        self.current_on_road = True
        self.dataset = dataset
        self.online_predictor.dataset = dataset

        self.valid_lane_types = [1, 2] if self.dataset == 'Waymo' else [0, 11]
        self.vehicle_types = [1] if self.dataset == 'Waymo' else [0, 7]  # Waymo: Unset=0, Vehicle=1, Pedestrian=2, Cyclist=3, Other=4
        self.map_api = map_api  # NuPlan only
        self.past_lanes = {}

    def reset(self, *args, **kwargs):
        """
        This function resets the planner's state and reset the online predictor with new data.

        Args:
            args: Additional positional arguments, if needed.
            kwargs: Keyword arguments including 'new_data', 'model_path', 'time_horizon', 'predict_device', and 'ego_planner'.

        Notes:
            - The time taken for resetting the predictor is measured and printed.
            - The 'current_on_road' flag is set to True, indicating the ego agent is on the road.
        """
        time1 = time.perf_counter()
        self.online_predictor(new_data=kwargs['new_data'], model_path=kwargs['model_path'],
                              time_horizon=kwargs['time_horizon'], predict_device=kwargs['predict_device'],
                              use_prediction=(self.follow_prediction_traj or self.predict_env_for_ego_collisions) and kwargs['ego_planner'],
                              predictor_list=kwargs['predictor_list'])
        time2 = time.perf_counter()
        self.online_predictor.setting_goal_points(current_data=kwargs['new_data'])
        self.current_on_road = True
        print(f"predictor reset with {time2-time1:04f}s")
        # self.data = self.online_predictor.data

    def is_planning(self, current_frame_idx):
        """
        This function checks if the current frame index is within the planning window and meets the planning conditions.

        Args:
            current_frame_idx (int): The current frame index to evaluate for planning.

        Returns:
            bool: True if the current frame index is suitable for planning; False otherwise.
        """
        self.scenario_frame_number = current_frame_idx
        frame_diff = self.scenario_frame_number - self.planning_from
        if frame_diff >= 0 and frame_diff % self.planning_interval == 0:
            return True
        return False

    def is_first_planning(self, current_frame_idx):
        """
        This function checks if it's the very first planning frame.

        Args:
            current_frame_idx (int): The current frame index to evaluate.

        Returns:
            bool: True if this is the first planning frame; False otherwise.
        """
        self.scenario_frame_number = current_frame_idx
        frame_diff = self.scenario_frame_number - self.planning_from
        if frame_diff >= 0 and frame_diff == 0: # frame_diff % self.planning_interval == 0:
            return True
        return False

    def collision_based_relevant_detection(self, current_frame_idx, current_state, predict_ego=True):
        """
        This function detects relevant agents based on potential collisions for planning purposes.

        Args:
            current_frame_idx (int): The index of the current frame in the scenario.
            current_state (dict): A dictionary containing the state of the current scenario.
            predict_ego (bool, optional): A flag to indicate whether to predict ego agent collisions. Defaults to True.

        Yields:
            A list of colliding agent pairs that are relevant for the prediction.

        Modifies current_state to record the updated relevant agents and their colliding pairs.
        """
        ego_agent = current_state['predicting']['ego_id'][1]
        # print("before: ", current_state['predicting']['relevant_agents'], bool(current_state['predicting']['relevant_agents']))
        if not current_state['predicting']['relevant_agents']:
            relevant_agents = [ego_agent]
            undetected_piles = [ego_agent]
        else:
            relevant_agents = current_state['predicting']['relevant_agents'].copy()
            if ego_agent not in relevant_agents:
                relevant_agents += [ego_agent]
            undetected_piles = relevant_agents.copy()
        colliding_pairs = []
        while len(undetected_piles) > 0:
            if self.all_relevant:
                # hard force all agents as relevant
                current_agent = undetected_piles.pop()
                for each_agent_id in current_state['agent']:
                    if each_agent_id != current_agent:
                        relevant_agents.append(each_agent_id)
                break

            current_agent = undetected_piles.pop()
            ego_poses = current_state['agent'][current_agent]['pose']
            ego_shape = current_state['agent'][current_agent]['shape'][0]
            detected_pairs = []
            ego_agent_0 = None
            for idx, each_pose in enumerate(ego_poses):
                if idx <= current_frame_idx:
                    continue
                ego_agent_packed =Agent(x=each_pose[0],
                                             y=each_pose[1],
                                             yaw=each_pose[3],
                                             length=max(1, ego_shape[1]),
                                             width=max(1, ego_shape[0]),
                                             agent_id=current_agent)
                if ego_agent_0 is None:
                    ego_agent_0 = ego_agent_packed
                for each_agent_id in current_state['agent']:
                    if [current_agent, each_agent_id] in detected_pairs:
                        continue
                    if each_agent_id == current_agent or each_agent_id in relevant_agents:
                        continue
                    each_agent_frame_num = current_state['agent'][each_agent_id]['pose'].shape[0]
                    if idx >= each_agent_frame_num:
                        continue
                    target_agent_packed =Agent(x=current_state['agent'][each_agent_id]['pose'][idx, 0],
                                                    y=current_state['agent'][each_agent_id]['pose'][idx, 1],
                                                    yaw=current_state['agent'][each_agent_id]['pose'][idx, 3],
                                                    length=current_state['agent'][each_agent_id]['shape'][0][1],
                                                    width=current_state['agent'][each_agent_id]['shape'][0][0],
                                                    agent_id=each_agent_id)
                    if each_pose[0] == -1 or each_pose[1] == -1 or current_state['agent'][each_agent_id]['pose'][idx, 0] == -1 or current_state['agent'][each_agent_id]['pose'][idx, 1] == -1:
                        continue
                    collision = utils.check_collision(ego_agent_packed, target_agent_packed)
                    if collision:
                        detected_pairs.append([current_agent, each_agent_id])
                        yield_ego = True

                        # FORWARD COLLISION CHECKINGS
                        collision_0 = utils.check_collision(ego_agent_0, target_agent_packed)
                        if collision_0:
                            detected_relation = [[ego_agent_0, target_agent_packed]]
                        else:
                            # check relation
                            # print(f"In: {current_agent} {each_agent_id} {undetected_piles} {current_state['predicting']['relation']}")
                            self.online_predictor.predict_one_time(each_pair=[current_agent, each_agent_id],
                                                                        current_frame=current_frame_idx,
                                                                        clear_history=True,
                                                                        current_data=current_state)
                            # print(f"Out: {current_agent} {each_agent_id} {undetected_piles} {current_state['predicting']['relation']}")
                            detected_relation = current_state['predicting']['relation']
                            if [each_agent_id, current_agent] in detected_relation:
                                if [current_agent, each_agent_id] in detected_relation:
                                    # bi-directional relations, still yield
                                    pass
                                else:
                                    yield_ego = False

                        if yield_ego or self.method_testing < 2:
                            relevant_agents.append(each_agent_id)
                            undetected_piles.append(each_agent_id)
                            if [current_agent, each_agent_id] not in colliding_pairs and [each_agent_id, current_agent] not in colliding_pairs:
                                colliding_pairs.append([current_agent, each_agent_id])

            # print(f"Detected for {current_agent} with {undetected_piles}")
        if self.test_task != 1:
            # don't predict ego
            relevant_agents.remove(ego_agent)
        current_state['predicting']['relevant_agents'] = relevant_agents
        current_state['predicting']['colliding_pairs'] = colliding_pairs
        # print(f"Collision based relevant agent detected finished: \n{relevant_agents} \n{colliding_pairs}")

    def clear_markers_per_step(self, current_state, current_frame_idx):
        """
        Clears markers for each step if planning is to be performed at the current frame index.

        Args:
            current_state (dict): The current state of the prediction environment.
            current_frame_idx (int): The current frame index to evaluate for planning requirements.
        """
        if self.is_planning(current_frame_idx):
            current_state['predicting']['relation'] = []
            current_state['predicting']['points_to_mark'] = []
            current_state['predicting']['trajectory_to_mark'] = []

    def get_prediction_trajectories(self, current_frame_idx, current_state=None, time_horizon=80):
        """
        Retrieves prediction trajectories based on the current state and planning conditions.

        Args:
            current_frame_idx (int): The index of the current frame to evaluate.
            current_state (dict, optional): The current state dictionary; defaults to None.
            time_horizon (int, optional): The prediction time horizon in frames; defaults to 80.

        Returns:
            bool: True if planning is performed, False otherwise.
        """
        if self.is_planning(current_frame_idx):
            frame_diff = self.scenario_frame_number - self.planning_from
            self.collision_based_relevant_detection(current_frame_idx, current_state)
            current_state['predicting']['relation'] = []
            for each_pair in current_state['predicting']['colliding_pairs']:
                self.online_predictor.predict_one_time(each_pair=each_pair, current_data=current_state,
                                                            current_frame=current_frame_idx)
            self.online_predictor.last_predict_frame = frame_diff + 5
            return True
        else:
            return False
        
    def find_closest_lane(self,current_state, agent_id=None, 
                        my_current_pose=None,
                        my_current_v_per_step=None,
                        include_unparallel=True,
                        selected_lanes=[],
                        valid_lane_types=[1, 2],
                        excluded_lanes=[]):
        """
        Finds the closest lane to the current pose of an agent from the provided state.

        Args:
            current_state (dict): The current state of the road and agents.
            agent_id (int, optional): The identifier of the agent; defaults to None.
            my_current_pose (list, optional): The current pose [x, y, yaw] of the agent; defaults to None.
            my_current_v_per_step (float, optional): The current speed of the agent; defaults to None.
            include_unparallel (bool, optional): Whether to include lanes regardless of yaw difference.
            selected_lanes (list, optional): A list of lanes to consider; other lanes are ignored.
            valid_lane_types (list, optional): A list of valid lane types to search within.
            excluded_lanes (list, optional): A list of lanes to be excluded from the search.

        Returns:
            tuple: A tuple containing the closest lane, the index of the closest point on that lane, and the distance to the lane.
        """
        # find a closest lane for a state
        closest_dist = 999999
        closest_dist_no_yaw = 999999
        closest_dist_threshold = 5
        closest_lane = None
        closest_lane_no_yaw = None
        closest_lane_pt_no_yaw_idx = None
        closest_lane_pt_idx = None

        current_lane = None
        current_closest_pt_idx = None
        dist_to_lane = None

        closest_lanes_same_dir = []
        closest_lanes_idx_same_dir = []

        for each_lane in current_state['road']:
            if each_lane in excluded_lanes:
                continue
            if len(selected_lanes) > 0 and each_lane not in selected_lanes:
                continue
            if isinstance(current_state['road'][each_lane]['type'], int):
                if current_state['road'][each_lane]['type'] not in valid_lane_types:
                    continue
            else:
                if current_state['road'][each_lane]['type'][0] not in valid_lane_types:
                    continue
            road_xy = current_state['road'][each_lane]['xyz'][:, :2]
            if road_xy.shape[0] < 3:
                continue
            current_lane_closest_dist = 999999
            current_lane_closest_idx = None

            for j, each_xy in enumerate(road_xy):
                road_yaw = current_state['road'][each_lane]['dir'][j]
                dist = utils.euclidean_distance(each_xy, my_current_pose[:2])
                yaw_diff = abs(utils.normalize_angle(my_current_pose[3] - road_yaw))
                if dist < closest_dist_no_yaw:
                    closest_lane_no_yaw = each_lane
                    closest_dist_no_yaw = dist
                    closest_lane_pt_no_yaw_idx = j
                if yaw_diff < math.pi / 180 * 20 and dist < closest_dist_threshold:
                    if dist < closest_dist:
                        closest_lane = each_lane
                        closest_dist = dist
                        closest_lane_pt_idx = j
                    if dist < current_lane_closest_dist:
                        current_lane_closest_dist = dist
                        current_lane_closest_idx = j

            # classify current agent as a lane changer or not:
            if my_current_v_per_step != None and my_current_v_per_step > 0.1 and 0.5 < current_lane_closest_dist < 3.2 and each_lane not in closest_lanes_same_dir and current_state['road'][each_lane]['turning'] == 0:
                    closest_lanes_same_dir.append(each_lane)
                    closest_lanes_idx_same_dir.append(current_lane_closest_idx)

        if closest_lane is not None:
            current_lane = closest_lane
            current_closest_pt_idx = closest_lane_pt_idx
            dist_to_lane = closest_dist
            # distance_threshold = max(7, max(7 * my_current_v_per_step, dist_to_lane))
        elif closest_lane_no_yaw is not None and include_unparallel:
            current_lane = closest_lane_no_yaw
            current_closest_pt_idx = closest_lane_pt_no_yaw_idx
            dist_to_lane = closest_dist_no_yaw
            # distance_threshold = max(10, dist_to_lane)
        else:
            logging.warning(f'No current lane founded: {agent_id}')
            # return
        return current_lane, current_closest_pt_idx, dist_to_lane


    def get_reroute_traj(self, current_state, agent_id, current_frame_idx,
                         follow_org_route=False, dynamic_turnings=True, current_route=[], is_ego=False):
        """
        This function generates a marginal planned trajectory using a simple lane follower algorithm.

        Args:
            current_state (dict): The current state of the simulation environment.
            agent_id (int): The unique identifier for the agent.
            current_frame_idx (int): The index of the current frame.
            follow_org_route (bool): Flag to follow the original route.
            dynamic_turnings (bool): Flag to consider dynamic turnings in the route.
            current_route (list): The current route being used, specified as a list of lane IDs.
            is_ego (bool): Flag to indicate if the agent is the ego (main) agent.

        Returns:
            tuple: A tuple containing the generated trajectory and the route used.
        """
        assert self.routed_traj is not None, self.routed_traj
        # generate a trajectory based on the route
        # 1. get the route for relevant agents
        # find the closest lane to trace
        my_current_pose, my_current_v_per_step = plan_helper.get_current_pose_and_v(current_state=current_state,
                                                                                    agent_id=agent_id,
                                                                                    current_frame_idx=current_frame_idx)
        my_current_v_per_step = np.clip(my_current_v_per_step, a_min=0, a_max=7)
        goal_pt, goal_yaw = self.online_predictor.get_goal(current_data=current_state,
                                                                       agent_id=agent_id,
                                                                       dataset=self.dataset)
        if PRINT_TIMER:
            last_tic = time.perf_counter()
        if agent_id not in self.past_lanes:
            self.past_lanes[agent_id] = []
        if self.dataset == 'NuPlan' and is_ego:
            goal_lane, _, _ = self.find_closest_lane(
                current_state=current_state,
                my_current_pose=[goal_pt[0], goal_pt[1], -1, goal_yaw],
                valid_lane_types=self.valid_lane_types,
            )
            # current_route is a list of multiple routes to choose
            if len(current_route) == 0:
                lanes_in_route = []
                route_roadblocks = current_state['route'] if 'route' in current_state else None
                for each_block in route_roadblocks:
                    if each_block not in current_state['road']:
                        continue
                    lanes_in_route += current_state['road'][each_block]['lower_level']
                current_lanes, current_closest_pt_indices, dist_to_lane = self.find_closest_lane(
                    current_state=current_state,
                    my_current_pose=my_current_pose,
                    selected_lanes=lanes_in_route,
                    valid_lane_types=self.valid_lane_types,
                    excluded_lanes=self.past_lanes[agent_id]
                )
            else:
                selected_lanes = []
                for each_route in current_route:
                    selected_lanes += each_route
                current_lanes, current_closest_pt_indices, dist_to_lane = self.find_closest_lane(
                    current_state=current_state,
                    my_current_pose=my_current_pose,
                    selected_lanes=selected_lanes,
                    valid_lane_types=self.valid_lane_types,
                    excluded_lanes=self.past_lanes[agent_id]
                )
        else:
            if len(current_route) > 0:
                current_route = current_route[0]
            current_lanes, current_closest_pt_indices, dist_to_lane = self.find_closest_lane(
                current_state=current_state,
                my_current_pose=my_current_pose,
                selected_lanes=current_route,
                valid_lane_types=self.valid_lane_types,
                excluded_lanes=self.past_lanes[agent_id]
            )
        if dist_to_lane is not None:
            distance_threshold = max(self.frame_rate, max(self.frame_rate * my_current_v_per_step, dist_to_lane))
        else:
            dist_to_lane = 999
        self.current_on_road = not (dist_to_lane > OFF_ROAD_DIST)
        if self.dataset == 'NuPlan' and len(current_route) == 0 and is_ego:
            pass
        else:
            if current_lanes in current_route and not isinstance(current_lanes, list):
                for each_past_lane in current_route[:current_route.index(current_lanes)]:
                    if each_past_lane not in self.past_lanes[agent_id]:
                        self.past_lanes[agent_id].append(each_past_lane)

        if isinstance(current_lanes, list):
            # deprecated
            lane_found_in_route = False
            for each_lane in current_lanes:
                if each_lane in current_route:
                    current_lane = each_lane
                    lane_found_in_route = True
                    break
            if not lane_found_in_route:
                current_lane = random.choice(current_lanes)
            idx = current_lanes.index(current_lane)
            current_closest_pt_idx = current_closest_pt_indices[idx]
        else:
            current_lane = current_lanes
            current_closest_pt_idx = current_closest_pt_indices

        if PRINT_TIMER:
            print(f"Time spent on first lane search:  {time.perf_counter() - last_tic:04f}s")
            last_tic = time.perf_counter()

        if self.dataset == 'NuPlan' and is_ego:
            # use route_roadblocks
            prior_lanes = []
            if current_lane is None:
                print("WARNING: Ego Current Lane not found")
        elif len(current_route) == 0:
            # get route from the original trajectory, this route does not have to be neither accurate nor connected
            prior_lanes = []
            org_closest_pt_idx = []
            for i in range(50):
                if i + current_frame_idx > 90:
                    break
                if i == 0:
                    continue
                if i % 10 != 0:
                    continue
                looping_pose, looping_v = plan_helper.get_current_pose_and_v(current_state=current_state,
                                                                 agent_id=agent_id,
                                                                 current_frame_idx=current_frame_idx + i)

                # looping_lane, looping_closest_idx, _, _ = self.find_closes_lane(current_state=current_state,
                #                                                                 agent_id=agent_id,
                #                                                                 my_current_v_per_step=looping_v,
                #                                                                 my_current_pose=looping_pose,
                #                                                                 no_unparallel=follow_org_route,
                #                                                                 return_list=False)

                looping_lane, looping_closest_idx, dist_to_lane = self.find_closest_lane(
                    current_state=current_state,
                    my_current_pose=looping_pose,
                    # include_unparallel=not follow_org_route
                    include_unparallel=False,
                    valid_lane_types=self.valid_lane_types,
                    excluded_lanes=self.past_lanes[agent_id]
                )

                if looping_lane is not None and looping_lane not in prior_lanes and dist_to_lane < 5:
                    prior_lanes.append(looping_lane)
                    org_closest_pt_idx.append(looping_closest_idx)

            if PRINT_TIMER:
                print(f"Time spent on loop lane search:  {time.perf_counter() - last_tic:04f}s")
                last_tic = time.perf_counter()
        else:
            prior_lanes = current_route

        # 2. find a spot to enter
        # Make connection with BC
        accum_dist = -0.0001
        p4 = None
        cuttin_lane_id = None
        cuttin_lane_idx = None
        first_lane = True

        def search_lanes(current_lane, route_roadblocks):
            result_lanes = []

            if goal_lane not in self.past_lanes['ego']:
                goal_roadblock = current_state['road'][goal_lane]['upper_level'][0]
                current_roadblock = current_state['road'][current_lane]['upper_level'][0]
                if goal_roadblock == current_roadblock:
                    current_lane = goal_lane

            lanes_to_loop = [[current_lane]]
            visited_lanes = [current_lane]

            while len(lanes_to_loop) > 0:
                looping_lanes = lanes_to_loop.pop()
                if len(looping_lanes) >= 3:
                    result_lanes.append(looping_lanes)
                    continue
                looping_lane = looping_lanes[-1]
                looping_roadblock = current_state['road'][looping_lane]['upper_level'][0]
                if looping_roadblock not in route_roadblocks:
                    continue
                # no lane changing
                # all_lanes_in_block = current_state['road'][looping_roadblock]['lower_level']
                # for each_lane in all_lanes_in_block:
                #     if each_lane not in visited_lanes:
                #         visited_lanes.append(each_lane)
                #         lanes_to_loop.append(looping_lanes[:-1]+[each_lane])
                next_lanes = current_state['road'][looping_lane]['next_lanes']
                for each_lane in next_lanes:
                    if each_lane not in visited_lanes:
                        visited_lanes.append(each_lane)
                        if each_lane not in current_state['road']:
                            result_lanes.append(looping_lanes)
                            continue
                        each_block = current_state['road'][each_lane]['upper_level'][0]
                        if each_block not in route_roadblocks:
                            continue
                        lanes_to_loop.append(looping_lanes+[each_lane])
                if len(lanes_to_loop) == 0 and len(looping_lanes) > 0:
                    result_lanes.append(looping_lanes)
            return result_lanes

        if self.dataset == 'NuPlan' and is_ego and current_lane is not None:
            route_roadblocks = current_state['route'] if 'route' in current_state else None
            current_upper_roadblock = current_state['road'][current_lane]['upper_level'][0]
            if current_upper_roadblock not in route_roadblocks:
                route_roadblocks.insert(0, current_upper_roadblock)
            while len(route_roadblocks) < 3 and route_roadblocks[-1] in current_state['road']:
                next_roadblocks = current_state['road'][route_roadblocks[-1]]['next_lanes']
                if len(next_roadblocks) == 0 or next_roadblocks[0] not in current_state['road']:
                    break
                route_roadblocks.append(current_state['road'][route_roadblocks[-1]]['next_lanes'][0])
            # assumption: not far from current lane
            result_lanes = search_lanes(current_lane, route_roadblocks)

            if len(result_lanes) == 0:
                # choose a random lane from the first roadblock
                print("WARNING: No available route found")
                assert False, 'No Available Route Found for ego'

            result_traj = []
            for each_route in result_lanes:
                current_trajectory = None
                reference_trajectory = None
                reference_yaw = None
                for each_lane in each_route:
                    if each_lane not in current_state['road']:
                        break
                    if reference_trajectory is None:
                        reference_trajectory = current_state['road'][each_lane]['xyz'][current_closest_pt_idx:, :2].copy()
                        reference_yaw = current_state['road'][each_lane]['dir'][current_closest_pt_idx:].copy()
                    else:
                        reference_trajectory = np.concatenate((reference_trajectory,
                                                               current_state['road'][each_lane]['xyz'][:, :2].copy()))
                        reference_yaw = np.concatenate((reference_yaw,
                                                        current_state['road'][each_lane]['dir'].copy()))
                # get CBC
                if reference_trajectory.shape[0] < 2:
                    p1 = my_current_pose[:2]
                    yaw = - utils.normalize_angle(my_current_pose[3] + math.pi / 2)
                    delta = self.planning_horizon
                    x, y = -math.sin(yaw) * delta + my_current_pose[0], -math.cos(yaw) * delta + \
                           my_current_pose[1]
                    p2 = [x, y]
                    p3 = p2
                    x, y = -math.sin(yaw) * delta + p2[0], -math.cos(yaw) * delta + p2[1]
                    p4 = [x, y]
                    # 4. generate a curve with cubic BC
                    if my_current_v_per_step < 1:
                        proper_v_for_cbc = (my_current_v_per_step + 1) / 2
                    else:
                        proper_v_for_cbc = my_current_v_per_step
                    if utils.euclidean_distance(p4, p1) > 1:
                        print(f"No lanes found for route of {agent_id} {proper_v_for_cbc} {my_current_pose}")
                        connection_traj = self.trajectory_from_cubic_BC(p1=p1, p2=p2, p3=p3, p4=p4, v=proper_v_for_cbc)
                    else:
                        assert False, f"Error: P4, P1 overlapping {p4, p1}"
                    assert connection_traj.shape[0] > 0, connection_traj.shape
                    result_traj.append(connection_traj)
                    current_state['predicting']['trajectory_to_mark'].append(current_trajectory)
                else:
                    starting_index = int(my_current_v_per_step * self.frame_rate * 2)
                    starting_index = min(starting_index, reference_trajectory.shape[0] - 1)
                    p4 = reference_trajectory[starting_index, :2]
                    starting_yaw = -utils.normalize_angle(reference_yaw[starting_index] + math.pi / 2)
                    delta = utils.euclidean_distance(p4, my_current_pose[:2]) / 4
                    x, y = math.sin(starting_yaw) * delta + p4[0], math.cos(starting_yaw) * delta + p4[1]
                    p3 = [x, y]

                    p1 = my_current_pose[:2]
                    yaw = - utils.normalize_angle(my_current_pose[3] + math.pi / 2)
                    delta = min(70/self.frame_rate, utils.euclidean_distance(p4, my_current_pose[:2]) / 2)
                    x, y = -math.sin(yaw) * delta + my_current_pose[0], -math.cos(yaw) * delta + my_current_pose[1]
                    p2 = [x, y]
                    if utils.euclidean_distance(p4, p1) > 2:
                        if my_current_v_per_step < 1:
                            proper_v_for_cbc = (my_current_v_per_step + 1) / 2
                        else:
                            proper_v_for_cbc = my_current_v_per_step

                        connection_traj = self.trajectory_from_cubic_BC(p1=p1, p2=p2, p3=p3, p4=p4, v=proper_v_for_cbc)
                        current_trajectory = np.concatenate((connection_traj, reference_trajectory[starting_index:, :2]))
                    else:
                        current_trajectory = reference_trajectory[starting_index:, :2]
                    result_traj.append(current_trajectory)
                    current_state['predicting']['trajectory_to_mark'].append(current_trajectory)

            assert len(result_traj) == len(result_lanes), f'unmatched shape {len(result_traj)} {len(result_lanes)}'
            self.routed_traj[agent_id] = result_traj
            return self.routed_traj[agent_id], result_lanes

        if current_lane is not None:
            current_looping_lane = current_lane
            while_counter = 0
            if distance_threshold > 100:
                print("Closest lane detection failded: ", agent_id, current_looping_lane, distance_threshold, my_current_v_per_step, dist_to_lane, current_route)
            else:
                distance_threshold = max(distance_threshold, self.frame_rate * my_current_v_per_step)

                while accum_dist < distance_threshold and distance_threshold <= 100:
                    if while_counter > 100:
                        print("ERROR: Infinite looping lanes")
                        break

                    while_counter += 1
                    # turning: 1=left turn, 2=right turn, 3=UTurn
                    # UTurn -> Skip
                    # Left/Right check distance, if < 15 then skip, else not skip
                    if current_looping_lane not in current_state['road']:
                        break
                    current_looping_lane_turning = current_state['road'][current_looping_lane]['turning']
                    if dynamic_turnings and current_looping_lane_turning == 3 or (current_looping_lane_turning in [1, 2] and utils.euclidean_distance(current_state['road'][current_looping_lane]['xyz'][-1, :2], my_current_pose[:2]) < 15):
                        # skip turning lanes
                        # accum_dist = distance_threshold - 0.1
                        pass
                    elif while_counter > 50:
                        print("Inifinite looping lanes (agent_id/current_lane): ", agent_id, current_looping_lane)
                        accum_dist = distance_threshold - 0.1
                    else:
                        if first_lane:
                            road_xy = current_state['road'][current_looping_lane]['xyz'][current_closest_pt_idx:, :2].copy()
                        else:
                            road_xy = current_state['road'][current_looping_lane]['xyz'][:, :2].copy()
                        for j, each_xy in enumerate(road_xy):
                            if j == 0:
                                continue
                            accum_dist += utils.euclidean_distance(each_xy, road_xy[j - 1])
                            if accum_dist >= distance_threshold:
                                p4 = each_xy
                                if first_lane:
                                    yaw = - utils.normalize_angle(
                                        current_state['road'][current_looping_lane]['dir'][j + current_closest_pt_idx] + math.pi / 2)
                                else:
                                    yaw = - utils.normalize_angle(
                                        current_state['road'][current_looping_lane]['dir'][j] + math.pi / 2)
                                delta = utils.euclidean_distance(p4, my_current_pose[:2]) / 4
                                x, y = math.sin(yaw) * delta + p4[0], math.cos(yaw) * delta + p4[1]
                                p3 = [x, y]
                                cuttin_lane_id = current_looping_lane
                                if first_lane:
                                    cuttin_lane_idx = j + current_closest_pt_idx
                                else:
                                    cuttin_lane_idx = j
                                break

                    if p4 is None:
                        if current_looping_lane in prior_lanes and current_looping_lane != prior_lanes[-1]:
                            # if already has route, then use previous route
                            current_lane_route_idx = prior_lanes.index(current_looping_lane)
                            current_looping_lane = prior_lanes[current_lane_route_idx+1]
                        else:
                            # if not, try to loop a new route
                            next_lanes = current_state['road'][current_looping_lane]['next_lanes']
                            next_lane_found = False
                            if follow_org_route:
                                if current_looping_lane in prior_lanes:  # True:
                                    # follow original lanes
                                    current_idx = prior_lanes.index(current_looping_lane)
                                    if current_idx < len(prior_lanes) - 1:
                                        next_lane = prior_lanes[current_idx + 1]
                                        next_lane_found = True
                                        if next_lane in next_lanes:
                                            # next lane connected, loop this next lane and continue next loop
                                            current_looping_lane = next_lane
                                        else:
                                            # next lane not connected
                                            # 1. find closest point
                                            road_xy = current_state['road'][current_looping_lane]['xyz'][:, :2].copy()
                                            closest_dist = 999999
                                            closest_lane_idx = None
                                            turning_yaw = None
                                            for j, each_xy in enumerate(road_xy):
                                                dist = utils.euclidean_distance(each_xy[:2], my_current_pose[:2])
                                                if dist < closest_dist:
                                                    closest_lane_idx = j
                                                    closest_dist = dist
                                                    turning_yaw = utils.normalize_angle(my_current_pose[3] - current_state['road'][current_looping_lane]['dir'][j])
                                            if closest_lane_idx is None:
                                                # follow no next lane logic below
                                                next_lane_found = False
                                            else:
                                                max_turning_dist = 120 / math.pi
                                                if closest_dist >= max_turning_dist:
                                                    # too far for max turning speed 15m/s
                                                    if turning_yaw > math.pi / 2:
                                                        # turn towards target lane first on the right
                                                        yaw = - utils.normalize_angle(my_current_pose[3] + math.pi / 2) + math / 2
                                                        delta = 180 / math.pi
                                                        x, y = math.sin(yaw) * delta + my_current_pose[0], math.cos(yaw) * delta + my_current_pose[1]
                                                        p4 = [x, y]
                                                        yaw = yaw - math / 2
                                                        delta = delta / 2
                                                        x, y = math.sin(yaw) * delta + my_current_pose[0], math.cos(yaw) * delta + my_current_pose[1]
                                                        p3 = [x, y]
                                                        break
                                                    if turning_yaw <= math.pi / 2:
                                                        # turn towards target lane first on the right
                                                        yaw = - utils.normalize_angle(my_current_pose[3] + math.pi / 2) - math / 2
                                                        delta = 180 / math.pi
                                                        x, y = math.sin(yaw) * delta + my_current_pose[0], math.cos(yaw) * delta + my_current_pose[1]
                                                        p4 = [x, y]
                                                        yaw = yaw + math / 2
                                                        delta = delta / 2
                                                        x, y = math.sin(yaw) * delta + my_current_pose[0], math.cos(yaw) * delta + my_current_pose[1]
                                                        p3 = [x, y]
                                                        break
                                                else:
                                                    accum_dist = distance_threshold - 0.1

                            if not next_lane_found:
                                # follow prior or choose a random one as the next
                                if len(next_lanes) > 0:
                                    current_looping_lane_changes = False
                                    for each_lane in next_lanes:
                                        if each_lane in prior_lanes:
                                            current_looping_lane = each_lane
                                            current_looping_lane_changes = True
                                    if not current_looping_lane_changes:
                                        # random choose one lane as route
                                        current_looping_lane = random.choice(next_lanes)
                                else:
                                    print("warning: no next lane found with breaking the lane finding loop")
                                    break
                                    # return
                    else:
                        break
                    first_lane = False

        if PRINT_TIMER:
            print(f"Time spent on while loop:  {time.perf_counter() - last_tic:04f}s")
            last_tic = time.perf_counter()

        if p4 is None:
            # not found any lane at all, generate a linear line forward
            # 3. gennerate p1 and p2
            p1 = my_current_pose[:2]
            yaw = - utils.normalize_angle(my_current_pose[3] + math.pi / 2)
            delta = self.planning_horizon
            x, y = -math.sin(yaw) * delta + my_current_pose[0], -math.cos(yaw) * delta + \
                   my_current_pose[1]
            p2 = [x, y]
            p3 = p2
            x, y = -math.sin(yaw) * delta + p2[0], -math.cos(yaw) * delta + p2[1]
            p4 = [x, y]
            # 4. generate a curve with cubic BC
            if my_current_v_per_step < 1:
                proper_v_for_cbc = (my_current_v_per_step + 1) / 2
            else:
                proper_v_for_cbc = my_current_v_per_step
            if utils.euclidean_distance(p4, p1) > 1:
                print(f"No lanes found for route of {agent_id} {proper_v_for_cbc} {my_current_pose}")
                connection_traj = self.trajectory_from_cubic_BC(p1=p1, p2=p2, p3=p3, p4=p4, v=proper_v_for_cbc)
            else:
                assert False, f"Error: P4, P1 overlapping {p4, p1}"
            assert connection_traj.shape[0] > 0, connection_traj.shape
            self.routed_traj[agent_id] = connection_traj
        else:
            assert cuttin_lane_id is not None
            # 3. gennerate p1 and p2
            p1 = my_current_pose[:2]
            yaw = - utils.normalize_angle(my_current_pose[3] + math.pi / 2)
            delta = min(7, utils.euclidean_distance(p4, my_current_pose[:2]) / 2)
            x, y = -math.sin(yaw) * delta + my_current_pose[0], -math.cos(yaw) * delta + \
                   my_current_pose[1]
            p2 = [x, y]

            if my_current_v_per_step < 1:
                proper_v_for_cbc = (my_current_v_per_step + 1) / 2
            else:
                proper_v_for_cbc = my_current_v_per_step

            connection_traj = self.trajectory_from_cubic_BC(p1=p1, p2=p2, p3=p3, p4=p4, v=proper_v_for_cbc)
            # loop out a route
            current_looping_lane = cuttin_lane_id
            lanes_in_a_route = [current_looping_lane]
            route_traj_left = np.array(current_state['road'][current_looping_lane]['xyz'][cuttin_lane_idx:, :2], ndmin=2)
            next_lanes = current_state['road'][current_looping_lane]['next_lanes']
            while len(next_lanes) > 0 and len(lanes_in_a_route) < 10:
                any_lane_in_route = False
                if len(prior_lanes) > 0:
                    for each_next_lane in next_lanes:
                        if each_next_lane in prior_lanes:
                            any_lane_in_route = True
                            current_looping_lane = each_next_lane
                            break
                if not any_lane_in_route:
                    # try to follow original route
                    current_lane_changed = False
                    lanes_to_choose = []
                    for each_next_lane in next_lanes:
                        if each_next_lane in prior_lanes:
                            current_looping_lane = each_next_lane
                            current_lane_changed = True
                            break
                        if each_next_lane in current_state['road']:
                            lanes_to_choose.append(each_next_lane)
                    if current_lane_changed:
                        pass
                    elif len(lanes_to_choose) == 0:
                        print("NO VALID NEXT LANE TO CHOOSE from env_planner for ", agent_id)
                        break
                    else:
                        # random choose one lane as route
                        current_looping_lane = random.choice(lanes_to_choose)

                # amend route manually for scenario 54 file 00000
                # if current_looping_lane == 109:
                #     current_looping_lane = 112
                # if current_looping_lane == 131:
                #     current_looping_lane = 132
                if current_looping_lane not in current_state['road']:
                    print("selected lane not found in road dic")
                    break
                lanes_in_a_route.append(current_looping_lane)
                next_lanes = current_state['road'][current_looping_lane]['next_lanes']
                # route_traj_left = np.concatenate(
                #     (route_traj_left, current_state['road'][current_looping_lane]['xyz'][:, :2]))
                route_traj_left = np.concatenate(
                    (route_traj_left, current_state['road'][current_looping_lane]['xyz'][10:, :2]))  # start with a margin to avoid overlapping ends and starts
            if len(current_route) == 0:
                # initiation the route and return
                current_route = lanes_in_a_route
                if is_ego:
                    goal_pt, goal_yaw = self.online_predictor.get_goal(current_data=current_state,
                                                                                   agent_id=agent_id,
                                                                                   dataset=self.dataset)
                    assert goal_pt is not None and goal_yaw is not None, goal_pt
                    ending_lane, ending_lane_idx, dist_to_ending_lane = self.find_closest_lane(
                        current_state=current_state,
                        my_current_pose=[goal_pt[0], goal_pt[1], 0, goal_yaw],
                        valid_lane_types=self.valid_lane_types
                    )

                    if ending_lane is not None:
                        if dist_to_ending_lane > 30:
                            logging.warning('Goal Point Off Road')
                        self.target_lanes = [ending_lane, ending_lane_idx]

                        if ending_lane not in lanes_in_a_route:
                            back_looping_counter = 0
                            back_to_loop_lanes = [ending_lane]
                            target_lane = ending_lane
                            while back_looping_counter < 10:
                                back_looping_counter += 1
                                current_back_looping_lane = back_to_loop_lanes.pop()
                                _, _, distance_to_ending_lane = self.find_closest_lane(
                                    current_state=current_state,
                                    my_current_pose=my_current_pose,
                                    selected_lanes=[current_back_looping_lane],
                                    valid_lane_types=self.valid_lane_types
                                )
                                if distance_to_ending_lane < OFF_ROAD_DIST:
                                    target_lane = current_back_looping_lane
                                    break
                                else:
                                    if current_back_looping_lane not in current_state['road']:
                                        break
                                    prev_lanes = current_state['road'][current_back_looping_lane]['previous_lanes']
                                    if not isinstance(prev_lanes, list):
                                        prev_lanes = prev_lanes.tolist()
                                    if len(prev_lanes) == 0:
                                        break
                                    back_to_loop_lanes += prev_lanes

                            current_route = [target_lane]
                    else:
                        logging.warning('No Lane Found for Goal Point at all')

            route_traj_left = np.array(route_traj_left, ndmin=2)
            # 4. generate a curve with cubic BC
            if utils.euclidean_distance(p4, p1) > 2:
                if len(route_traj_left.shape) < 2:
                    print(route_traj_left.shape, route_traj_left)
                    self.routed_traj[agent_id] = connection_traj
                else:
                    if utils.euclidean_distance(p4, p1) > 1 and len(connection_traj.shape) > 0 and connection_traj.shape[0] > 1:
                        # concatenate org_traj, connection_traj, route_traj_left
                        self.routed_traj[agent_id] = np.concatenate(
                            (connection_traj, route_traj_left))
                    else:
                        self.routed_traj[agent_id] = route_traj_left
            else:
                self.routed_traj[agent_id] = route_traj_left

        if PRINT_TIMER:
            print(f"Time spent on CBC:  {time.perf_counter() - last_tic:04f}s")
            last_tic = time.perf_counter()
        if DRAW_CBC_PTS:
            current_state['predicting']['mark_pts'] = [p4, p3, p2, p1]
        if is_ego:
            if self.dataset == 'NuPlan':
                return [self.routed_traj[agent_id]], current_route
            else:
                return [self.routed_traj[agent_id]], [current_route]
        else:
            return self.routed_traj[agent_id], current_route

    def adjust_speed_for_collision(self, interpolator, distance_to_end, current_v, end_point_v, reschedule_speed_profile=False):
        """
        Adjusts the agent's speed to avoid a potential collision by applying constant deceleration.

        Args:
            interpolator: The interpolator instance used for generating trajectory points.
            distance_to_end (float): The distance to the end of the trajectory or collision point.
            current_v (float): The current speed of the agent.
            end_point_v (float): The desired speed at the collision point.
            reschedule_speed_profile (bool, optional): Flag to reschedule the speed profile; defaults to False.

        Returns:
            np.array: A numpy array containing the adjusted trajectory points.
        """
        # constant deceleration
        time_to_collision = min(self.planning_horizon, distance_to_end / (current_v + end_point_v + 0.0001) * 2)
        time_to_decelerate = abs(current_v - end_point_v) / (0.1/self.frame_rate)
        traj_to_return = []
        desired_deceleration = 0.2 /self.frame_rate
        if time_to_collision < time_to_decelerate:
            # decelerate more than 3m/ss
            deceleration = (end_point_v - current_v) / time_to_collision
            dist_travelled = 0
            for i in range(int(time_to_collision)):
                current_v += deceleration * 1.2
                current_v = max(0, current_v)
                dist_travelled += current_v
                traj_to_return.append(interpolator.interpolate(dist_travelled))
            current_len = len(traj_to_return)
            while current_len < 100:
                dist_travelled += current_v
                traj_to_return.append(interpolator.interpolate(dist_travelled))
                current_len = len(traj_to_return)
        else:
            # decelerate with 2.5m/ss
            time_for_current_speed = np.clip(((distance_to_end - 3 - (current_v+end_point_v)/2*time_to_decelerate) / (current_v + 0.0001)), 0, self.frame_rate*self.frame_rate)
            dist_travelled = 0
            if time_for_current_speed > 1:
                for i in range(int(time_for_current_speed)):
                    if reschedule_speed_profile:
                        dist_travelled += current_v
                    else:
                        if i == 0:
                            dist_travelled += current_v
                        elif i >= interpolator.trajectory.shape[0]:
                            dist_travelled += current_v
                        else:
                            current_v_hat = interpolator.get_speed_with_index(i)
                            if abs(current_v_hat - current_v) > 2 / self.frame_rate:
                                print("WARNING: sharp speed changing", current_v, current_v_hat)
                            current_v = current_v_hat
                            dist_travelled += current_v
                    traj_to_return.append(interpolator.interpolate(dist_travelled))
            for i in range(int(time_to_decelerate)):
                current_v -= desired_deceleration
                current_v = max(0, current_v)
                dist_travelled += current_v
                traj_to_return.append(interpolator.interpolate(dist_travelled))
            current_len = len(traj_to_return)
            while current_len < 100:
                dist_travelled += current_v
                traj_to_return.append(interpolator.interpolate(dist_travelled))
                current_len = len(traj_to_return)
        if len(traj_to_return) > 0:
            short = self.planning_horizon - len(traj_to_return)
            for _ in range(short):
                traj_to_return.append(traj_to_return[-1])
        else:
            for _ in range(self.planning_horizon):
                traj_to_return.append(interpolator.interpolate(0))
        return np.array(traj_to_return, ndmin=2)

    def get_traffic_light_collision_pts(self, current_state, current_frame_idx,
                                        continue_time_threshold=5):
        """
        Identifies collision points at traffic lights based on the current state and frame index.

        Args:
            current_state (dict): The current state of the traffic environment.
            current_frame_idx (int): The index of the current frame.
            continue_time_threshold (int, optional): The time threshold for continuous traffic light states.

        Returns:
            list: A list of end points for traffic light collision checks, useful for planning routes.
        """
        tl_dics = current_state['traffic_light']
        road_dics = current_state['road']
        traffic_light_ending_pts = []
        for lane_id in tl_dics.keys():
            if lane_id == -1:
                continue
            tl = tl_dics[lane_id]
            # get the position of the end of this lane
            # Unknown = 0, Arrow_Stop = 1, Arrow_Caution = 2, Arrow_Go = 3, Stop = 4, Caution = 5, Go = 6, Flashing_Stop = 7, Flashing_Caution = 8
            try:
                tl_state = tl["state"][current_frame_idx]
            except:
                tl_state = tl["state"][0]

            if tl_state in [1, 4, 7]:
                end_of_tf_checking = min(len(tl["state"]), current_frame_idx + continue_time_threshold)
                all_red = True
                for k in range(current_frame_idx, end_of_tf_checking):
                    if tl["state"][k] not in [1, 4, 7]:
                        all_red = False
                        break
                if all_red:
                    for seg_id in road_dics.keys():
                        if lane_id == seg_id:
                            road_seg = road_dics[seg_id]
                            if self.dataset == 'Waymo':
                                if road_seg["type"] in [1, 2, 3]:
                                    if len(road_seg["dir"].shape) < 1:
                                        continue
                                    if road_seg['turning'] == 1 and tl_state in [4, 7]:
                                        # can do right turn with red light
                                        continue
                                    end_point = road_seg["xyz"][0][:2]
                                    traffic_light_ending_pts.append(end_point)
                                break
                            elif self.dataset == 'NuPlan':
                                end_point = road_seg["xyz"][0][:2]
                                traffic_light_ending_pts.append(end_point)
                                break
                            else:
                                assert False, f'Unknown dataset in env planner - {self.dataset}'
        return traffic_light_ending_pts

    def get_trajectory_from_interpolator(self, my_interpolator, my_current_speed, a_per_step=None,
                                         check_turning_dynamics=True, desired_speed=7,
                                         emergency_stop=False, hold_still=False,
                                         agent_id=None, a_scale_turning=0.7, a_scale_not_turning=0.9):
        """
        Generates a trajectory based on the interpolator and dynamic speed adjustments for turns.

        Args:
            my_interpolator: The interpolator instance for trajectory generation.
            my_current_speed (float): The current speed of the agent.
            a_per_step (float, optional): The acceleration or deceleration step value.
            check_turning_dynamics (bool, optional): Flag to check for turning dynamics.
            desired_speed (float, optional): The desired speed for the agent.
            emergency_stop (bool, optional): Flag to indicate an emergency stop situation.
            hold_still (bool, optional): Flag to hold the agent in place.
            agent_id (int, optional): The identifier for the agent, used for specific checks.
            a_scale_turning (float, optional): The acceleration scale factor for turning scenarios.
            a_scale_not_turning (float, optional): The acceleration scale factor for non-turning scenarios.

        Returns:
            np.array: A numpy array containing the generated trajectory.
        """
        total_frames = self.planning_horizon
        total_pts_in_interpolator = my_interpolator.trajectory.shape[0]
        trajectory = np.ones((total_frames, 4)) * -1
        # get proper speed for turning
        largest_yaw_change = -1
        largest_yaw_change_idx = None
        if check_turning_dynamics and not emergency_stop:
            for i in range(min(200, total_pts_in_interpolator - 2)):
                if my_interpolator.trajectory[i, 0] == -1.0 or my_interpolator.trajectory[i+1, 0] == -1.0 or my_interpolator.trajectory[i+2, 0] == -1.0:
                    continue
                current_yaw = utils.normalize_angle(plan_helper.get_angle_of_a_line(pt1=my_interpolator.trajectory[i, :2], pt2=my_interpolator.trajectory[i+1, :2]))
                next_yaw = utils.normalize_angle(plan_helper.get_angle_of_a_line(pt1=my_interpolator.trajectory[i+1, :2], pt2=my_interpolator.trajectory[i+2, :2]))
                dist = utils.euclidean_distance(pt1=my_interpolator.trajectory[i, :2], pt2=my_interpolator.trajectory[i+1, :2])
                yaw_diff = abs(utils.normalize_angle(next_yaw - current_yaw))
                if yaw_diff > largest_yaw_change and 0.04 < yaw_diff < math.pi / 2 * 0.9 and 100 > dist > 0.3:
                    largest_yaw_change = yaw_diff
                    largest_yaw_change_idx = i
            proper_speed_minimal = max(5, math.pi / 3 / largest_yaw_change)  # calculate based on 20m/s turning for 12s a whole round with a 10hz data in m/s
            proper_speed_minimal_per_frame = proper_speed_minimal / self.frame_rate
            if largest_yaw_change_idx is not None:
                deceleration_frames = max(0, largest_yaw_change_idx - abs(my_current_speed - proper_speed_minimal_per_frame) / (A_SLOWDOWN_DESIRE / self.frame_rate / self.frame_rate / 2))
            else:
                deceleration_frames = 99999
        if agent_id is not None:
            pass
        dist_past = 0
        current_speed = my_current_speed
        for i in range(total_frames):
            if current_speed < 0.1:
                low_speed_a_scale = 1 * self.frame_rate
            else:
                low_speed_a_scale = 0.1 * self.frame_rate
            if hold_still:
                trajectory[i] = my_interpolator.interpolate(0)
                continue
            elif emergency_stop:
                current_speed -= A_SLOWDOWN_DESIRE / self.frame_rate
            elif largest_yaw_change_idx is not None:
                proper_speed_minimal_per_frame = max(0.5, min(proper_speed_minimal_per_frame, 5))
                if largest_yaw_change_idx >= i >= deceleration_frames:
                    if current_speed > proper_speed_minimal_per_frame:
                        current_speed -= A_SLOWDOWN_DESIRE / self.frame_rate / 2
                    else:
                        current_speed += A_SPEEDUP_DESIRE / self.frame_rate * a_scale_not_turning * low_speed_a_scale
                elif i < deceleration_frames:
                    if current_speed < desired_speed / 4.7:
                        # if far away from the turnings and current speed is smaller than 15m/s, then speed up
                        # else keep current speed
                        if a_per_step is not None:
                            current_speed += max(-A_SLOWDOWN_DESIRE / self.frame_rate, min(A_SPEEDUP_DESIRE / self.frame_rate * low_speed_a_scale, a_per_step))
                        else:
                            current_speed += A_SPEEDUP_DESIRE / self.frame_rate * a_scale_turning * low_speed_a_scale
                elif i > largest_yaw_change_idx:
                    if current_speed > proper_speed_minimal_per_frame:
                        current_speed -= A_SLOWDOWN_DESIRE / self.frame_rate
                    else:
                        if a_per_step is not None:
                            current_speed += max(-A_SLOWDOWN_DESIRE / self.frame_rate, min(A_SPEEDUP_DESIRE / self.frame_rate * low_speed_a_scale, a_per_step))
                        else:
                            current_speed += A_SPEEDUP_DESIRE / self.frame_rate * a_scale_turning * low_speed_a_scale
            else:
                if current_speed < desired_speed:
                    if a_per_step is not None:
                        current_speed += max(-A_SLOWDOWN_DESIRE / self.frame_rate, min(A_SPEEDUP_DESIRE / self.frame_rate * low_speed_a_scale, a_per_step))
                    else:
                        current_speed += A_SPEEDUP_DESIRE / self.frame_rate * a_scale_not_turning * low_speed_a_scale  # accelerate with 0.2 of desired acceleration
            current_speed = max(0, current_speed)
            dist_past += current_speed
            trajectory[i] = my_interpolator.interpolate(dist_past)
        return trajectory

    def update_env_trajectory_reguild(self, current_frame_idx, relevant_only=True,
                                      current_state=None, plan_for_ego=False, dynamic_env=True):
        """
        This function plans and updates trajectory to commit for relevant environment agents

        Args:
            current_frame_idx (int): The starting frame index for planning trajectories(1,2,3,...,11(first frame to plan)).
            relevant_only (bool, optional): When set to True, only plan for relevant agents.
            current_state (dict): The current state of the simulation environment.
            plan_for_ego (bool, optional): Flag indicating whether to plan for the ego vehicle.
            dynamic_env (bool, optional): Indicates if the environment is dynamic.

        Returns:
            dict: The updated current state with planned and adjusted trajectories for agents.
        """
        if not dynamic_env:
            return current_state

        if self.is_planning(current_frame_idx):
        # if frame_diff >= 0 and frame_diff % self.planning_interval == 0:
            # load scenario data
            if current_state is None:
                return
            agents = current_state['agent']
            relevant_agents = current_state['predicting']['relevant_agents']
            edges = current_state['predicting']['relation']
            ego_id = current_state['predicting']['ego_id'][1]
            agents_dic_copy = copy.deepcopy(current_state['agent'])

            for agent_id in agents:
                # loop each relevant agent
                if relevant_only and agent_id not in relevant_agents:
                    continue

                current_state['agent'][agent_id]['action'] = None
                total_time_frame = current_state['agent'][agent_id]['pose'].shape[0]
                goal_point = current_state['predicting']['goal_pts'][agent_id]
                my_current_pose = current_state['agent'][agent_id]['pose'][current_frame_idx - 1]
                my_current_v_per_step = utils.euclidean_distance(current_state['agent'][agent_id]['pose'][current_frame_idx - 1, :2],
                                                           current_state['agent'][agent_id]['pose'][current_frame_idx - 6, :2])/5
                my_target_speed = 70 / self.frame_rate

                if my_current_v_per_step > 100 / self.frame_rate:
                    my_current_v_per_step = 10 / self.frame_rate
                org_pose = current_state['predicting']['original_trajectory'][agent_id]['pose'].copy()

                # for non-vehicle types agent, skip
                if int(current_state['agent'][agent_id]['type']) not in self.vehicle_types:
                    continue

                # rst = prediction_traj_dic_m[agent_id]['rst']
                # score = np.exp(prediction_traj_dic_m[agent_id]['score'])
                # score /= np.sum(score)
                # best_idx = np.argmax(score)
                # prediction_traj_m = rst[best_idx]

                # use_rules = 0  # 0=hybird, 1=use rules only
                # info: always use rules for env agents
                use_rules = not self.follow_prediction_traj # use_rules=True
                if use_rules:
                    # past_goal = self.check_past_goal(traj=current_state['agent'][agent_id]['pose'],
                    #                                  current_idx=current_frame_idx,
                    #                                  current_state=current_state,
                    #                                  agent_id=agent_id)
                    my_traj, _ = self.get_reroute_traj(current_state=current_state,
                                                       agent_id=agent_id,
                                                       current_frame_idx=current_frame_idx)
                else:
                    routed_traj, _ = self.get_reroute_traj(current_state=current_state,
                                                           agent_id=agent_id,
                                                           current_frame_idx=current_frame_idx)
                    marginal_trajs = current_state['predicting']['marginal_trajectory'][agent_id]['rst'][0]
                    x_dist = []
                    for r_p in routed_traj[:50, :2]:
                        line_dist = []
                        for m_p in marginal_trajs[:50, :2]:
                            dist = utils.euclidean_distance(r_p, m_p)
                            line_dist.append(dist)
                        x_dist.append(min(line_dist))
                    minimal_distance = max(x_dist)
                    if True:
                    # if minimal_distance < 3:
                        my_traj = marginal_trajs
                    else:
                        my_traj = routed_traj

                my_interpolator = SudoInterpolator(my_traj.copy(), my_current_pose)
                interpolated_trajectory = self.get_trajectory_from_interpolator(my_interpolator=my_interpolator,
                                                                                my_current_speed=my_current_v_per_step,
                                                                                agent_id=agent_id)
                my_interpolator = SudoInterpolator(interpolated_trajectory.copy(), my_current_pose)

                earliest_collision_idx = None
                earliest_target_agent = None
                collision_point = None

                traffic_light_ending_pts = self.get_traffic_light_collision_pts(current_state=current_state,
                                                                                current_frame_idx=current_frame_idx)
                tl_checked = False
                running_red_light = False

                if self.method_testing < 1:
                    continue

                # check collisions for ego from frame 1 of the prediction trajectory
                ego_index_checking = 1  # current_frame_idx+1
                collision_detected_now = False
                latest_collision_id = None
                end_checking_frame = np.clip(current_frame_idx + REACTION_AFTER, 0, total_time_frame)
                end_checking_frame = min(end_checking_frame, current_frame_idx+self.planning_horizon)
                # pack an Agent object for collision detection
                my_reactors = []
                for i in range(current_frame_idx, end_checking_frame):
                    ego_index_checking = i - current_frame_idx
                    ego_pose2_valid = False
                    if i - current_frame_idx > 0:
                        ego_pose2 = interpolated_trajectory[ego_index_checking - 1]
                        if abs(ego_pose2[0]) < 1.1 and abs(ego_pose2[1]) < 1.1:
                            pass
                        else:
                            ego_agent2 =Agent(x=(ego_pose2[0] + ego_pose[0]) / 2,
                                               y=(ego_pose2[1] + ego_pose[1]) / 2,
                                               yaw=plan_helper.get_angle_of_a_line(ego_pose2[:2], ego_pose[:2]),
                                               length=utils.euclidean_distance(ego_pose2[:2], ego_pose[:2]),
                                               width=max(1, current_state['agent'][agent_id]['shape'][0][0]),
                                               agent_id=agent_id)
                            ego_pose2_valid = True

                    for each_other_agent in agents:
                        if each_other_agent == agent_id:
                            continue
                        if each_other_agent in my_reactors:
                            continue
                        if current_state['agent'][each_other_agent]['shape'][0][1] == -1:
                            continue
                        if ego_index_checking >= interpolated_trajectory.shape[0]:
                            continue
                        ego_pose = interpolated_trajectory[ego_index_checking, :]  # ego start checking from frame 0
                        if abs(ego_pose[0]) < 1.1 and abs(ego_pose[1]) < 1.1:
                            # print("WARNING invalid pose for collision detection: ", pose_in_pred)
                            continue
                        ego_agent =Agent(x=ego_pose[0],
                                          y=ego_pose[1],
                                          yaw=ego_pose[3],
                                          length=max(1, current_state['agent'][agent_id]['shape'][0][1]),
                                          width=max(1, current_state['agent'][agent_id]['shape'][0][0]),
                                          agent_id=agent_id)

                        # check traffic light violation
                        for tl_pt in traffic_light_ending_pts:
                            dummy_tf_agent = Agent(x=tl_pt[0], y=tl_pt[1], yaw=0,
                                                   length=TRAFFIC_LIGHT_COLLISION_SIZE,
                                                   width=TRAFFIC_LIGHT_COLLISION_SIZE, agent_id=99999)
                            running = utils.check_collision(
                                checking_agent=ego_agent,
                                target_agent=dummy_tf_agent)
                            if ego_pose2_valid:
                                running |= utils.check_collision(
                                    checking_agent=ego_agent2,
                                    target_agent=dummy_tf_agent)
                            if running:
                                running_red_light = True
                                earliest_collision_idx = ego_index_checking
                                collision_point = [ego_pose[0], ego_pose[1]]
                                earliest_target_agent = 99999
                                target_speed = 0
                                # break collision detection
                                break

                        if running_red_light:
                            to_yield = True
                            break

                        each_other_agent_pose_array = current_state['agent'][each_other_agent]['pose']
                        target_current_pose = each_other_agent_pose_array[i]
                        target_agent =Agent(x=target_current_pose[0],
                                             y=target_current_pose[1],
                                             yaw=target_current_pose[3],
                                             length=max(1, current_state['agent'][each_other_agent]['shape'][0][1]),
                                             width=max(1, current_state['agent'][each_other_agent]['shape'][0][0]),
                                             agent_id=each_other_agent)
                        has_collision = utils.check_collision(checking_agent=ego_agent,
                                                              target_agent=target_agent)
                        if ego_pose2_valid:
                            has_collision |= utils.check_collision(checking_agent=ego_agent2,
                                                                   target_agent=target_agent)
                        to_yield = False
                        if has_collision:
                            to_yield = True
                            # solve this conflict
                            found_in_loaded = False
                            if self.follow_loaded_relation:
                                detected_relation = []
                                for edge in current_state['edges']:
                                    if agent_id == edge[0] and each_other_agent == edge[1]:
                                        to_yield = False
                                        found_in_loaded = True
                                        break
                                current_state['predicting']['relation'] += [agent_id, each_other_agent]
                            if not found_in_loaded:
                                # FORWARD COLLISION CHECKINGS
                                target_pose_0 = each_other_agent_pose_array[current_frame_idx]
                                target_agent_0 =Agent(x=target_pose_0[0],
                                                       y=target_pose_0[1],
                                                       yaw=target_pose_0[3],
                                                       length=max(1, current_state['agent'][each_other_agent]['shape'][0][1]),
                                                       width=max(1, current_state['agent'][each_other_agent]['shape'][0][0]),
                                                       agent_id=each_other_agent)
                                collision_0 = utils.check_collision(ego_agent, target_agent_0)
                                if ego_pose2_valid:
                                    collision_0 |= utils.check_collision(ego_agent2, target_agent_0)
                                if collision_0:
                                    # yield
                                    detected_relation = [[each_other_agent, agent_id]]
                                else:
                                    # FCC backwards
                                    ego_agent_0 =Agent(
                                        x=interpolated_trajectory[0, 0],
                                        y=interpolated_trajectory[0, 1],
                                        yaw=interpolated_trajectory[0, 3],
                                        length=max(1, current_state['agent'][agent_id]['shape'][0][1]),
                                        width=max(1, current_state['agent'][agent_id]['shape'][0][0]),
                                        agent_id=agent_id)
                                    collision_back = utils.check_collision(ego_agent_0, target_agent)
                                    if collision_back:
                                        # not yield
                                        detected_relation = [[agent_id, each_other_agent]]
                                    else:
                                        # check relation
                                        self.online_predictor.predict_one_time(each_pair=[agent_id, each_other_agent],
                                                                                    current_frame=current_frame_idx,
                                                                                    clear_history=True,
                                                                                    current_data=current_state)
                                        detected_relation = current_state['predicting']['relation']

                                # data to save
                                if 'relations_per_frame_env' not in current_state['predicting']:
                                    current_state['predicting']['relations_per_frame_env'] = {}
                                for dt in range(self.planning_interval):
                                    if (current_frame_idx + dt) not in current_state['predicting']['relations_per_frame_env']:
                                        current_state['predicting']['relations_per_frame_env'][current_frame_idx + dt] = []
                                    current_state['predicting']['relations_per_frame_env'][current_frame_idx + dt] += detected_relation

                                if [agent_id, each_other_agent] in detected_relation:
                                    if [each_other_agent, agent_id] in detected_relation:
                                        # bi-directional relations, still yield
                                        pass
                                    else:
                                        my_reactors.append(each_other_agent)
                                        to_yield = False

                        if to_yield:
                            earliest_collision_idx = ego_index_checking
                            collision_point = [ego_pose[0], ego_pose[1]]
                            earliest_target_agent = each_other_agent
                            if abs(each_other_agent_pose_array[i, 0] + 1) < 0.1 or abs(each_other_agent_pose_array[i-5, 0] + 1) < 0.1:
                                target_speed = 0
                            else:
                                target_speed = utils.euclidean_distance(each_other_agent_pose_array[i, :2], each_other_agent_pose_array[i-5, :2]) / 5
                            break
                    if earliest_collision_idx is not None:
                        break

                if earliest_collision_idx is not None or self.method_testing < 2:
                    distance_to_travel = my_interpolator.get_distance_with_index(earliest_collision_idx) - S0
                    stopping_point = my_interpolator.interpolate(max(0, distance_to_travel - S0))[:2]
                    if utils.euclidean_distance(interpolated_trajectory[0, :2],
                                          stopping_point) < MINIMAL_DISTANCE_TO_TRAVEL or distance_to_travel < MINIMAL_DISTANCE_TO_TRAVEL or my_current_v_per_step < 0.1:
                        planed_traj = self.get_trajectory_from_interpolator(my_interpolator=my_interpolator,
                                                                            my_current_speed=my_current_v_per_step,
                                                                            desired_speed=my_target_speed,
                                                                            emergency_stop=True)
                        agents_dic_copy[agent_id]['action'] = 'stop'
                    else:
                        planed_traj = self.adjust_speed_for_collision(interpolator=my_interpolator,
                                                                      distance_to_end=distance_to_travel,
                                                                      current_v=my_current_v_per_step,
                                                                      end_point_v=min(my_current_v_per_step * 0.8,
                                                                                      target_speed))
                        assert len(planed_traj.shape) > 1, planed_traj.shape
                        agents_dic_copy[agent_id]['action'] = 'yield'

                    # print("Yielding log: ", agent_id, each_other_agent, earliest_target_agent, earliest_collision_idx, distance_to_travel)
                else:
                    # no conflicts to yield
                    if utils.euclidean_distance(interpolated_trajectory[0, :2], interpolated_trajectory[-1, :2]) < MINIMAL_DISTANCE_TO_TRAVEL:
                        planed_traj = self.get_trajectory_from_interpolator(my_interpolator=my_interpolator,
                                                                            my_current_speed=my_current_v_per_step,
                                                                            desired_speed=my_target_speed,
                                                                            hold_still=True)
                    else:
                        planed_traj = interpolated_trajectory
                    agents_dic_copy[agent_id]['action'] = 'controlled'

                if self.test_task == 1:
                    plan_for_ego = True
                if not plan_for_ego and ego_id == agent_id:
                    agents_dic_copy[agent_id]['action'] = None
                else:
                    if self.test_task != 2:
                        if collision_point is not None:
                            current_state['predicting']['points_to_mark'].append(collision_point)
                        current_state['predicting']['trajectory_to_mark'].append(planed_traj)

                    # if agent_id == 181:
                    #     for each_traj in prediction_traj_dic_m[agent_id]['rst']:
                    #         current_state['predicting']['trajectory_to_mark'].append(each_traj)

                    # replace the trajectory
                    planning_horizon, _ = planed_traj.shape
                    agents_dic_copy[agent_id]['pose'][current_frame_idx:planning_horizon+current_frame_idx, :] = planed_traj[:total_time_frame - current_frame_idx, :]
            current_state['agent'] = agents_dic_copy
        return current_state

    def trajectory_from_cubic_BC(self, p1, p2, p3, p4, v):
        """
        Generates a trajectory based on a cubic Bezier curve with given control points and a target speed.

        Args:
            p1 (list): The starting point of the curve [x1, y1].
            p2 (list): The first control point [x2, y2].
            p3 (list): The second control point [x3, y3].
            p4 (list): The ending point of the curve [x4, y4].
            v (float): The target speed (in m/s) for generating the trajectory.

        Returns:
            np.array: A numpy array containing the generated trajectory points.
        """
        # form a Bezier Curve
        total_dist = utils.euclidean_distance(p4, p1)
        total_t = min(93, int(total_dist/max(1, v)))
        traj_to_return = []
        for i in range(total_t):
            if i >= 92:
                break
            t = (i+1)/total_t
            p0_x = pow((1 - t), 3) * p1[0]
            p0_y = pow((1 - t), 3) * p1[1]
            p1_x = 3 * pow((1 - t), 2) * t * p2[0]
            p1_y = 3 * pow((1 - t), 2) * t * p2[1]
            p2_x = 3 * (1 - t) * pow(t, 2) * p3[0]
            p2_y = 3 * (1 - t) * pow(t, 2) * p3[1]
            p3_x = pow(t, 3) * p4[0]
            p3_y = pow(t, 3) * p4[1]
            traj_to_return.append((p0_x+p1_x+p2_x+p3_x, p0_y+p1_y+p2_y+p3_y))
        return np.array(traj_to_return, ndmin=2)

    def assert_traj(self, traj):
        """
        Validates the smoothness of the trajectory by checking the distance between consecutive points.

        Args:
            traj (np.array): The trajectory to validate in the form of a numpy array.

        Returns:
            int: Returns -1 if the trajectory is smooth throughout or the index of the point where a 
            significant jump occurred, indicating a potential issue with the trajectory.
        """
        total_time, _ = traj.shape
        if total_time < 30:
            return -1
        for i in range(total_time):
            if i == 0:
                continue
            if i >= total_time - 3 or i >= 20:
                break
            dist_1 = utils.euclidean_distance(traj[6+i, :2], traj[1+i, :2]) / 5
            dist_2 = utils.euclidean_distance(traj[5+i, :2], traj[i, :2]) / 5
            if abs(dist_1 - dist_2) > 5.0/self.frame_rate:
                print("Warning: frame jumping at: ", i, abs(dist_1 - dist_2))
                return i
        return -1

class SudoInterpolator:
    """
    A class used for interpolating trajectories within a given framework, allowing the calculation of positions
    along a trajectory at specified distances.
    """
    def __init__(self, trajectory, current_pose):
        """
        This function initializes the SudoInterpolator with a trajectory and a current pose.

        Args:
            trajectory (array-like): The trajectory data points to interpolate.
            current_pose (array-like): The current pose of the agent, typically [x, y, z, yaw].
        """
        self.trajectory = trajectory
        self.current_pose = current_pose

    def interpolate(self, distance: float, starting_from=None, debug=False):
        """
        This function interpolates the trajectory to find the pose at a specified distance from the current pose.

        Args:
            distance (float): The distance along the trajectory to find the corresponding pose.
            starting_from (int, optional): The index to start interpolation from; defaults to None, indicating no implementation yet.
            debug (bool, optional): Flag to enable debug mode for detailed logging.

        Returns:
            list: The pose [x, y, z, yaw] vector at the specified distance along the trajectory.
        """
        if starting_from is not None:
            assert False, 'not implemented'
        else:
            pose = self.trajectory.copy()
        if distance <= MINIMAL_DISTANCE_PER_STEP:
            return self.current_pose
        if pose.shape is None or len(pose.shape) < 2:
            return self.current_pose
        total_frame, _ = pose.shape
        # assert distance > 0, distance
        distance_input = distance
        for i in range(total_frame):
            if i == 0:
                pose1 = self.current_pose[:2]
                pose2 = pose[0, :2]
            else:
                pose1 = pose[i - 1, :2]
                pose2 = pose[i, :2]
            next_step = utils.euclidean_distance(pose1, pose2)
            if debug:
                print(f"{i} {next_step} {distance} {total_frame} {self.current_pose}")
            if next_step >= MINIMAL_DISTANCE_PER_STEP:
                if distance > next_step and i != total_frame - 1:
                    distance -= next_step
                    continue
                else:
                    return self.get_state_from_poses(pose1, pose2, distance, next_step)
        if distance_input - 2 > distance:
            # hide it outshoot
            # logging.warning(f'Over shooting while planning!!!!!!!!!')
            return self.get_state_from_poses(pose1, pose2, distance, next_step)
        else:
            # return current pose if trajectory not moved at all
            return self.current_pose
            # pose1 = self.current_pose[:2]
            # pose2 = pose[0, :2]
            # return self.get_state_from_poses(pose1, pose2, 0, 0.001)

    def get_state_from_poses(self, pose1, pose2, mul, divider):
        """
        This function calculates the intermediate state between two poses based on a multiplier and divider.

        Args:
            pose1 (list): The first pose [x, y, z, yaw].
            pose2 (list): The second pose [x, y, z, yaw].
            mul (float): The multiplier to adjust the position between the two poses.
            divider (float): The divisor used in the multiplication to calculate the intermediate position.

        Returns:
            list: The calculated intermediate state [x, y, z, yaw].
        """
        x = (pose2[0] - pose1[0]) * mul / (divider + 0.0001) + pose1[0]
        y = (pose2[1] - pose1[1]) * mul / (divider + 0.0001) + pose1[1]
        yaw = utils.normalize_angle(plan_helper.get_angle_of_a_line(pt1=pose1, pt2=pose2))
        return [x, y, 0, yaw]

    def get_distance_with_index(self, index: int):
        """
        This function calculates the cumulative distance up to a given index in the trajectory.

        Args:
            index (int): The index in the trajectory up to which the cumulative distance is calculated.

        Returns:
            float: The cumulative distance up to the specified index.
        """
        distance = 0
        if index != 0:
            pose = self.trajectory.copy()
            total_frame, _ = pose.shape
            for i in range(total_frame):
                if i >= index != -1:
                    # pass -1 to travel through all indices
                    break
                elif i == 0:
                    step = utils.euclidean_distance(self.current_pose[:2], pose[i, :2])
                else:
                    step = utils.euclidean_distance(pose[i, :2], pose[i-1, :2])
                if step > MINIMAL_DISTANCE_PER_STEP:
                    distance += step
        return distance

    def get_speed_with_index(self, index: int):
        """
        This function gets the speed at a specific index in the trajectory based on the distance between consecutive points.

        Args:
            index (int): The index in the trajectory at which the speed is calculated.

        Returns:
            float or None: The speed at the specified index or None if the index is 0.
        """
        if index != 0:
            p_t = self.trajectory[index, :2]
            p_t1 = self.trajectory[index - 1, :2]
            speed_per_step = utils.euclidean_distance(p_t, p_t1)
            return speed_per_step
        else:
            return None